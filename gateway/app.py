"""
API Gateway — Single entry point for the ML Mesh.

Routes prediction requests through the traffic router,
applies shadow evaluation, exposes Prometheus metrics,
and manages the rollback lifecycle.
"""

import os
import sys
import time
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

from router import TrafficRouter
from shadow import ShadowEvaluator
from rollback import RollbackManager

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("gateway")

# ── Components ──
router = TrafficRouter()
shadow = ShadowEvaluator()
rollback_mgr = RollbackManager(router)


# ── Lifespan ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API Gateway...")
    await rollback_mgr.start()
    yield
    logger.info("Shutting down API Gateway...")
    await router.close()
    await shadow.close()
    await rollback_mgr.stop()


# ── App ──
app = FastAPI(
    title="ML Mesh API Gateway",
    version="1.0.0",
    description="Self-healing ML microservice gateway with canary routing, "
                "shadow evaluation, and automatic rollback.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus Metrics ──
GATEWAY_REQUESTS = Counter(
    "gateway_requests_total", "Total gateway requests", ["status", "routed_to"]
)
GATEWAY_LATENCY = Histogram(
    "gateway_request_latency_seconds", "End-to-end gateway latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
ACTIVE_REQUESTS = Gauge("gateway_active_requests", "Currently in-flight requests")
MODEL_TRAFFIC = Counter(
    "gateway_model_traffic_total", "Requests routed per model", ["model"]
)


# ── Schemas ──
class PredictionRequest(BaseModel):
    amount: float = Field(..., description="Transaction amount in USD")
    merchant_category: int = Field(..., ge=0, le=9)
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    distance_from_home: float = Field(..., ge=0)
    distance_from_last_txn: float = Field(..., ge=0)
    is_foreign: int = Field(..., ge=0, le=1)
    velocity_last_1h: int = Field(..., ge=0)
    velocity_last_24h: int = Field(..., ge=0)
    avg_amount_last_7d: float = Field(..., ge=0)
    card_age_days: int = Field(..., ge=0)
    amount_to_avg_ratio: Optional[float] = None
    is_weekend: Optional[int] = None
    is_night: Optional[int] = None


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model_version: str
    model_type: str
    latency_ms: float
    routed_to: str
    fallback_reason: Optional[str] = None


class TrafficUpdateRequest(BaseModel):
    xgboost: float = Field(..., ge=0, le=1)
    pytorch: float = Field(..., ge=0, le=1)


# ── Endpoints ──
@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint with routing, shadow, and metrics."""
    ACTIVE_REQUESTS.inc()
    start = time.time()

    try:
        payload = request.model_dump()
        result = await router.route_request(payload)

        routed_to = result.get("routed_to", "unknown")
        GATEWAY_REQUESTS.labels(status="success", routed_to=routed_to).inc()
        MODEL_TRAFFIC.labels(model=routed_to).inc()

        # Fire shadow evaluation (non-blocking)
        asyncio.create_task(shadow.evaluate_shadow(payload, result))

        total_latency = (time.time() - start) * 1000
        GATEWAY_LATENCY.observe(time.time() - start)

        return PredictionResponse(
            fraud_probability=result.get("fraud_probability", 0),
            is_fraud=result.get("is_fraud", False),
            model_version=result.get("model_version", "unknown"),
            model_type=result.get("model_type", "unknown"),
            latency_ms=round(total_latency, 2),
            routed_to=routed_to,
            fallback_reason=result.get("fallback_reason"),
        )

    except Exception as e:
        GATEWAY_REQUESTS.labels(status="error", routed_to="none").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health")
async def health():
    """Gateway health check."""
    return {
        "status": "healthy",
        "services": router.get_service_states(),
    }


@app.get("/api/v1/status")
async def status():
    """Full system status: routing, shadow, and rollback info."""
    return {
        "services": router.get_service_states(),
        "shadow": shadow.get_shadow_stats(),
        "rollback": rollback_mgr.get_status(),
    }


@app.post("/api/v1/traffic")
async def update_traffic(update: TrafficUpdateRequest):
    """Dynamically update traffic split weights."""
    router.update_weights({"xgboost": update.xgboost, "pytorch": update.pytorch})
    return {
        "message": "Traffic weights updated",
        "new_weights": {"xgboost": update.xgboost, "pytorch": update.pytorch},
    }


@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
