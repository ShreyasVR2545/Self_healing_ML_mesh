"""
Fallback Heuristic Inference Service.

Rule-based fraud detection used when ML model services are unavailable.
Provides degraded but functional predictions as a safety net.
"""

import time
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="Fallback Fraud Detection Service", version="1.0.0")

# ── Prometheus Metrics ──
REQUEST_COUNT = Counter("fallback_request_total", "Total requests", ["status"])
REQUEST_LATENCY = Histogram(
    "fallback_request_latency_seconds", "Request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)
PREDICTION_COUNTER = Counter("fallback_predictions_total", "Predictions by class", ["prediction"])


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
    model_type: str = "fallback-heuristic"
    latency_ms: float


def heuristic_score(req: PredictionRequest) -> float:
    """Rule-based fraud scoring heuristic.

    Returns a pseudo-probability between 0 and 1 based on
    weighted rule triggers. This is NOT an ML model — it's a
    handcrafted safety net for when ML services are down.

    Rules:
      - High amount (> $3000): +0.25
      - Foreign transaction: +0.20
      - Night transaction (23:00-05:00): +0.15
      - High velocity (> 5 txns/hour): +0.15
      - Large distance from home (> 100km): +0.10
      - Amount >> historical average (> 5x): +0.15
      - New card (< 30 days): +0.10
    """
    score = 0.0

    if req.amount > 3000:
        score += 0.25
    elif req.amount > 1000:
        score += 0.10

    if req.is_foreign:
        score += 0.20

    is_night = req.is_night if req.is_night is not None else int(
        req.hour_of_day >= 23 or req.hour_of_day <= 5
    )
    if is_night:
        score += 0.15

    if req.velocity_last_1h > 5:
        score += 0.15
    elif req.velocity_last_1h > 3:
        score += 0.05

    if req.distance_from_home > 100:
        score += 0.10

    ratio = req.amount_to_avg_ratio or (
        req.amount / max(req.avg_amount_last_7d, 1)
    )
    if ratio > 5:
        score += 0.15
    elif ratio > 3:
        score += 0.05

    if req.card_age_days < 30:
        score += 0.10

    return min(score, 1.0)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()

    score = heuristic_score(request)
    is_fraud = score >= 0.5
    latency_ms = (time.time() - start_time) * 1000

    REQUEST_COUNT.labels(status="success").inc()
    REQUEST_LATENCY.observe(time.time() - start_time)
    PREDICTION_COUNTER.labels(prediction="fraud" if is_fraud else "legit").inc()

    return PredictionResponse(
        fraud_probability=round(score, 6),
        is_fraud=is_fraud,
        model_version="heuristic_v1",
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
async def health():
    return {"status": "healthy", "model_version": "heuristic_v1", "model_type": "fallback-heuristic"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)  # nosec B104
