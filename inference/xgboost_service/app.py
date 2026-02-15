"""
XGBoost Model Inference Service.

FastAPI microservice serving the XGBoost fraud detection model.
Exposes /predict and /health endpoints with Prometheus metrics.
"""

import os
import sys
import time
import json
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
)
from starlette.responses import Response

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.feature_engineering import request_to_dataframe, extract_features, FEATURE_COLUMNS

# ── App ──
app = FastAPI(title="XGBoost Fraud Detection Service", version="1.0.0")

# ── Prometheus Metrics ──
REQUEST_COUNT = Counter(
    "xgboost_request_total", "Total prediction requests", ["status"]
)
REQUEST_LATENCY = Histogram(
    "xgboost_request_latency_seconds", "Request latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_COUNTER = Counter(
    "xgboost_predictions_total", "Predictions by class", ["prediction"]
)
MODEL_LOADED = Gauge("xgboost_model_loaded", "Whether model is loaded")

# ── Model Loading ──
MODEL = None
MODEL_VERSION = os.environ.get("MODEL_VERSION", "xgboost_v1")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/xgboost_v1.json")


@app.on_event("startup")
async def load_model():
    global MODEL
    try:
        MODEL = xgb.XGBClassifier()
        MODEL.load_model(MODEL_PATH)
        MODEL_LOADED.set(1)
        print(f"Loaded XGBoost model from {MODEL_PATH}")
    except Exception as e:
        MODEL_LOADED.set(0)
        print(f"FAILED to load model: {e}")


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
    model_type: str = "xgboost"
    latency_ms: float


# ── Endpoints ──
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if MODEL is None:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    try:
        df = request_to_dataframe(request.model_dump())
        features = extract_features(df)
        proba = float(MODEL.predict_proba(features)[:, 1][0])
        is_fraud = proba >= 0.5
        latency_ms = (time.time() - start_time) * 1000

        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        PREDICTION_COUNTER.labels(prediction="fraud" if is_fraud else "legit").inc()

        return PredictionResponse(
            fraud_probability=round(proba, 6),
            is_fraud=is_fraud,
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms, 2),
        )
    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": MODEL_VERSION, "model_type": "xgboost"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
