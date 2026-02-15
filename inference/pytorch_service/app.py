"""
PyTorch Model Inference Service.

FastAPI microservice serving the PyTorch MLP fraud detection model.
"""

import os
import sys
import time
import json
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from training.feature_engineering import request_to_dataframe, extract_features, FEATURE_COLUMNS

# ── App ──
app = FastAPI(title="PyTorch Fraud Detection Service", version="1.0.0")

# ── Prometheus Metrics ──
REQUEST_COUNT = Counter("pytorch_request_total", "Total requests", ["status"])
REQUEST_LATENCY = Histogram(
    "pytorch_request_latency_seconds", "Request latency",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
PREDICTION_COUNTER = Counter("pytorch_predictions_total", "Predictions by class", ["prediction"])
MODEL_LOADED = Gauge("pytorch_model_loaded", "Whether model is loaded")

# ── Model ──
MODEL = None
SCALER_PARAMS = None
MODEL_VERSION = os.environ.get("MODEL_VERSION", "pytorch_v1")
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/pytorch_v1.pt")


class FraudMLP(torch.nn.Module):
    """Mirror of training model architecture."""

    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, h_dim),
                torch.nn.BatchNorm1d(h_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
            ])
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, 1))
        layers.append(torch.nn.Sigmoid())
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


@app.on_event("startup")
async def load_model():
    global MODEL, SCALER_PARAMS
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        model = FraudMLP(
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        MODEL = model
        SCALER_PARAMS = checkpoint.get("scaler_params")
        MODEL_LOADED.set(1)
        print(f"Loaded PyTorch model from {MODEL_PATH}")
    except Exception as e:
        MODEL_LOADED.set(0)
        print(f"FAILED to load model: {e}")


def scale_features(features: np.ndarray) -> np.ndarray:
    """Apply stored StandardScaler transform."""
    if SCALER_PARAMS is None:
        return features
    mean = np.array(SCALER_PARAMS["mean"])
    scale = np.array(SCALER_PARAMS["scale"])
    return (features - mean) / scale


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
    model_type: str = "pytorch"
    latency_ms: float


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if MODEL is None:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    try:
        df = request_to_dataframe(request.model_dump())
        features = extract_features(df)
        features_scaled = scale_features(features)

        with torch.no_grad():
            tensor = torch.FloatTensor(features_scaled)
            proba = float(MODEL(tensor).item())

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
    return {"status": "healthy", "model_version": MODEL_VERSION, "model_type": "pytorch"}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
