"""
Shared feature engineering pipeline.

Used consistently across training and inference to ensure
feature parity between train-time and serve-time.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


# Canonical feature order used by all models
FEATURE_COLUMNS: List[str] = [
    "amount",
    "merchant_category",
    "hour_of_day",
    "day_of_week",
    "distance_from_home",
    "distance_from_last_txn",
    "is_foreign",
    "velocity_last_1h",
    "velocity_last_24h",
    "avg_amount_last_7d",
    "amount_to_avg_ratio",
    "is_weekend",
    "is_night",
    "card_age_days",
]

TARGET_COLUMN: str = "is_fraud"


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transforms to a DataFrame.

    Ensures derived features exist and types are correct.
    """
    df = df.copy()

    # Ensure derived features
    if "amount_to_avg_ratio" not in df.columns:
        df["amount_to_avg_ratio"] = (df["amount"] / df["avg_amount_last_7d"].clip(lower=1)).clip(0, 50)

    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    if "is_night" not in df.columns:
        df["is_night"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)

    # Clip outliers
    df["amount"] = df["amount"].clip(0, 50000)
    df["distance_from_home"] = df["distance_from_home"].clip(0, 1000)
    df["distance_from_last_txn"] = df["distance_from_last_txn"].clip(0, 1000)
    df["velocity_last_1h"] = df["velocity_last_1h"].clip(0, 50)
    df["velocity_last_24h"] = df["velocity_last_24h"].clip(0, 200)

    return df


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix in canonical order."""
    df = preprocess_dataframe(df)
    return df[FEATURE_COLUMNS].values.astype(np.float32)


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    """Extract target labels."""
    return df[TARGET_COLUMN].values.astype(np.float32)


def request_to_dataframe(request_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert a single prediction request dict to a DataFrame row.

    Used by inference services to transform incoming API requests.
    """
    row = {}
    for col in FEATURE_COLUMNS:
        if col in request_data:
            row[col] = request_data[col]
        else:
            # Provide sensible defaults for derived features
            row[col] = _default_value(col, request_data)
    
    df = pd.DataFrame([row])
    return preprocess_dataframe(df)


def _default_value(col: str, data: Dict[str, Any]) -> Any:
    """Compute default value for a missing feature."""
    if col == "amount_to_avg_ratio":
        amount = data.get("amount", 0)
        avg = data.get("avg_amount_last_7d", 1)
        return min(amount / max(avg, 1), 50)
    elif col == "is_weekend":
        return int(data.get("day_of_week", 0) >= 5)
    elif col == "is_night":
        hour = data.get("hour_of_day", 12)
        return int(hour >= 23 or hour <= 5)
    return 0


def get_feature_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute per-feature statistics for drift reference."""
    df = preprocess_dataframe(df)
    stats = {}
    for col in FEATURE_COLUMNS:
        stats[col] = {
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
        }
    return stats
