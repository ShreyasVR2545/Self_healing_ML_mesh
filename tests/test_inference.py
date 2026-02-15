"""
Unit tests for inference services.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFeatureEngineering:
    """Tests for feature_engineering module."""

    def test_feature_columns_defined(self):
        from training.feature_engineering import FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 14
        assert "amount" in FEATURE_COLUMNS
        assert "is_fraud" not in FEATURE_COLUMNS

    def test_preprocess_dataframe(self):
        import pandas as pd
        from training.feature_engineering import preprocess_dataframe

        df = pd.DataFrame({
            "amount": [100.0],
            "merchant_category": [1],
            "hour_of_day": [14],
            "day_of_week": [2],
            "distance_from_home": [5.0],
            "distance_from_last_txn": [3.0],
            "is_foreign": [0],
            "velocity_last_1h": [1],
            "velocity_last_24h": [5],
            "avg_amount_last_7d": [80.0],
            "card_age_days": [365],
        })

        result = preprocess_dataframe(df)
        assert "amount_to_avg_ratio" in result.columns
        assert "is_weekend" in result.columns
        assert "is_night" in result.columns

    def test_request_to_dataframe(self):
        from training.feature_engineering import request_to_dataframe, FEATURE_COLUMNS

        request = {
            "amount": 150.0,
            "merchant_category": 3,
            "hour_of_day": 2,
            "day_of_week": 6,
            "distance_from_home": 25.0,
            "distance_from_last_txn": 10.0,
            "is_foreign": 1,
            "velocity_last_1h": 3,
            "velocity_last_24h": 8,
            "avg_amount_last_7d": 100.0,
            "card_age_days": 60,
        }

        df = request_to_dataframe(request)
        assert len(df) == 1
        assert all(col in df.columns for col in FEATURE_COLUMNS)
        assert df["is_night"].values[0] == 1  # hour_of_day = 2
        assert df["is_weekend"].values[0] == 1  # day_of_week = 6

    def test_extract_features_shape(self):
        import pandas as pd
        from training.feature_engineering import extract_features, FEATURE_COLUMNS

        df = pd.DataFrame({
            "amount": [100.0, 200.0],
            "merchant_category": [1, 2],
            "hour_of_day": [14, 3],
            "day_of_week": [2, 5],
            "distance_from_home": [5.0, 50.0],
            "distance_from_last_txn": [3.0, 20.0],
            "is_foreign": [0, 1],
            "velocity_last_1h": [1, 5],
            "velocity_last_24h": [5, 15],
            "avg_amount_last_7d": [80.0, 150.0],
            "card_age_days": [365, 10],
            "is_fraud": [0, 1],
        })

        features = extract_features(df)
        assert features.shape == (2, len(FEATURE_COLUMNS))
        assert features.dtype == np.float32


class TestDatasetGenerator:
    """Tests for synthetic dataset generation."""

    def test_generate_transactions(self):
        from data.generate_dataset import generate_transactions

        df = generate_transactions(n_samples=1000, fraud_ratio=0.05, seed=42)
        assert len(df) == 1000
        assert "is_fraud" in df.columns
        fraud_ratio = df["is_fraud"].mean()
        assert 0.03 < fraud_ratio < 0.07  # Within tolerance

    def test_generate_transactions_features(self):
        from data.generate_dataset import generate_transactions

        df = generate_transactions(n_samples=100, seed=0)
        expected_cols = [
            "amount", "merchant_category", "hour_of_day", "day_of_week",
            "distance_from_home", "distance_from_last_txn", "is_foreign",
            "velocity_last_1h", "velocity_last_24h", "avg_amount_last_7d",
            "amount_to_avg_ratio", "is_weekend", "is_night", "card_age_days",
            "is_fraud",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestFallbackHeuristic:
    """Tests for the fallback heuristic scoring."""

    def test_low_risk_transaction(self):
        from inference.fallback_service.app import heuristic_score, PredictionRequest

        req = PredictionRequest(
            amount=25.0, merchant_category=1, hour_of_day=14,
            day_of_week=2, distance_from_home=3.0, distance_from_last_txn=1.0,
            is_foreign=0, velocity_last_1h=1, velocity_last_24h=3,
            avg_amount_last_7d=30.0, card_age_days=500
        )
        score = heuristic_score(req)
        assert score < 0.5, f"Low-risk transaction scored {score}"

    def test_high_risk_transaction(self):
        from inference.fallback_service.app import heuristic_score, PredictionRequest

        req = PredictionRequest(
            amount=8000.0, merchant_category=7, hour_of_day=2,
            day_of_week=6, distance_from_home=200.0, distance_from_last_txn=150.0,
            is_foreign=1, velocity_last_1h=8, velocity_last_24h=20,
            avg_amount_last_7d=100.0, card_age_days=10
        )
        score = heuristic_score(req)
        assert score >= 0.5, f"High-risk transaction scored {score}"
