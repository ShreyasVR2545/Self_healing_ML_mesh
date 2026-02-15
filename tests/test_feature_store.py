"""
Unit tests for the Feature Store.
"""

import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestFeatureStore:
    """Tests for the Redis-backed feature store (mocked)."""

    def test_init_defaults(self):
        with patch.dict(os.environ, {"REDIS_HOST": "testhost", "REDIS_PORT": "1234"}):
            from feature_store.store import FeatureStore
            store = FeatureStore()
            assert store.redis_host == "testhost"
            assert store.redis_port == 1234

    def test_log_features_called(self):
        from feature_store.store import FeatureStore

        store = FeatureStore()
        mock_client = MagicMock()
        store._client = mock_client

        features = {"amount": 100.0, "is_foreign": 1}
        store.log_features("req-001", features, prediction=0.85, model_version="xgboost_v1")

        mock_client.xadd.assert_called_once()
        mock_client.setex.assert_called_once()

    def test_get_recent_features_empty(self):
        from feature_store.store import FeatureStore

        store = FeatureStore()
        mock_client = MagicMock()
        mock_client.xrange.return_value = []
        store._client = mock_client

        results = store.get_recent_features(window_minutes=60)
        assert results == []
