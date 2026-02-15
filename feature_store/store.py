"""
Redis-backed Online Feature Store.

Logs feature vectors from live prediction requests
and retrieves recent features for drift analysis.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any

import redis
import numpy as np

logger = logging.getLogger("feature_store")


class FeatureStore:
    """Redis-backed online feature store."""

    def __init__(self, redis_host: str = None, redis_port: int = None):
        self.redis_host = redis_host or os.environ.get("REDIS_HOST", "localhost")
        self.redis_port = redis_port or int(os.environ.get("REDIS_PORT", "6379"))
        self._client: Optional[redis.Redis] = None
        self.feature_key_prefix = "features:"
        self.feature_stream_key = "feature_stream"
        self.ttl_seconds = 86400  # 24 hours

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
        return self._client

    def log_features(self, request_id: str, features: Dict[str, Any],
                     prediction: float = None, model_version: str = None):
        """Log a feature vector from a live prediction request.

        Args:
            request_id: Unique identifier for this request
            features: Dictionary of feature name → value
            prediction: Model prediction (optional)
            model_version: Which model served this request
        """
        try:
            entry = {
                "request_id": request_id,
                "timestamp": time.time(),
                "features": json.dumps(features),
            }
            if prediction is not None:
                entry["prediction"] = str(prediction)
            if model_version:
                entry["model_version"] = model_version

            # Store in Redis stream for time-ordered retrieval
            self.client.xadd(
                self.feature_stream_key,
                entry,
                maxlen=100000  # Keep last 100k entries
            )

            # Also store individual feature for quick lookup
            key = f"{self.feature_key_prefix}{request_id}"
            self.client.setex(key, self.ttl_seconds, json.dumps(entry))

            logger.debug(f"Logged features for request {request_id}")

        except Exception as e:
            logger.error(f"Failed to log features: {e}")

    def get_recent_features(self, window_minutes: int = 60,
                            max_entries: int = 10000) -> List[Dict[str, float]]:
        """Retrieve recent feature vectors for drift analysis.

        Args:
            window_minutes: Time window to look back
            max_entries: Maximum entries to retrieve

        Returns:
            List of feature dictionaries
        """
        try:
            # Calculate timestamp threshold
            min_time = int((time.time() - window_minutes * 60) * 1000)

            # Read from stream
            entries = self.client.xrange(
                self.feature_stream_key,
                min=str(min_time),
                max="+",
                count=max_entries
            )

            features_list = []
            for entry_id, data in entries:
                try:
                    features = json.loads(data.get("features", "{}"))
                    features_list.append(features)
                except json.JSONDecodeError:
                    continue

            logger.info(f"Retrieved {len(features_list)} feature vectors "
                       f"from last {window_minutes} minutes")
            return features_list

        except Exception as e:
            logger.error(f"Failed to retrieve features: {e}")
            return []

    def get_feature_arrays(self, feature_names: List[str],
                           window_minutes: int = 60) -> Dict[str, np.ndarray]:
        """Get feature arrays suitable for drift detection.

        Returns:
            Dictionary mapping feature name → numpy array of values
        """
        records = self.get_recent_features(window_minutes)
        if not records:
            return {}

        arrays = {}
        for fname in feature_names:
            values = [r.get(fname) for r in records if fname in r]
            if values:
                arrays[fname] = np.array(values, dtype=float)

        return arrays

    def get_stream_info(self) -> dict:
        """Return metadata about the feature stream."""
        try:
            info = self.client.xinfo_stream(self.feature_stream_key)
            return {
                "length": info.get("length", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
            }
        except Exception:
            return {"length": 0, "error": "stream not found"}

    def close(self):
        if self._client:
            self._client.close()
