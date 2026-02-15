"""
Shadow Evaluation Pipeline.

Sends live traffic to shadow model versions asynchronously.
Logs predictions without affecting the primary response path.
"""

import os
import time
import logging

import httpx

logger = logging.getLogger("gateway.shadow")


class ShadowEvaluator:
    """Fire-and-forget shadow evaluation for model versions."""

    def __init__(self):
        self.enabled = os.environ.get("SHADOW_MODE_ENABLED", "false").lower() == "true"
        self.shadow_model = os.environ.get("SHADOW_MODEL", "pytorch")
        self.service_urls = {
            "xgboost": os.environ.get("XGBOOST_SERVICE_URL", "http://xgboost-service:8001"),
            "pytorch": os.environ.get("PYTORCH_SERVICE_URL", "http://pytorch-service:8002"),
        }
        self.client = httpx.AsyncClient(timeout=3.0)
        self.shadow_log: list = []  # In-memory ring buffer for recent shadow results
        self.max_log_size = 1000

    async def evaluate_shadow(self, payload: dict, primary_result: dict):
        """Send request to shadow model and log result (fire-and-forget).

        This method should be called without awaiting if you want
        non-blocking behavior, or use asyncio.create_task().
        """
        if not self.enabled:
            return

        shadow_url = self.service_urls.get(self.shadow_model)
        if not shadow_url:
            logger.warning(f"Shadow model '{self.shadow_model}' URL not configured")
            return

        try:
            start = time.time()
            response = await self.client.post(
                f"{shadow_url}/predict",
                json=payload,
                timeout=3.0
            )
            response.raise_for_status()
            shadow_result = response.json()
            latency = (time.time() - start) * 1000

            log_entry = {
                "timestamp": time.time(),
                "shadow_model": self.shadow_model,
                "shadow_prediction": shadow_result.get("fraud_probability"),
                "shadow_is_fraud": shadow_result.get("is_fraud"),
                "primary_prediction": primary_result.get("fraud_probability"),
                "primary_is_fraud": primary_result.get("is_fraud"),
                "primary_model": primary_result.get("routed_to"),
                "agreement": (
                    shadow_result.get("is_fraud") == primary_result.get("is_fraud")
                ),
                "shadow_latency_ms": round(latency, 2),
            }

            self._append_log(log_entry)
            logger.debug(f"Shadow eval: agreement={log_entry['agreement']}, "
                        f"shadow_p={log_entry['shadow_prediction']:.4f}")

        except Exception as e:
            logger.warning(f"Shadow evaluation failed: {e}")

    def _append_log(self, entry: dict):
        """Append to ring buffer."""
        self.shadow_log.append(entry)
        if len(self.shadow_log) > self.max_log_size:
            self.shadow_log = self.shadow_log[-self.max_log_size:]

    def get_shadow_stats(self) -> dict:
        """Return shadow evaluation statistics."""
        if not self.shadow_log:
            return {"enabled": self.enabled, "shadow_model": self.shadow_model, "total_evaluations": 0}

        total = len(self.shadow_log)
        agreements = sum(1 for e in self.shadow_log if e.get("agreement"))
        avg_latency = sum(e.get("shadow_latency_ms", 0) for e in self.shadow_log) / total

        return {
            "enabled": self.enabled,
            "shadow_model": self.shadow_model,
            "total_evaluations": total,
            "agreement_rate": round(agreements / total, 4),
            "avg_shadow_latency_ms": round(avg_latency, 2),
            "recent_entries": self.shadow_log[-5:],
        }

    async def close(self):
        await self.client.aclose()
