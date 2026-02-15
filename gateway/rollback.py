"""
Health Check & Auto-Rollback Manager.

Periodically checks canary model metrics and automatically
rolls back traffic if health degrades beyond thresholds.
"""

import os
import time
import asyncio
import logging
from typing import Dict

import httpx

logger = logging.getLogger("gateway.rollback")


class RollbackManager:
    """Monitors canary metrics and triggers automatic rollback."""

    def __init__(self, router):
        self.router = router
        self.error_threshold = float(os.environ.get("CANARY_ERROR_THRESHOLD", "0.1"))
        self.latency_threshold_ms = float(os.environ.get("CANARY_LATENCY_P95_THRESHOLD", "500"))
        self.check_interval = int(os.environ.get("HEALTH_CHECK_INTERVAL_SECONDS", "10"))

        # Track per-service metrics
        self.metrics: Dict[str, dict] = {}
        self.rollback_history: list = []
        self._client = httpx.AsyncClient(timeout=5.0)
        self._running = False

    async def start(self):
        """Start background health check loop."""
        self._running = True
        logger.info(f"Rollback manager started (interval={self.check_interval}s, "
                    f"error_threshold={self.error_threshold}, "
                    f"latency_threshold={self.latency_threshold_ms}ms)")
        asyncio.create_task(self._health_loop())

    async def stop(self):
        self._running = False
        await self._client.aclose()

    async def _health_loop(self):
        """Continuous health checking loop."""
        while self._running:
            try:
                await self._check_all_services()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            await asyncio.sleep(self.check_interval)

    async def _check_all_services(self):
        """Check health of all registered services."""
        for name, service in self.router.services.items():
            try:
                response = await self._client.get(
                    f"{service.url}/health", timeout=3.0
                )
                healthy = response.status_code == 200
                self._update_metrics(name, healthy=healthy, latency_ms=response.elapsed.total_seconds() * 1000)

                if not healthy:
                    logger.warning(f"Service {name} health check failed: {response.status_code}")

            except Exception as e:
                self._update_metrics(name, healthy=False, latency_ms=0)
                logger.warning(f"Service {name} unreachable: {e}")

        # Evaluate rollback conditions
        self._evaluate_rollback()

    def _update_metrics(self, service_name: str, healthy: bool, latency_ms: float):
        """Update rolling metrics for a service."""
        if service_name not in self.metrics:
            self.metrics[service_name] = {
                "total_checks": 0,
                "failures": 0,
                "latency_samples": [],
                "last_check": 0,
            }

        m = self.metrics[service_name]
        m["total_checks"] += 1
        if not healthy:
            m["failures"] += 1
        m["latency_samples"].append(latency_ms)
        m["last_check"] = time.time()

        # Keep rolling window of last 100 samples
        if len(m["latency_samples"]) > 100:
            m["latency_samples"] = m["latency_samples"][-100:]

    def _evaluate_rollback(self):
        """Check if any canary service needs rollback."""
        for name, m in self.metrics.items():
            if m["total_checks"] < 5:
                continue  # Not enough data

            error_rate = m["failures"] / m["total_checks"]
            latency_samples = m["latency_samples"]

            if latency_samples:
                sorted_latencies = sorted(latency_samples)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
            else:
                p95_latency = 0

            # Check thresholds
            should_rollback = False
            reason = []

            if error_rate > self.error_threshold:
                should_rollback = True
                reason.append(f"error_rate={error_rate:.2%} > {self.error_threshold:.2%}")

            if p95_latency > self.latency_threshold_ms:
                should_rollback = True
                reason.append(f"p95_latency={p95_latency:.0f}ms > {self.latency_threshold_ms}ms")

            if should_rollback:
                self._trigger_rollback(name, reason)

    def _trigger_rollback(self, service_name: str, reasons: list):
        """Roll back traffic from degraded service."""
        service = self.router.services.get(service_name)
        if not service or service.weight == 0:
            return  # Already rolled back

        reason_str = "; ".join(reasons)
        logger.critical(f"ROLLBACK triggered for {service_name}: {reason_str}")

        old_weight = service.weight

        # Shift all traffic to other healthy services
        service.weight = 0
        remaining_services = [
            s for s in self.router.services.values()
            if s.name != service_name and s.circuit_breaker.is_available()
        ]

        if remaining_services:
            extra_weight = old_weight / len(remaining_services)
            for s in remaining_services:
                s.weight += extra_weight
        else:
            logger.critical("No healthy services remaining — fallback only")

        self.rollback_history.append({
            "timestamp": time.time(),
            "service": service_name,
            "reason": reason_str,
            "old_weight": old_weight,
        })

        # Reset metrics for rolled-back service
        self.metrics[service_name] = {
            "total_checks": 0, "failures": 0,
            "latency_samples": [], "last_check": time.time()
        }

    def get_status(self) -> dict:
        """Return rollback manager status."""
        return {
            "running": self._running,
            "error_threshold": self.error_threshold,
            "latency_threshold_ms": self.latency_threshold_ms,
            "service_metrics": {
                name: {
                    "total_checks": m["total_checks"],
                    "error_rate": m["failures"] / max(m["total_checks"], 1),
                    "p95_latency_ms": (
                        sorted(m["latency_samples"])[int(len(m["latency_samples"]) * 0.95)]
                        if m["latency_samples"] else 0
                    ),
                }
                for name, m in self.metrics.items()
            },
            "rollback_history": self.rollback_history[-10:],
        }
