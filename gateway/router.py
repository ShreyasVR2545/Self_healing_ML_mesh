"""
Traffic Router — Weighted routing with circuit breaker.

Directs incoming prediction requests to model services based on
configurable traffic splits. Implements circuit breaker pattern
to detect and route around unhealthy services.
"""

import os
import time
import random
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger("gateway.router")


class ServiceState(Enum):
    CLOSED = "closed"          # Healthy, accepting traffic
    OPEN = "open"              # Unhealthy, rejecting traffic
    HALF_OPEN = "half_open"    # Testing recovery


@dataclass
class CircuitBreaker:
    """Per-service circuit breaker state."""
    failure_count: int = 0
    last_failure_time: float = 0
    state: ServiceState = ServiceState.CLOSED
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds

    def record_success(self):
        self.failure_count = 0
        self.state = ServiceState.CLOSED

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = ServiceState.OPEN
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")

    def is_available(self) -> bool:
        if self.state == ServiceState.CLOSED:
            return True
        if self.state == ServiceState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = ServiceState.HALF_OPEN
                logger.info("Circuit breaker moving to HALF_OPEN for recovery probe")
                return True
            return False
        # HALF_OPEN: allow one test request
        return True


@dataclass
class ServiceConfig:
    """Configuration for a model service."""
    name: str
    url: str
    weight: float
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class TrafficRouter:
    """Weighted traffic router with circuit breaker and fallback."""

    def __init__(self):
        self.services: Dict[str, ServiceConfig] = {}
        self.fallback_url: str = os.environ.get("FALLBACK_SERVICE_URL", "http://fallback-service:8003")
        self.client = httpx.AsyncClient(timeout=5.0)
        self._setup_services()

    def _setup_services(self):
        """Initialize service configs from environment."""
        xgb_url = os.environ.get("XGBOOST_SERVICE_URL", "http://xgboost-service:8001")
        pt_url = os.environ.get("PYTORCH_SERVICE_URL", "http://pytorch-service:8002")
        xgb_weight = float(os.environ.get("TRAFFIC_SPLIT_XGBOOST", "0.8"))
        pt_weight = float(os.environ.get("TRAFFIC_SPLIT_PYTORCH", "0.2"))

        failure_threshold = int(os.environ.get("ROLLBACK_CONSECUTIVE_FAILURES", "5"))

        self.services["xgboost"] = ServiceConfig(
            name="xgboost", url=xgb_url, weight=xgb_weight,
            circuit_breaker=CircuitBreaker(failure_threshold=failure_threshold)
        )
        self.services["pytorch"] = ServiceConfig(
            name="pytorch", url=pt_url, weight=pt_weight,
            circuit_breaker=CircuitBreaker(failure_threshold=failure_threshold)
        )

    def select_service(self) -> Optional[ServiceConfig]:
        """Select a service based on weighted routing, respecting circuit breakers."""
        available = [
            s for s in self.services.values()
            if s.circuit_breaker.is_available() and s.weight > 0
        ]

        if not available:
            logger.warning("No healthy ML services available, using fallback")
            return None

        weights = [s.weight for s in available]
        total = sum(weights)
        normalized = [w / total for w in weights]

        return random.choices(available, weights=normalized, k=1)[0]

    async def route_request(self, payload: dict) -> dict:
        """Route a prediction request to the selected service.

        Returns the prediction response dict including routing metadata.
        """
        service = self.select_service()

        if service is None:
            return await self._call_fallback(payload)

        try:
            response = await self.client.post(
                f"{service.url}/predict",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            result = response.json()
            service.circuit_breaker.record_success()
            result["routed_to"] = service.name
            return result

        except Exception as e:
            service.circuit_breaker.record_failure()
            logger.error(f"Service {service.name} failed: {e}")

            # Try another available service
            other = [s for s in self.services.values()
                     if s.name != service.name and s.circuit_breaker.is_available()]
            if other:
                try:
                    alt = other[0]
                    response = await self.client.post(
                        f"{alt.url}/predict", json=payload, timeout=5.0
                    )
                    response.raise_for_status()
                    result = response.json()
                    alt.circuit_breaker.record_success()
                    result["routed_to"] = alt.name
                    result["fallback_reason"] = f"{service.name} failed"
                    return result
                except Exception as e2:
                    alt.circuit_breaker.record_failure()
                    logger.error(f"Alternate service {alt.name} also failed: {e2}")

            return await self._call_fallback(payload)

    async def _call_fallback(self, payload: dict) -> dict:
        """Call the heuristic fallback service."""
        try:
            response = await self.client.post(
                f"{self.fallback_url}/predict",
                json=payload,
                timeout=5.0
            )
            response.raise_for_status()
            result = response.json()
            result["routed_to"] = "fallback"
            result["fallback_reason"] = "all ML services unavailable"
            return result
        except Exception as e:
            logger.critical(f"Fallback service also failed: {e}")
            return {
                "fraud_probability": 0.5,
                "is_fraud": True,  # Conservative: flag as fraud when all services down
                "model_version": "emergency_fallback",
                "model_type": "hardcoded",
                "routed_to": "emergency",
                "fallback_reason": "all services down, conservative default",
                "latency_ms": 0,
            }

    def get_service_states(self) -> Dict[str, dict]:
        """Return current state of all services for monitoring."""
        states = {}
        for name, svc in self.services.items():
            states[name] = {
                "url": svc.url,
                "weight": svc.weight,
                "circuit_state": svc.circuit_breaker.state.value,
                "failure_count": svc.circuit_breaker.failure_count,
            }
        states["fallback"] = {"url": self.fallback_url, "weight": 0, "circuit_state": "always_on"}
        return states

    def update_weights(self, weights: Dict[str, float]):
        """Dynamically update traffic split weights."""
        for name, weight in weights.items():
            if name in self.services:
                self.services[name].weight = weight
                logger.info(f"Updated {name} weight to {weight}")

    async def close(self):
        await self.client.aclose()
