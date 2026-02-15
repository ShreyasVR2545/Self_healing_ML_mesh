"""
Unit tests for the API Gateway routing logic.
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))


class TestCircuitBreaker:
    """Tests for circuit breaker logic."""

    def test_initial_state_closed(self):
        from gateway.router import CircuitBreaker
        cb = CircuitBreaker()
        assert cb.is_available()

    def test_opens_after_threshold(self):
        from gateway.router import CircuitBreaker, ServiceState
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.is_available()  # Still closed
        cb.record_failure()
        assert not cb.is_available()  # Now open
        assert cb.state == ServiceState.OPEN

    def test_resets_on_success(self):
        from gateway.router import CircuitBreaker
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.is_available()

    def test_half_open_after_timeout(self):
        import time
        from gateway.router import CircuitBreaker, ServiceState

        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        cb.record_failure()
        assert not cb.is_available()

        time.sleep(0.15)
        assert cb.is_available()  # Should be half-open now
        assert cb.state == ServiceState.HALF_OPEN


class TestTrafficRouter:
    """Tests for weighted traffic routing."""

    def test_select_service_weighted(self):
        from gateway.router import TrafficRouter

        with patch.dict(os.environ, {
            "XGBOOST_SERVICE_URL": "http://localhost:8001",
            "PYTORCH_SERVICE_URL": "http://localhost:8002",
            "TRAFFIC_SPLIT_XGBOOST": "1.0",
            "TRAFFIC_SPLIT_PYTORCH": "0.0",
        }):
            router = TrafficRouter()
            service = router.select_service()
            assert service is not None
            assert service.name == "xgboost"

    def test_fallback_when_all_open(self):
        import time
        from gateway.router import TrafficRouter, ServiceState

        router = TrafficRouter()
        for svc in router.services.values():
            svc.circuit_breaker.failure_count = 100
            svc.circuit_breaker.state = ServiceState.OPEN
            svc.circuit_breaker.last_failure_time = time.time()  # Recent failure prevents HALF_OPEN

        result = router.select_service()
        assert result is None  # Should return None, triggering fallback

    def test_get_service_states(self):
        from gateway.router import TrafficRouter
        router = TrafficRouter()
        states = router.get_service_states()
        assert "xgboost" in states
        assert "pytorch" in states
        assert "fallback" in states

    def test_update_weights(self):
        from gateway.router import TrafficRouter
        router = TrafficRouter()
        router.update_weights({"xgboost": 0.5, "pytorch": 0.5})
        assert router.services["xgboost"].weight == 0.5
        assert router.services["pytorch"].weight == 0.5


class TestShadowEvaluator:
    """Tests for shadow evaluation."""

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {"SHADOW_MODE_ENABLED": "false"}):
            from gateway.shadow import ShadowEvaluator
            shadow = ShadowEvaluator()
            assert not shadow.enabled

    def test_stats_empty_initially(self):
        from gateway.shadow import ShadowEvaluator
        shadow = ShadowEvaluator()
        stats = shadow.get_shadow_stats()
        assert stats["total_evaluations"] == 0
