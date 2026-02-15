"""
Unit tests for drift detection module.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestKSTest:
    """Tests for Kolmogorov-Smirnov test."""

    def test_same_distribution(self):
        from feature_store.drift import compute_ks_test
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0, 1, 1000)
        stat, pvalue = compute_ks_test(a, b)
        assert stat < 0.1  # Should be similar
        assert pvalue > 0.05

    def test_different_distribution(self):
        from feature_store.drift import compute_ks_test
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(5, 1, 1000)  # Very different mean
        stat, pvalue = compute_ks_test(a, b)
        assert stat > 0.5  # Should be very different
        assert pvalue < 0.01

    def test_empty_arrays(self):
        from feature_store.drift import compute_ks_test
        stat, pvalue = compute_ks_test(np.array([]), np.array([1.0]))
        assert stat == 0.0
        assert pvalue == 1.0


class TestPSI:
    """Tests for Population Stability Index."""

    def test_no_shift(self):
        from feature_store.drift import compute_psi
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 5000)
        b = rng.normal(0, 1, 5000)
        psi = compute_psi(a, b)
        assert psi < 0.1  # No significant shift

    def test_significant_shift(self):
        from feature_store.drift import compute_psi
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 5000)
        b = rng.normal(3, 2, 5000)  # Major shift
        psi = compute_psi(a, b)
        assert psi > 0.2  # Significant shift

    def test_empty_arrays(self):
        from feature_store.drift import compute_psi
        psi = compute_psi(np.array([]), np.array([1.0]))
        assert psi == 0.0


class TestDriftDetector:
    """Tests for the DriftDetector class."""

    def test_no_drift(self):
        from feature_store.drift import DriftDetector
        rng = np.random.RandomState(42)

        detector = DriftDetector()
        detector.set_reference_data("amount", rng.normal(100, 20, 5000))

        current = {"amount": rng.normal(100, 20, 1000)}
        results = detector.check_drift(current)

        assert len(results) == 1
        assert not results[0].is_drifted

    def test_drift_detected(self):
        from feature_store.drift import DriftDetector
        rng = np.random.RandomState(42)

        detector = DriftDetector()
        detector.ks_threshold = 0.05
        detector.psi_threshold = 0.1
        detector.set_reference_data("amount", rng.normal(100, 20, 5000))

        # Major distribution shift
        current = {"amount": rng.normal(500, 50, 1000)}
        results = detector.check_drift(current)

        assert len(results) == 1
        assert results[0].is_drifted

    def test_drift_summary(self):
        from feature_store.drift import DriftDetector, DriftResult

        detector = DriftDetector()
        results = [
            DriftResult("f1", 0.05, 0.3, 0.05, False, False),
            DriftResult("f2", 0.15, 0.01, 0.25, True, True),
        ]

        summary = detector.get_drift_summary(results)
        assert summary["total_features_checked"] == 2
        assert summary["drifted_features"] == 1
        assert summary["overall_drift"] is True
        assert "f2" in summary["drifted_feature_names"]
