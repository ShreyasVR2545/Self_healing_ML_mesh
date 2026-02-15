"""
Drift Detection Module.

Implements statistical drift detection using:
  - Kolmogorov-Smirnov (KS) test
  - Population Stability Index (PSI)

Compares current live feature distributions against
a reference distribution from training data.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger("drift_detector")


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    psi_score: float
    is_drifted_ks: bool
    is_drifted_psi: bool

    @property
    def is_drifted(self) -> bool:
        return self.is_drifted_ks or self.is_drifted_psi


def compute_ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Tests whether two samples come from the same distribution.

    Args:
        reference: Reference (training) distribution
        current: Current (live) distribution

    Returns:
        (ks_statistic, p_value)
    """
    if len(reference) < 2 or len(current) < 2:
        return 0.0, 1.0

    statistic, pvalue = stats.ks_2samp(reference, current)
    return float(statistic), float(pvalue)


def compute_psi(reference: np.ndarray, current: np.ndarray,
                n_bins: int = 10, epsilon: float = 1e-4) -> float:
    """Population Stability Index (PSI).

    Measures how much a distribution has shifted over time.
    PSI < 0.1: no significant change
    0.1 ≤ PSI < 0.2: moderate shift
    PSI ≥ 0.2: significant shift

    Args:
        reference: Reference distribution
        current: Current distribution
        n_bins: Number of bins for discretization
        epsilon: Small constant to avoid log(0)

    Returns:
        PSI score (float ≥ 0)
    """
    if len(reference) < 2 or len(current) < 2:
        return 0.0

    # Create bins from reference distribution
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    # Compute bin proportions
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current, bins=breakpoints)[0]

    ref_proportions = (ref_counts / len(reference)) + epsilon
    cur_proportions = (cur_counts / len(current)) + epsilon

    # PSI formula
    psi = np.sum(
        (cur_proportions - ref_proportions) * np.log(cur_proportions / ref_proportions)
    )

    return float(psi)


class DriftDetector:
    """Drift detection manager comparing live data against reference."""

    def __init__(self, reference_stats_path: str = None):
        self.ks_threshold = float(os.environ.get("DRIFT_KS_THRESHOLD", "0.1"))
        self.psi_threshold = float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.2"))
        self.reference_data: Dict[str, np.ndarray] = {}
        self.reference_stats: Dict[str, dict] = {}

        if reference_stats_path:
            self.load_reference_stats(reference_stats_path)

    def load_reference_stats(self, path: str):
        """Load reference statistics from training data."""
        try:
            with open(path) as f:
                self.reference_stats = json.load(f)
            logger.info(f"Loaded reference stats from {path}")
        except Exception as e:
            logger.error(f"Failed to load reference stats: {e}")

    def set_reference_data(self, feature_name: str, data: np.ndarray):
        """Set reference distribution for a feature."""
        self.reference_data[feature_name] = data

    def set_reference_data_from_csv(self, csv_path: str, feature_names: List[str]):
        """Load reference data from training CSV."""
        import pandas as pd
        df = pd.read_csv(csv_path)
        for fname in feature_names:
            if fname in df.columns:
                self.reference_data[fname] = df[fname].values.astype(float)

    def check_drift(self, current_data: Dict[str, np.ndarray]) -> List[DriftResult]:
        """Check drift for all features with available reference data.

        Args:
            current_data: Dict mapping feature name → current values array

        Returns:
            List of DriftResult for each feature checked
        """
        results = []

        for feature_name, current in current_data.items():
            reference = self.reference_data.get(feature_name)
            if reference is None or len(current) < 10:
                continue

            ks_stat, ks_pvalue = compute_ks_test(reference, current)
            psi_score = compute_psi(reference, current)

            result = DriftResult(
                feature_name=feature_name,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pvalue,
                psi_score=psi_score,
                is_drifted_ks=ks_stat > self.ks_threshold,
                is_drifted_psi=psi_score > self.psi_threshold,
            )
            results.append(result)

            if result.is_drifted:
                logger.warning(
                    f"DRIFT detected on '{feature_name}': "
                    f"KS={ks_stat:.4f} (thresh={self.ks_threshold}), "
                    f"PSI={psi_score:.4f} (thresh={self.psi_threshold})"
                )

        return results

    def get_drift_summary(self, results: List[DriftResult]) -> dict:
        """Summarize drift detection results."""
        if not results:
            return {"total_features_checked": 0, "drifted_features": 0, "overall_drift": False}

        drifted = [r for r in results if r.is_drifted]
        return {
            "total_features_checked": len(results),
            "drifted_features": len(drifted),
            "drifted_feature_names": [r.feature_name for r in drifted],
            "overall_drift": len(drifted) > 0,
            "max_ks_statistic": max(r.ks_statistic for r in results),
            "max_psi_score": max(r.psi_score for r in results),
            "details": [
                {
                    "feature": r.feature_name,
                    "ks_stat": round(r.ks_statistic, 4),
                    "ks_pvalue": round(r.ks_pvalue, 4),
                    "psi": round(r.psi_score, 4),
                    "drifted": r.is_drifted,
                }
                for r in results
            ],
        }
