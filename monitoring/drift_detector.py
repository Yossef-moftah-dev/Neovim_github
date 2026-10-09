"""Statistical drift detection algorithms for production ML monitoring.

Implements five rigorous statistical detection methods:
1. Chi-Square Test (Categorical features / predicted classes)
2. Wasserstein Distance (Earth Mover's Distance for continuous features)
3. Population Stability Index (PSI) (Categorical and continuous feature stability)
4. Jensen-Shannon Divergence (Smoothed symmetric probability divergence)
5. Maximum Mean Discrepancy (MMD) (Kernel two-sample test for representations)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.spatial.distance as sp_dist
import scipy.stats as sp_stats


@dataclass
class DriftEvaluationResult:
    """Standardized drift test outcome."""

    method: str
    feature_name: str
    statistic: float
    threshold: float
    p_value: float | None
    drift_detected: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize outcome to a standard JSON-compatible dictionary."""
        return {
            "method": self.method,
            "feature_name": self.feature_name,
            "statistic": round(float(self.statistic), 6),
            "threshold": round(float(self.threshold), 6),
            "p_value": round(float(self.p_value), 6) if self.p_value is not None else None,
            "drift_detected": bool(self.drift_detected),
            "details": self.details,
        }


def calculate_chi_square(
    reference_counts: dict[str | int, int] | list[int] | np.ndarray,
    current_counts: dict[str | int, int] | list[int] | np.ndarray,
    feature_name: str = "class_label",
    alpha: float = 0.05,
) -> DriftEvaluationResult:
    """Calculate Chi-Square test of independence between reference and current categories."""
    if isinstance(reference_counts, dict) and isinstance(current_counts, dict):
        all_categories = sorted(set(reference_counts.keys()) | set(current_counts.keys()))
        ref_arr = np.array([reference_counts.get(k, 0) for k in all_categories], dtype=float)
        cur_arr = np.array([current_counts.get(k, 0) for k in all_categories], dtype=float)
    else:
        ref_arr = np.asarray(reference_counts, dtype=float)
        cur_arr = np.asarray(current_counts, dtype=float)

    total_ref = np.sum(ref_arr)
    total_cur = np.sum(cur_arr)

    if total_ref == 0 or total_cur == 0 or len(ref_arr) <= 1:
        return DriftEvaluationResult(
            method="chi_square",
            feature_name=feature_name,
            statistic=0.0,
            threshold=alpha,
            p_value=1.0,
            drift_detected=False,
            details={"error": "Insufficient samples or categories"},
        )

    # 2xK Contingency table
    contingency = np.vstack([ref_arr, cur_arr])
    res = sp_stats.chi2_contingency(contingency)
    stat = float(res.statistic)
    p_val = float(res.pvalue)

    return DriftEvaluationResult(
        method="chi_square",
        feature_name=feature_name,
        statistic=stat,
        threshold=alpha,
        p_value=p_val,
        drift_detected=bool(p_val < alpha),
        details={"degrees_of_freedom": int(res.dof), "alpha": alpha},
    )


def calculate_wasserstein_distance(
    reference_samples: list[float] | np.ndarray,
    current_samples: list[float] | np.ndarray,
    feature_name: str = "feature",
    threshold: float = 0.15,
) -> DriftEvaluationResult:
    """Calculate 1-Wasserstein Distance (Earth Mover's Distance) for continuous features."""
    ref = np.asarray(reference_samples, dtype=float).ravel()
    cur = np.asarray(current_samples, dtype=float).ravel()

    if len(ref) == 0 or len(cur) == 0:
        return DriftEvaluationResult(
            method="wasserstein",
            feature_name=feature_name,
            statistic=0.0,
            threshold=threshold,
            p_value=None,
            drift_detected=False,
            details={"error": "Empty distribution"},
        )

    w_dist = float(sp_stats.wasserstein_distance(ref, cur))
    return DriftEvaluationResult(
        method="wasserstein",
        feature_name=feature_name,
        statistic=w_dist,
        threshold=threshold,
        p_value=None,
        drift_detected=bool(w_dist >= threshold),
        details={"ref_count": len(ref), "cur_count": len(cur)},
    )


def calculate_psi(
    reference: list[float] | np.ndarray,
    current: list[float] | np.ndarray,
    feature_name: str = "feature",
    num_bins: int = 10,
    threshold: float = 0.20,
    epsilon: float = 1e-4,
) -> DriftEvaluationResult:
    """Calculate Population Stability Index (PSI).

    Interpretations:
    - PSI < 0.1: No significant shift (Stable)
    - 0.1 <= PSI < 0.2: Moderate shift (Monitor)
    - PSI >= 0.2: Significant shift (Action/Retrain Required)
    """
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()

    if len(ref) == 0 or len(cur) == 0:
        return DriftEvaluationResult(
            method="psi",
            feature_name=feature_name,
            statistic=0.0,
            threshold=threshold,
            p_value=None,
            drift_detected=False,
            details={"error": "Empty distribution"},
        )

    # Binning based on reference percentiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    bin_edges = np.unique(bin_edges)

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / len(ref)
    cur_pct = cur_counts / len(cur)

    # Apply smoothing epsilon to avoid zero division / log(0)
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    ref_pct = ref_pct / np.sum(ref_pct)
    cur_pct = cur_pct / np.sum(cur_pct)

    psi_val = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    status = "stable"
    if psi_val >= 0.20:
        status = "significant_drift"
    elif psi_val >= 0.10:
        status = "moderate_drift"

    return DriftEvaluationResult(
        method="psi",
        feature_name=feature_name,
        statistic=psi_val,
        threshold=threshold,
        p_value=None,
        drift_detected=bool(psi_val >= threshold),
        details={"status": status, "bins_used": len(bin_edges) - 1},
    )


def calculate_jensen_shannon_divergence(
    reference_dist: list[float] | np.ndarray,
    current_dist: list[float] | np.ndarray,
    feature_name: str = "distribution",
    threshold: float = 0.25,
) -> DriftEvaluationResult:
    """Calculate Jensen-Shannon Divergence between two probability distributions."""
    ref = np.asarray(reference_dist, dtype=float)
    cur = np.asarray(current_dist, dtype=float)

    if ref.sum() == 0 or cur.sum() == 0:
        return DriftEvaluationResult(
            method="jensen_shannon",
            feature_name=feature_name,
            statistic=0.0,
            threshold=threshold,
            p_value=None,
            drift_detected=False,
            details={"error": "Distribution sum is zero"},
        )

    p = ref / ref.sum()
    q = cur / cur.sum()

    # jensenshannon returns the distance (square root of divergence); square it for divergence
    js_distance = float(sp_dist.jensenshannon(p, q, base=2.0))
    js_divergence = float(js_distance**2)

    return DriftEvaluationResult(
        method="jensen_shannon",
        feature_name=feature_name,
        statistic=js_divergence,
        threshold=threshold,
        p_value=None,
        drift_detected=bool(js_divergence >= threshold),
        details={"js_distance": js_distance},
    )


def calculate_mmd(
    reference_vectors: np.ndarray,
    current_vectors: np.ndarray,
    feature_name: str = "embeddings",
    gamma: float | None = None,
    threshold: float = 0.10,
) -> DriftEvaluationResult:
    """Calculate Maximum Mean Discrepancy (MMD) with an RBF kernel."""
    x = np.atleast_2d(reference_vectors).astype(float)
    y = np.atleast_2d(current_vectors).astype(float)

    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return DriftEvaluationResult(
            method="mmd",
            feature_name=feature_name,
            statistic=0.0,
            threshold=threshold,
            p_value=None,
            drift_detected=False,
            details={"error": "Empty vectors"},
        )

    # Median heuristic for gamma if not provided
    if gamma is None:
        pairwise_dists = sp_dist.pdist(np.vstack([x, y]), metric="sqeuclidean")
        median_dist = float(np.median(pairwise_dists)) if len(pairwise_dists) > 0 else 1.0
        gamma = 1.0 / (2.0 * max(median_dist, 1e-4))

    # Kernel matrices
    k_xx = np.exp(-gamma * sp_dist.cdist(x, x, metric="sqeuclidean"))
    k_yy = np.exp(-gamma * sp_dist.cdist(y, y, metric="sqeuclidean"))
    k_xy = np.exp(-gamma * sp_dist.cdist(x, y, metric="sqeuclidean"))

    mmd_squared = float(
        (np.sum(k_xx) / (n * n)) - (2.0 * np.sum(k_xy) / (n * m)) + (np.sum(k_yy) / (m * m))
    )
    mmd_stat = float(np.sqrt(max(mmd_squared, 0.0)))

    return DriftEvaluationResult(
        method="mmd",
        feature_name=feature_name,
        statistic=mmd_stat,
        threshold=threshold,
        p_value=None,
        drift_detected=bool(mmd_stat >= threshold),
        details={"gamma": gamma, "mmd_squared": mmd_squared},
    )


class MultiMethodDriftDetector:
    """Holistic multi-method drift evaluator combining categorical and continuous tests."""

    def __init__(
        self,
        chi_square_alpha: float = 0.05,
        psi_threshold: float = 0.20,
        wasserstein_threshold: float = 0.15,
        js_threshold: float = 0.25,
    ) -> None:
        self.chi_square_alpha = chi_square_alpha
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.js_threshold = js_threshold

    def evaluate_batch(
        self,
        ref_labels: list[int] | np.ndarray,
        cur_labels: list[int] | np.ndarray,
        ref_confidences: list[float] | np.ndarray,
        cur_confidences: list[float] | np.ndarray,
    ) -> dict[str, DriftEvaluationResult]:
        """Evaluate categorical and continuous drift metrics on reference vs current batches."""
        # Class categorical checks
        ref_label_counts = dict(zip(*np.unique(ref_labels, return_counts=True), strict=False))
        cur_label_counts = dict(zip(*np.unique(cur_labels, return_counts=True), strict=False))

        chi_res = calculate_chi_square(
            ref_label_counts,
            cur_label_counts,
            feature_name="predicted_labels",
            alpha=self.chi_square_alpha,
        )

        # Categorical probability distributions for JS Divergence
        categories = sorted(set(ref_label_counts.keys()) | set(cur_label_counts.keys()))
        ref_probs = [ref_label_counts.get(c, 0) for c in categories]
        cur_probs = [cur_label_counts.get(c, 0) for c in categories]
        js_res = calculate_jensen_shannon_divergence(
            ref_probs,
            cur_probs,
            feature_name="label_distribution",
            threshold=self.js_threshold,
        )

        # Continuous checks on confidence scores
        wass_res = calculate_wasserstein_distance(
            ref_confidences,
            cur_confidences,
            feature_name="prediction_confidence",
            threshold=self.wasserstein_threshold,
        )
        psi_res = calculate_psi(
            ref_confidences,
            cur_confidences,
            feature_name="prediction_confidence",
            threshold=self.psi_threshold,
        )

        return {
            "chi_square": chi_res,
            "jensen_shannon": js_res,
            "wasserstein": wass_res,
            "psi": psi_res,
        }
