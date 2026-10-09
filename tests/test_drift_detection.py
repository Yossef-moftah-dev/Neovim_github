"""Unit tests for statistical drift detection algorithms and drift simulation."""

import numpy as np
import pytest

from monitoring.drift_detector import (
    MultiMethodDriftDetector,
    calculate_chi_square,
    calculate_jensen_shannon_divergence,
    calculate_mmd,
    calculate_psi,
    calculate_wasserstein_distance,
)
from monitoring.simulate_drift import (
    DriftTypology,
    apply_gradual_drift,
    apply_incremental_drift,
    apply_periodic_drift,
    apply_sudden_drift,
    generate_baseline_batch,
    generate_drift_dataset,
)


def test_chi_square_identical_distribution():
    """Verify Chi-Square test yields no drift for identical categorical splits."""
    ref = {0: 100, 1: 50, 2: 150}
    cur = {0: 100, 1: 50, 2: 150}
    res = calculate_chi_square(ref, cur, alpha=0.05)

    assert not res.drift_detected
    assert res.p_value is not None
    assert res.p_value > 0.90
    assert res.statistic == pytest.approx(0.0, abs=1e-5)


def test_chi_square_shifted_distribution():
    """Verify Chi-Square detects severe categorical drift."""
    ref = {0: 10, 1: 20, 2: 200}  # Mostly positive
    cur = {0: 200, 1: 20, 2: 10}  # Mostly negative
    res = calculate_chi_square(ref, cur, alpha=0.05)

    assert res.drift_detected
    assert res.p_value is not None
    assert res.p_value < 0.001
    assert res.statistic > 50.0


def test_wasserstein_distance_calculation():
    """Verify Wasserstein Distance properly measures continuous scalar distribution shift."""
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=0.9, scale=0.05, size=200)
    cur_same = rng.normal(loc=0.9, scale=0.05, size=200)
    cur_drifted = rng.normal(loc=0.3, scale=0.05, size=200)

    res_same = calculate_wasserstein_distance(ref, cur_same, threshold=0.15)
    assert not res_same.drift_detected
    assert res_same.statistic < 0.10

    res_drifted = calculate_wasserstein_distance(ref, cur_drifted, threshold=0.15)
    assert res_drifted.drift_detected
    assert res_drifted.statistic > 0.50


def test_psi_calculation_and_stability():
    """Verify Population Stability Index (PSI) flags stable vs drifted continuous distributions."""
    rng = np.random.default_rng(123)
    ref = rng.normal(loc=50.0, scale=10.0, size=500)
    cur_stable = rng.normal(loc=50.5, scale=10.0, size=500)
    cur_drifted = rng.normal(loc=75.0, scale=15.0, size=500)

    res_stable = calculate_psi(ref, cur_stable, threshold=0.20)
    assert not res_stable.drift_detected
    assert res_stable.statistic < 0.10
    assert res_stable.details["status"] == "stable"

    res_drifted = calculate_psi(ref, cur_drifted, threshold=0.20)
    assert res_drifted.drift_detected
    assert res_drifted.statistic > 0.50
    assert res_drifted.details["status"] == "significant_drift"


def test_jensen_shannon_divergence():
    """Verify Jensen-Shannon Divergence bounds and detection."""
    p_identical = [0.2, 0.3, 0.5]
    q_identical = [0.2, 0.3, 0.5]
    res_ident = calculate_jensen_shannon_divergence(p_identical, q_identical, threshold=0.25)
    assert not res_ident.drift_detected
    assert res_ident.statistic == pytest.approx(0.0, abs=1e-5)

    p_div = [0.05, 0.05, 0.90]
    q_div = [0.90, 0.05, 0.05]
    res_div = calculate_jensen_shannon_divergence(p_div, q_div, threshold=0.25)
    assert res_div.drift_detected
    assert res_div.statistic > 0.50


def test_mmd_calculation():
    """Verify Maximum Mean Discrepancy (MMD) two-sample kernel test."""
    rng = np.random.default_rng(42)
    x = rng.normal(loc=0.0, scale=1.0, size=(100, 4))
    y_same = rng.normal(loc=0.0, scale=1.0, size=(100, 4))
    y_diff = rng.normal(loc=4.0, scale=1.0, size=(100, 4))

    res_same = calculate_mmd(x, y_same, threshold=0.15)
    assert not res_same.drift_detected
    assert res_same.statistic < 0.15

    res_diff = calculate_mmd(x, y_diff, threshold=0.15)
    assert res_diff.drift_detected
    assert res_diff.statistic > 0.30


def test_multi_method_evaluator():
    """Verify MultiMethodDriftDetector aggregates tests appropriately."""
    detector = MultiMethodDriftDetector()
    ref_labels = np.array([0, 1, 2, 2, 2, 1, 0, 2])
    cur_labels = np.array([0, 0, 0, 0, 1, 0, 0, 1])  # Shift to negative (0)
    ref_conf = np.array([0.9, 0.85, 0.95, 0.92, 0.88, 0.91, 0.89, 0.94])
    cur_conf = np.array([0.55, 0.52, 0.61, 0.58, 0.54, 0.59, 0.53, 0.56])  # Degraded confidence

    report = detector.evaluate_batch(ref_labels, cur_labels, ref_conf, cur_conf)
    assert "chi_square" in report
    assert "jensen_shannon" in report
    assert "wasserstein" in report
    assert "psi" in report
    assert report["wasserstein"].drift_detected


def test_simulate_drift_typologies():
    """Verify all 4 drift typologies produce expected datasets and columns."""
    # Baseline
    df_base = generate_baseline_batch(n_samples=50)
    assert len(df_base) == 50
    assert "text" in df_base.columns
    assert "label" in df_base.columns

    # 1. Sudden
    df_sudden = apply_sudden_drift(n_samples=60, shift_ratio=0.8)
    assert len(df_sudden) == 60
    assert (df_sudden["is_drifted"] == 1).sum() > 40

    # 2. Gradual
    df_grad = apply_gradual_drift(n_steps=4, samples_per_step=15)
    assert len(df_grad) == 60
    assert "step" in df_grad.columns

    # 3. Incremental
    df_inc = apply_incremental_drift(n_phases=3, samples_per_phase=20)
    assert len(df_inc) == 60
    assert "phase" in df_inc.columns

    # 4. Periodic
    df_per = apply_periodic_drift(n_cycles=2, points_per_cycle=4, samples_per_point=10)
    assert len(df_per) == 80
    assert "time_point" in df_per.columns

    # Unified generator
    df_uni = generate_drift_dataset(typology=DriftTypology.SUDDEN, n_samples=30)
    assert len(df_uni) == 30
