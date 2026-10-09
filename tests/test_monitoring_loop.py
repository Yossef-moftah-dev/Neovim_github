"""Unit tests for closed retraining loop storm protections and Evidently monitor."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from monitoring.evidently_monitor import extract_nlp_features, run_statistical_drift_suite
from monitoring.retraining_trigger import RetrainingStormGuard


def test_retraining_storm_guard_cooldown_and_rate_limit():
    """Verify Dwell Time cooldown and rate limiting prevent retraining storms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"
        guard = RetrainingStormGuard(
            min_cooldown_seconds=10,
            max_triggers_per_day=2,
            min_samples_threshold=10,
            state_file=state_file,
        )

        dummy_data = pd.DataFrame(
            {
                "text": ["نص ممتاز جدا", "تجربة سيئة جدا", "عادي لا بأس"] * 5,
                "label": [2, 0, 1] * 5,
            }
        )

        # 1. First trigger should pass
        passed, reason = guard.check_protections(data_df=dummy_data)
        assert passed
        assert reason == "PASSED"
        guard.record_successful_trigger()

        # 2. Immediate second trigger should be blocked by cooldown gate
        passed2, reason2 = guard.check_protections(data_df=dummy_data)
        assert not passed2
        assert "REJECTED_COOLDOWN" in reason2

        # 3. Force override should bypass cooldown
        passed_force, _ = guard.check_protections(data_df=dummy_data, force=True)
        assert passed_force


def test_retraining_storm_guard_data_quality_gates():
    """Verify Data Quality & Volume gates block small, null, or single-class data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"
        guard = RetrainingStormGuard(
            min_cooldown_seconds=0,
            max_triggers_per_day=10,
            min_samples_threshold=20,
            state_file=state_file,
        )

        # Insufficient volume (< 20 rows)
        small_df = pd.DataFrame({"text": ["نص"] * 5, "label": [1] * 5})
        passed_vol, reason_vol = guard.check_protections(data_df=small_df)
        assert not passed_vol
        assert "REJECTED_DATA_VOLUME" in reason_vol

        # Null values
        null_df = pd.DataFrame(
            {
                "text": ["نص", None] * 15,
                "label": [0, 1] * 15,
            }
        )
        passed_null, reason_null = guard.check_protections(data_df=null_df)
        assert not passed_null
        assert "REJECTED_DATA_NULLS" in reason_null

        # Single class (no diversity)
        mono_df = pd.DataFrame(
            {
                "text": ["نص"] * 25,
                "label": [1] * 25,
            }
        )
        passed_div, reason_div = guard.check_protections(data_df=mono_df)
        assert not passed_div
        assert "REJECTED_DATA_DIVERSITY" in reason_div


def test_extract_nlp_features():
    """Verify text NLP feature extraction for Arabic text."""
    df = pd.DataFrame({"text": ["المنتج رائع جدا", "Good product"]})
    res = extract_nlp_features(df)

    assert "char_length" in res.columns
    assert "word_count" in res.columns
    assert "arabic_ratio" in res.columns
    assert res.loc[0, "arabic_ratio"] > 0.8
    assert res.loc[1, "arabic_ratio"] == pytest.approx(0.0, abs=1e-3)


def test_run_statistical_drift_suite():
    """Verify statistical drift suite runs without errors on pandas inputs."""
    ref_df = pd.DataFrame(
        {
            "text": ["منتج رائع جدا وممتاز"] * 50 + ["خدمة سيئة وتالف"] * 10,
            "label": [2] * 50 + [0] * 10,
        }
    )
    cur_df = pd.DataFrame(
        {
            "text": ["منتج رائع جدا وممتاز"] * 50 + ["خدمة سيئة وتالف"] * 10,
            "label": [2] * 50 + [0] * 10,
        }
    )

    summary = run_statistical_drift_suite(ref_df, cur_df)
    assert "timestamp" in summary
    assert "evaluations" in summary
    assert len(summary["evaluations"]) >= 4
