"""Unit and integration tests for Canary Rollback Guard and sub-second rollback execution."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "serving") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "serving"))

from serving.rollback_guard import CanaryRollbackGuard


def test_evaluate_sample_metrics_pass() -> None:
    """Verify metrics pass when within SLA thresholds."""
    guard = CanaryRollbackGuard(max_p95_latency_ms=100.0, max_error_rate=0.02)
    latencies = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 75.0, 80.0]
    is_ok, reason = guard.evaluate_sample_metrics(latencies, errors=0, total=10)
    assert is_ok is True
    assert reason == "SLA satisfied"


def test_evaluate_sample_metrics_latency_breach() -> None:
    """Verify SLA breach detected when p95 latency exceeds 100ms."""
    guard = CanaryRollbackGuard(max_p95_latency_ms=100.0, max_error_rate=0.02)
    # p95 will be 145.0 ms
    latencies = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 110.0, 145.0]
    is_ok, reason = guard.evaluate_sample_metrics(latencies, errors=0, total=10)
    assert is_ok is False
    assert "exceeded SLA" in reason
    assert "p95 latency" in reason


def test_evaluate_sample_metrics_error_rate_breach() -> None:
    """Verify SLA breach detected when error rate exceeds 2%."""
    guard = CanaryRollbackGuard(max_p95_latency_ms=100.0, max_error_rate=0.02)
    latencies = [30.0] * 10
    # 1 error out of 10 requests = 10% error rate > 2% SLA
    is_ok, reason = guard.evaluate_sample_metrics(latencies, errors=1, total=10)
    assert is_ok is False
    assert "Error rate" in reason


def test_trigger_rollback_updates_nginx_config_and_measures_duration(tmp_path: Path) -> None:
    """Verify automated rollback modifies config to 0% canary and measures duration in seconds."""
    mock_conf = tmp_path / "nginx-canary.conf"
    mock_conf.write_text(
        """
    split_clients "${remote_addr}${request_id}" $split_upstream {
        25%     canary_backend;
        *       production_backend;
    }
    """,
        encoding="utf-8",
    )

    guard = CanaryRollbackGuard(conf_path=mock_conf)
    duration = guard.trigger_rollback(reason="Test simulated slow model")

    # Verify duration was recorded and is sub-second
    assert isinstance(duration, float)
    assert duration > 0.0
    assert duration < 2.0, f"Rollback duration {duration:.4f}s took longer than 2s SLA"

    # Verify config updated to 0% canary
    updated_conf = mock_conf.read_text(encoding="utf-8")
    assert "0%     canary_backend;" in updated_conf
    assert "25%" not in updated_conf
