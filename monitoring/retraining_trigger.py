"""Closed retraining loop trigger with 3-tier retraining storm protections.

Storm Protections:
1. Dwell Time / Cooldown Window: Enforces mandatory waiting interval between training executions.
2. Rate Limiting: Limits max retraining runs within a rolling temporal window.
3. Data Integrity & Volume Check: Enforces sample volume (>= N rows), non-null content, and label validity.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("monitoring.retraining_trigger")

STATE_FILE = Path("outputs/retraining_state.json")


class RetrainingStormGuard:
    """Evaluates the 3-tier storm protection gates before permitting pipeline trigger."""

    def __init__(
        self,
        min_cooldown_seconds: int = 60,
        max_triggers_per_day: int = 5,
        min_samples_threshold: int = 50,
        state_file: Path | str = STATE_FILE,
    ) -> None:
        self.min_cooldown_seconds = min_cooldown_seconds
        self.max_triggers_per_day = max_triggers_per_day
        self.min_samples_threshold = min_samples_threshold
        self.state_file = Path(state_file)
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not read state file: %s", exc)
        return {"last_trigger_timestamp": 0.0, "daily_trigger_history": []}

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def check_protections(
        self,
        data_df: pd.DataFrame | None = None,
        force: bool = False,
    ) -> tuple[bool, str]:
        """Verify Dwell Time, Rate Limit, and Data Integrity."""
        if force:
            logger.info("Storm protections bypassed via force override flag.")
            return True, "BYPASSED_FORCE"

        now = time.time()

        # Gate 1: Dwell Time / Cooldown Window
        last_time = float(self.state.get("last_trigger_timestamp", 0.0))
        elapsed = now - last_time
        if elapsed < self.min_cooldown_seconds:
            remaining = int(self.min_cooldown_seconds - elapsed)
            msg = f"Dwell time cooldown active! Elapsed {int(elapsed)}s < {self.min_cooldown_seconds}s. Wait {remaining}s."
            logger.warning("Storm Protection Gate 1 Failed: %s", msg)
            return False, f"REJECTED_COOLDOWN: {msg}"

        # Gate 2: Rolling 24-hour Rate Limiting
        history = self.state.get("daily_trigger_history", [])
        one_day_ago = now - 86400
        recent = [t for t in history if t > one_day_ago]
        if len(recent) >= self.max_triggers_per_day:
            msg = f"Rate limit reached! {len(recent)} triggers in last 24h exceeds limit of {self.max_triggers_per_day}."
            logger.warning("Storm Protection Gate 2 Failed: %s", msg)
            return False, f"REJECTED_RATE_LIMIT: {msg}"

        # Gate 3: Data Quality & Volume Check
        if data_df is not None:
            if len(data_df) < self.min_samples_threshold:
                msg = f"Insufficient sample volume: {len(data_df)} < threshold {self.min_samples_threshold}."
                logger.warning("Storm Protection Gate 3 Failed: %s", msg)
                return False, f"REJECTED_DATA_VOLUME: {msg}"

            if "text" in data_df.columns and data_df["text"].isnull().sum() > 0:
                msg = f"Dataset contains {data_df['text'].isnull().sum()} null text rows."
                logger.warning("Storm Protection Gate 3 Failed: %s", msg)
                return False, f"REJECTED_DATA_NULLS: {msg}"

            if "label" in data_df.columns and len(data_df["label"].unique()) < 2:
                msg = "Dataset lacks class diversity (less than 2 distinct labels)."
                logger.warning("Storm Protection Gate 3 Failed: %s", msg)
                return False, f"REJECTED_DATA_DIVERSITY: {msg}"

        logger.info("All 3 Retraining Storm Protection Gates PASSED!")
        return True, "PASSED"

    def record_successful_trigger(self) -> None:
        """Update persistent state after a successful retraining invocation."""
        now = time.time()
        self.state["last_trigger_timestamp"] = now
        history = self.state.get("daily_trigger_history", [])
        history.append(now)
        # Keep last 100 entries
        self.state["daily_trigger_history"] = history[-100:]
        self._save_state()


def trigger_airflow_dag(
    dag_id: str = "arabic_sentiment_training_pipeline",
    airflow_url: str = "http://localhost:8080",
    username: str = "admin",
    password: str = "admin",
) -> bool:
    """Dispatch DAG run execution via Apache Airflow REST API."""
    import base64
    import urllib.request

    endpoint = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
    logger.info("Dispatching DAG run request to Airflow: %s", endpoint)

    auth_str = f"{username}:{password}"
    auth_header = f"Basic {base64.b64encode(auth_str.encode()).decode()}"

    payload = json.dumps(
        {
            "conf": {
                "trigger_reason": "drift_detected_closed_loop",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": auth_header},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            res_body = json.loads(response.read().decode())
            logger.info(
                "Successfully triggered Airflow DAG run: ID=%s, State=%s",
                res_body.get("dag_run_id"),
                res_body.get("state"),
            )
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Airflow REST API unreachable (%s). Using local pipeline execution fallback.", exc
        )
        return False


def trigger_local_pipeline_execution() -> bool:
    """Fallback local pipeline execution using scripts/run_pipeline_train.py."""
    logger.info("Executing local retraining pipeline fallback...")
    cmd = [
        sys.executable,
        "scripts/run_pipeline_train.py",
        "--train-data",
        "data/processed/train.csv",
        "--val-data",
        "data/processed/val.csv",
        "--metrics",
        "reports/eval_metrics.json",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Local pipeline executed successfully! Output:\n%s", res.stdout)
        return True
    except subprocess.CalledProcessError as err:
        logger.error("Local pipeline execution failed:\n%s\n%s", err.stdout, err.stderr)
        return False


def dispatch_retraining_loop(
    drift_data_path: Path | str | None = None,
    force: bool = False,
    cooldown_seconds: int = 60,
) -> dict[str, Any]:
    """Evaluate storm protections and trigger closed retraining loop."""
    guard = RetrainingStormGuard(min_cooldown_seconds=cooldown_seconds)

    df = None
    if drift_data_path and Path(drift_data_path).exists():
        df = pd.read_csv(drift_data_path)

    passed, reason = guard.check_protections(data_df=df, force=force)

    if not passed:
        return {
            "success": False,
            "reason": reason,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # Dispatch to Airflow; fallback to local script
    triggered = trigger_airflow_dag()
    if not triggered:
        triggered = trigger_local_pipeline_execution()

    if triggered:
        guard.record_successful_trigger()

    return {
        "success": triggered,
        "reason": "TRIGGERED" if triggered else "DISPATCH_FAILED",
        "timestamp": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Closed Retraining Loop Trigger with Storm Protections"
    )
    parser.add_argument(
        "--drift-data", type=str, default=None, help="Path to accumulated drift data"
    )
    parser.add_argument("--force", action="store_true", help="Bypass storm protection gates")
    parser.add_argument("--cooldown", type=int, default=60, help="Cooldown dwell time in seconds")

    args = parser.parse_args()
    res = dispatch_retraining_loop(
        drift_data_path=args.drift_data, force=args.force, cooldown_seconds=args.cooldown
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
