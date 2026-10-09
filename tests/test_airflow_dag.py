"""Unit and structural tests for Apache Airflow training pipeline DAG."""

from __future__ import annotations

import ast
import json
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DAG_FILE = PROJECT_ROOT / "pipelines" / "dags" / "train_pipeline.py"


def test_airflow_dag_file_exists() -> None:
    """Verify that the Airflow DAG file is located at the expected path."""
    assert DAG_FILE.exists(), f"DAG file missing at {DAG_FILE}"


def test_airflow_dag_ast_syntax() -> None:
    """Verify the DAG file has valid Python syntax and contains key Airflow components."""
    content = DAG_FILE.read_text(encoding="utf-8")
    parsed = ast.parse(content)
    assert isinstance(parsed, ast.Module)

    # Check for presence of required Airflow constructs in code
    assert "arabic_sentiment_training_pipeline" in content
    assert "FileSensor" in content
    assert "BranchPythonOperator" in content
    assert "extract_data" in content
    assert "validate_data" in content
    assert "train_model" in content
    assert "evaluate_model" in content
    assert "branch_quality_gate" in content
    assert "register_model" in content
    assert "model_rejected" in content


def test_branch_quality_gate_logic(tmp_path: Path) -> None:
    """Verify branching function routes correctly based on metric SLA thresholds."""
    # Read and execute branch callable logic
    from pipelines.dags.train_pipeline import branch_quality_gate_callable

    metrics_file = tmp_path / "eval_metrics.json"

    with patch("pipelines.dags.train_pipeline.METRICS_PATH", metrics_file):
        # Case 1: High performance metrics -> register_model
        metrics_file.write_text(
            json.dumps({"accuracy": 0.885, "macro_f1": 0.862, "mean_latency_ms": 22.4}),
            encoding="utf-8",
        )
        assert branch_quality_gate_callable() == "register_model"

        # Case 2: Accuracy regression -> model_rejected
        metrics_file.write_text(
            json.dumps({"accuracy": 0.550, "macro_f1": 0.862, "mean_latency_ms": 22.4}),
            encoding="utf-8",
        )
        assert branch_quality_gate_callable() == "model_rejected"

        # Case 3: F1 regression -> model_rejected
        metrics_file.write_text(
            json.dumps({"accuracy": 0.850, "macro_f1": 0.510, "mean_latency_ms": 22.4}),
            encoding="utf-8",
        )
        assert branch_quality_gate_callable() == "model_rejected"

        # Case 4: Latency regression -> model_rejected
        metrics_file.write_text(
            json.dumps({"accuracy": 0.885, "macro_f1": 0.862, "mean_latency_ms": 150.0}),
            encoding="utf-8",
        )
        assert branch_quality_gate_callable() == "model_rejected"


def test_validate_data_callable_logic(tmp_path: Path) -> None:
    """Verify data validation catches invalid formats and passes clean splits."""
    import pandas as pd

    from pipelines.dags.train_pipeline import AirflowException, validate_data_callable

    train_file = tmp_path / "train.csv"
    val_file = tmp_path / "val.csv"

    # Case 1: Valid dataset
    df_valid = pd.DataFrame(
        {
            "text": ["رائع جدا", "سيء للغاية", "متوسط"],
            "label": [2, 0, 1],
        }
    )
    df_valid.to_csv(train_file, index=False)
    df_valid.to_csv(val_file, index=False)

    with (
        patch("pipelines.dags.train_pipeline.TRAIN_DATA_PATH", train_file),
        patch("pipelines.dags.train_pipeline.VAL_DATA_PATH", val_file),
    ):
        res = validate_data_callable()
        assert res["status"] == "VALID"
        assert res["train_count"] == 3

    # Case 2: Missing required column
    df_invalid = pd.DataFrame({"text": ["رائع"], "wrong_label": [1]})
    df_invalid.to_csv(train_file, index=False)

    with (
        patch("pipelines.dags.train_pipeline.TRAIN_DATA_PATH", train_file),
        patch("pipelines.dags.train_pipeline.VAL_DATA_PATH", val_file),
        pytest.raises(AirflowException, match="Missing required columns"),
    ):
        validate_data_callable()


def test_airflow_dag_graph_structure() -> None:
    """Verify full DAG graph structure, tasks, and retry policies if airflow is available."""
    from pipelines.dags.train_pipeline import AIRFLOW_AVAILABLE, dag

    if not AIRFLOW_AVAILABLE or dag is None:
        pytest.skip("Airflow library not installed in this environment; skipping DAG graph test")

    assert dag.dag_id == "arabic_sentiment_training_pipeline"
    assert dag.default_args["retries"] == 2
    assert dag.default_args["retry_delay"] == timedelta(seconds=15)

    task_ids = {t.task_id for t in dag.tasks}
    expected_tasks = {
        "sensor_upstream_data",
        "extract_data",
        "validate_data",
        "train_model",
        "evaluate_model",
        "branch_quality_gate",
        "register_model",
        "model_rejected",
    }
    assert expected_tasks.issubset(task_ids)

    # Check upstream / downstream edges
    extract_task = dag.get_task("extract_data")
    assert "sensor_upstream_data" in extract_task.upstream_task_ids
    assert "validate_data" in extract_task.downstream_task_ids

    gate_task = dag.get_task("branch_quality_gate")
    assert "evaluate_model" in gate_task.upstream_task_ids
    assert {"register_model", "model_rejected"}.issubset(gate_task.downstream_task_ids)
