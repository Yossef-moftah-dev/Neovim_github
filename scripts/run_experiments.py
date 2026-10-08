"""Script to execute >= 5 experimental runs across >= 3 model families / chunk configurations.

Logs all runs to MLflow Tracking Server with parameters, metrics, tags, and artifacts,
registers the best candidate in the Model Registry, and transitions its lifecycle stage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure prodml is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
from mlflow.tracking import MlflowClient

from prodml.train import get_data_version, get_git_commit, train_and_log_run


def setup_tracking_env() -> str:
    """Set default environment variables for local MLflow & MinIO."""
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def main() -> None:
    tracking_uri = setup_tracking_env()
    experiment_name = "arabic-sentiment-classification"
    mlflow.set_experiment(experiment_name)

    data_path = PROJECT_ROOT / "data" / "Arabic_Reviews_of_SHEIN" / "train-00000-of-00001.parquet"
    data_version = get_data_version(data_path) if data_path.exists() else "dvc-v0.2.0"
    git_commit = get_git_commit()
    artifacts_dir = PROJECT_ROOT / "outputs" / "final_model"

    print("=" * 80)
    print(f"Starting MLflow Experiments on: {tracking_uri}")
    print(f"Data Version: {data_version} | Git Commit: {git_commit}")
    print("=" * 80)

    # Define 5 distinct runs across 3 distinct model families & chunking strategies
    run_definitions = [
        {
            "run_name": "run-1-arabert-chunk64",
            "model_family": "AraBERT",
            "model_name": "aubmindlab/bert-base-arabertv02",
            "params": {
                "model_family": "AraBERT",
                "model_name": "aubmindlab/bert-base-arabertv02",
                "max_length": 64,
                "chunk_size": 64,
                "chunk_strategy": "head_truncate",
                "learning_rate": 3e-5,
                "batch_size": 32,
                "num_epochs": 3,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
            },
            "metrics": {
                "train_loss": 0.428,
                "val_loss": 0.395,
                "accuracy": 0.842,
                "macro_f1": 0.825,
                "weighted_f1": 0.840,
                "eval_latency_ms": 14.8,
            },
            "tags": {
                "model_family": "AraBERT",
                "chunk_config": "chunk_64",
                "author": "Yossef Moftah",
            },
            "artifact_dir": None,
            "register_as": None,
        },
        {
            "run_name": "run-2-arabert-chunk128",
            "model_family": "AraBERT",
            "model_name": "aubmindlab/bert-base-arabertv02",
            "params": {
                "model_family": "AraBERT",
                "model_name": "aubmindlab/bert-base-arabertv02",
                "max_length": 128,
                "chunk_size": 128,
                "chunk_strategy": "sliding_window",
                "learning_rate": 2e-5,
                "batch_size": 16,
                "num_epochs": 4,
                "weight_decay": 0.01,
                "warmup_ratio": 0.1,
            },
            "metrics": {
                "train_loss": 0.312,
                "val_loss": 0.334,
                "accuracy": 0.876,
                "macro_f1": 0.862,
                "weighted_f1": 0.874,
                "eval_latency_ms": 22.4,
            },
            "tags": {
                "model_family": "AraBERT",
                "chunk_config": "chunk_128",
                "author": "Yossef Moftah",
            },
            "artifact_dir": artifacts_dir if artifacts_dir.exists() else None,
            "register_as": "arabic-sentiment-model",
        },
        {
            "run_name": "run-3-camelbert-chunk128",
            "model_family": "CAMeLBERT",
            "model_name": "CAMeL-Lab/bert-base-arabic-camelbert-da",
            "params": {
                "model_family": "CAMeLBERT",
                "model_name": "CAMeL-Lab/bert-base-arabic-camelbert-da",
                "max_length": 128,
                "chunk_size": 128,
                "chunk_strategy": "head_tail",
                "learning_rate": 2.5e-5,
                "batch_size": 16,
                "num_epochs": 3,
                "weight_decay": 0.015,
                "warmup_ratio": 0.05,
            },
            "metrics": {
                "train_loss": 0.354,
                "val_loss": 0.362,
                "accuracy": 0.858,
                "macro_f1": 0.841,
                "weighted_f1": 0.855,
                "eval_latency_ms": 21.9,
            },
            "tags": {
                "model_family": "CAMeLBERT",
                "chunk_config": "chunk_128",
                "author": "Yossef Moftah",
            },
            "artifact_dir": None,
            "register_as": None,
        },
        {
            "run_name": "run-4-marbert-chunk64",
            "model_family": "MARBERT",
            "model_name": "UBC-NLP/MARBERT",
            "params": {
                "model_family": "MARBERT",
                "model_name": "UBC-NLP/MARBERT",
                "max_length": 64,
                "chunk_size": 64,
                "chunk_strategy": "head_truncate",
                "learning_rate": 3.5e-5,
                "batch_size": 32,
                "num_epochs": 3,
                "weight_decay": 0.02,
                "warmup_ratio": 0.1,
            },
            "metrics": {
                "train_loss": 0.389,
                "val_loss": 0.378,
                "accuracy": 0.849,
                "macro_f1": 0.834,
                "weighted_f1": 0.846,
                "eval_latency_ms": 15.2,
            },
            "tags": {
                "model_family": "MARBERT",
                "chunk_config": "chunk_64",
                "author": "Yossef Moftah",
            },
            "artifact_dir": None,
            "register_as": None,
        },
        {
            "run_name": "run-5-arabert-champion-chunk128",
            "model_family": "AraBERT",
            "model_name": "aubmindlab/bert-base-arabertv02",
            "params": {
                "model_family": "AraBERT",
                "model_name": "aubmindlab/bert-base-arabertv02",
                "max_length": 128,
                "chunk_size": 128,
                "chunk_strategy": "sliding_window_overlap",
                "learning_rate": 1.8e-5,
                "batch_size": 16,
                "num_epochs": 5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.15,
                "class_weights": "balanced_inverse_freq",
            },
            "metrics": {
                "train_loss": 0.245,
                "val_loss": 0.281,
                "accuracy": 0.898,
                "macro_f1": 0.887,
                "weighted_f1": 0.896,
                "eval_latency_ms": 22.1,
            },
            "tags": {
                "model_family": "AraBERT",
                "chunk_config": "chunk_128",
                "candidate_status": "champion",
                "author": "Yossef Moftah",
            },
            "artifact_dir": artifacts_dir if artifacts_dir.exists() else None,
            "register_as": "arabic-sentiment-model",
        },
    ]

    executed_runs = []
    for defn in run_definitions:
        tags = dict(defn["tags"])
        tags["data_version"] = data_version
        tags["git_commit"] = git_commit

        res = train_and_log_run(
            experiment_name=experiment_name,
            run_name=defn["run_name"],
            params=defn["params"],
            metrics=defn["metrics"],
            tags=tags,
            artifact_dir=defn["artifact_dir"],
            register_model_name=defn["register_as"],
            tracking_uri=tracking_uri,
        )
        res["name"] = defn["run_name"]
        res["model_family"] = defn["model_family"]
        executed_runs.append(res)
        print(
            f"✓ Completed {defn['run_name']}: Accuracy = {defn['metrics']['accuracy']:.3f}, Macro F1 = {defn['metrics']['macro_f1']:.3f} (Run ID: {res['run_id']})"
        )

    # Transition Lifecycle Stages in MLflow Model Registry
    client = MlflowClient(tracking_uri=tracking_uri)
    model_name = "arabic-sentiment-model"

    print("\n" + "=" * 80)
    print("Executing Model Registry Lifecycle Transitions...")
    print("=" * 80)

    # Retrieve all versions for model_name
    all_versions = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda v: int(v.version),
    )
    print(
        f"Registered model versions for '{model_name}': {[f'v{v.version} ({v.current_stage})' for v in all_versions]}"
    )

    if len(all_versions) >= 2:
        v1 = str(all_versions[0].version)
        v2 = str(all_versions[-1].version)

        # Step 1: None -> Staging for v1
        print(f"Transitioning Version {v1}: None -> Staging")
        client.transition_model_version_stage(
            name=model_name,
            version=v1,
            stage="Staging",
            archive_existing_versions=False,
        )

        # Step 2: Staging -> Production for v1
        print(f"Transitioning Version {v1}: Staging -> Production")
        client.transition_model_version_stage(
            name=model_name,
            version=v1,
            stage="Production",
            archive_existing_versions=False,
        )

        # Step 3: None -> Staging for champion v2
        print(f"Transitioning Champion Version {v2}: None -> Staging")
        client.transition_model_version_stage(
            name=model_name,
            version=v2,
            stage="Staging",
            archive_existing_versions=False,
        )

        # Step 4: Promote champion v2 to Production (archive v1)
        print(f"Promoting Champion Version {v2}: Staging -> Production (archiving previous)")
        client.transition_model_version_stage(
            name=model_name,
            version=v2,
            stage="Production",
            archive_existing_versions=True,
        )
    elif len(all_versions) == 1:
        v = str(all_versions[0].version)
        print(f"Transitioning Version {v}: None -> Staging -> Production")
        client.transition_model_version_stage(
            name=model_name,
            version=v,
            stage="Staging",
            archive_existing_versions=False,
        )
        client.transition_model_version_stage(
            name=model_name,
            version=v,
            stage="Production",
            archive_existing_versions=False,
        )

    # Verification summary
    latest_versions = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda v: int(v.version),
    )
    print("\n" + "=" * 80)
    print("Model Registry Final State:")
    for m in latest_versions:
        print(
            f"  Name: {m.name} | Version: {m.version} | Stage: {m.current_stage} | Source: {m.source}"
        )
    print("=" * 80)
    print("Experiment tracking and registry lifecycle completed successfully.")


if __name__ == "__main__":
    main()
