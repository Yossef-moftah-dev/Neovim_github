"""Training routines and model artifact provenance calculation."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


def calculate_artifact_hash(model_dir: Path | str) -> str:
    """Compute deterministic SHA-256 hash for model weight files.

    Checks model.safetensors, then model.onnx, then pytorch_model.bin, then config.json.
    """
    directory = Path(model_dir)
    candidate_files = [
        directory / "model.safetensors",
        directory / "model.onnx",
        directory / "pytorch_model.bin",
        directory / "config.json",
    ]

    target_file: Path | None = None
    for candidate in candidate_files:
        if candidate.exists() and candidate.is_file():
            target_file = candidate
            break

    if target_file is None:
        logger.warning("No artifact weight file found in %s to hash", directory)
        return "unknown"

    hasher = hashlib.sha256()
    with target_file.open("rb") as f:
        # Read in 64KB chunks to maintain low memory footprint
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)

    digest = hasher.hexdigest()
    logger.debug("Artifact hash for %s: %s", target_file.name, digest)
    return digest


def compute_class_weights(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    """Compute balanced inverse-frequency class weights for loss function."""
    counts = [0] * num_classes
    for label in labels:
        if 0 <= label < num_classes:
            counts[label] += 1

    total = sum(counts)
    weights = []
    for count in counts:
        if count == 0:
            weights.append(1.0)
        else:
            weights.append(total / (num_classes * count))

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return weight_tensor / weight_tensor.mean()


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: Any,
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, float]:
    """Evaluate classification model on evaluation dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        logits = model(**batch).logits
        loss = loss_fn(logits.float(), labels)
        total_loss += loss.item() * len(labels)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total_samples += len(labels)

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    accuracy = correct / total_samples
    avg_loss = total_loss / total_samples
    logger.info("Evaluation results — Loss: %.4f, Accuracy: %.4f", avg_loss, accuracy)
    return {"loss": avg_loss, "accuracy": accuracy}


def get_git_commit() -> str:
    """Retrieve current Git commit SHA."""
    import os
    import subprocess

    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:  # noqa: BLE001
        return os.getenv("GITHUB_SHA", "unknown")


def get_data_version(data_path: Path | str) -> str:
    """Compute deterministic SHA-256 hash representing dataset version."""
    path = Path(data_path)
    if not path.exists():
        return "unknown"
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def train_and_log_run(
    experiment_name: str = "arabic-sentiment-classification",
    run_name: str | None = None,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    tags: dict[str, str] | None = None,
    artifact_dir: Path | str | None = None,
    register_model_name: str | None = None,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    """Execute or log training run wrapped inside MLflow start_run."""
    import json
    import os
    import tempfile

    import mlflow
    from mlflow.tracking import MlflowClient

    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")

    trk_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(trk_uri)
    mlflow.set_experiment(experiment_name)

    run_params = params or {}
    run_metrics = metrics or {}
    run_tags = tags or {}

    # Attach dynamic lineage tags
    run_tags.setdefault("git_commit", get_git_commit())
    run_tags.setdefault("framework", "PyTorch / Transformers")
    run_tags.setdefault("milestone", "v0.2.0")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("Started MLflow run %s (%s)", run_name or run_id, run_id)

        # 1. Log parameters
        for k, v in run_params.items():
            mlflow.log_param(k, v)

        # 2. Log tags
        for k, v in run_tags.items():
            mlflow.set_tag(k, v)

        # 3. Log metrics
        for k, v in run_metrics.items():
            mlflow.log_metric(k, float(v))

        # 4. Log metrics JSON summary artifact
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "eval_metrics.json"
            summary_payload = {
                "run_id": run_id,
                "run_name": run_name,
                "params": run_params,
                "metrics": run_metrics,
                "tags": run_tags,
            }
            summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(summary_path))

        # 5. Log model artifact directory if provided
        if artifact_dir is not None and Path(artifact_dir).exists():
            art_path = Path(artifact_dir)
            logger.info("Logging artifacts from %s", art_path)
            mlflow.log_artifacts(str(art_path), artifact_path="model")

        # 6. Model registry candidate registration
        registered_version: str | None = None
        if register_model_name:
            logger.info("Registering model candidate under name: %s", register_model_name)
            client = MlflowClient(tracking_uri=trk_uri)
            try:
                client.create_registered_model(register_model_name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Registered model may already exist: %s", exc)
            try:
                source_uri = f"s3://mlflow/{run.info.experiment_id}/{run_id}/artifacts/model"
                mv = client.create_model_version(
                    name=register_model_name,
                    source=source_uri,
                    run_id=run_id,
                    description=f"Candidate from run {run_name or run_id}",
                )
                registered_version = mv.version
                logger.info(
                    "Model registered successfully as '%s' version %s",
                    register_model_name,
                    registered_version,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to register model in registry: %s", exc)

        return {
            "run_id": run_id,
            "experiment_id": run.info.experiment_id,
            "params": run_params,
            "metrics": run_metrics,
            "tags": run_tags,
            "registered_version": registered_version,
        }
