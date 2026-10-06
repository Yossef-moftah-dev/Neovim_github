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
