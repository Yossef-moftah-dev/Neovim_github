"""Tests for training routines and artifact checksum calculation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from prodml.train import (
    calculate_artifact_hash,
    compute_class_weights,
    evaluate_model,
)


def test_calculate_artifact_hash_real_dir(tmp_path: Path) -> None:
    model_dir = Path(__file__).resolve().parent.parent / "outputs" / "final_model"
    if not model_dir.exists() or not any(model_dir.iterdir()):
        mock_dir = tmp_path / "mock_model"
        mock_dir.mkdir()
        (mock_dir / "config.json").write_text('{"test": true}', encoding="utf-8")
        model_dir = mock_dir

    digest = calculate_artifact_hash(model_dir)
    assert isinstance(digest, str)
    assert len(digest) == 64  # SHA-256 is 64 hex characters


def test_calculate_artifact_hash_nonexistent(tmp_path: Path) -> None:
    digest = calculate_artifact_hash(tmp_path / "empty")
    assert digest == "unknown"


def test_compute_class_weights() -> None:
    labels = [0, 0, 1, 2, 2, 2]
    weights = compute_class_weights(labels, num_classes=3)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (3,)
    assert weights[1] > weights[0]  # Class 1 is least frequent, so it has higher weight
    assert weights[0] > weights[2]  # Class 0 is less frequent than Class 2


def test_evaluate_model_dummy() -> None:
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
            logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
            from types import SimpleNamespace

            return SimpleNamespace(logits=logits)

    dataloader = [
        {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            "labels": torch.tensor([0, 1]),
        }
    ]
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = evaluate_model(DummyModel(), dataloader, torch.device("cpu"), loss_fn)
    assert metrics["accuracy"] == 1.0
    assert metrics["loss"] >= 0.0
