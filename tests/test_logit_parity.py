"""Logit parity tests comparing PyTorch native vs exported ONNX model."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from prodml.predict import SentimentPredictor


def test_native_vs_onnx_logit_parity(
    fast_predictor: SentimentPredictor,
    sample_arabic_texts: list[str],
    tmp_path: Path,
) -> None:
    """Assert logit parity between native PyTorch and exported ONNX runtime within 1e-4 tolerance."""
    onnx_file = tmp_path / "model.onnx"

    # Export to ONNX
    fast_predictor.export_onnx(onnx_file)
    assert onnx_file.exists(), "ONNX export failed to create artifact file"

    # Run native PyTorch inference
    native_logits = fast_predictor.predict_logits_native(sample_arabic_texts)

    # Run ONNX Runtime inference
    onnx_logits = fast_predictor.predict_logits_onnx(sample_arabic_texts, onnx_path=onnx_file)

    assert native_logits.shape == onnx_logits.shape, (
        f"Shape mismatch: native {native_logits.shape} vs ONNX {onnx_logits.shape}"
    )

    max_diff = float(np.max(np.abs(native_logits - onnx_logits)))
    tolerance = 1e-4

    assert max_diff < tolerance, (
        f"Logit parity violation! Max absolute difference {max_diff:.6e} exceeded tolerance {tolerance:.6e}"
    )
