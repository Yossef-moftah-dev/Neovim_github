"""Tests for SentimentPredictor OOP pattern."""

from __future__ import annotations

from pathlib import Path

import pytest

from prodml.predict import PredictionResult, SentimentPredictor


def test_predict_one_and_batch(fast_predictor: SentimentPredictor) -> None:
    assert fast_predictor.is_loaded()

    # Predict one
    res = fast_predictor.predict_one("المنتج ممتاز جدا")
    assert isinstance(res, PredictionResult)
    assert res.label in ("Negative", "Neutral", "Positive")
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.probabilities) == 3
    assert pytest.approx(sum(res.probabilities.values()), rel=1e-3) == 1.0

    # Predict batch
    batch_res = fast_predictor.predict_batch(["منتج رائع", "منتج سيء"])
    assert len(batch_res) == 2
    assert all(isinstance(r, PredictionResult) for r in batch_res)


def test_predict_batch_empty(fast_predictor: SentimentPredictor) -> None:
    assert fast_predictor.predict_batch([]) == []


def test_load_nonexistent_directory() -> None:
    with pytest.raises(FileNotFoundError):
        SentimentPredictor.load(model_dir=Path("/nonexistent/model/path"))
