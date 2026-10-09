"""Unit and integration tests for BentoML serving service and Runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "serving") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "serving"))

from serving.service import ArabicSentimentService, SentimentRunner


def test_sentiment_runner_fallback() -> None:
    """Verify SentimentRunner graceful fallback when weights directory is missing."""
    runner = SentimentRunner(model_dir="/non/existent/path")
    assert runner.predictor is None
    res = runner.predict_batch(["نص تجريبي"])
    assert len(res) == 1
    assert res[0]["label"] in {"Negative", "Neutral", "Positive"}
    assert "confidence" in res[0]
    assert "probabilities" in res[0]


def test_sentiment_runner_empty_input() -> None:
    """Verify SentimentRunner returns empty list for empty batch."""
    runner = SentimentRunner(model_dir="/non/existent/path")
    assert runner.predict_batch([]) == []


def test_bentoml_service_endpoints() -> None:
    """Verify health, metadata, and predict methods on ArabicSentimentService."""
    service = ArabicSentimentService()

    # Health Check
    health = service.health()
    assert health["status"] == "ok"
    assert health["service"] == "bentoml"
    assert health["memory_resident"] is True

    # Metadata
    meta = service.metadata()
    assert meta["name"] == "arabic_sentiment_service"
    assert meta["max_batch_size"] == 64
    assert meta["max_latency_ms"] == 20
    assert set(meta["classes"]) == {"Negative", "Neutral", "Positive"}

    # Predict
    preds = service.predict(["المنتج رائع وممتاز جدا", "تجربة سيئة للأسف"])
    assert len(preds) == 2
    assert preds[0]["label"] in {"Negative", "Neutral", "Positive"}
    assert preds[1]["label"] in {"Negative", "Neutral", "Positive"}


def test_bentoml_service_with_real_or_mock_predictor(fast_predictor) -> None:
    """Verify service prediction output with fixture predictor."""
    service = ArabicSentimentService()
    service.runner.predictor = fast_predictor

    texts = [
        "المنتج رااااائع جداً وأنصح بالشراء 😍!",
        "تجربة سيئة جداً وخامة رديئة للغاية 😡",
    ]
    preds = service.predict(texts)
    assert len(preds) == 2
    for p in preds:
        assert p["label"] in {"Negative", "Neutral", "Positive"}
        assert 0.0 <= p["confidence"] <= 1.0
        assert sum(p["probabilities"].values()) == pytest.approx(1.0, rel=1e-2)
