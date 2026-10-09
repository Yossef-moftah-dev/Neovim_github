"""Unit and integration tests for AcceleratedInferenceEngine."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "serving") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "serving"))

from serving.accelerated_engine import AcceleratedInferenceEngine, create_accelerated_app


def test_accelerated_engine_fallback() -> None:
    """Verify fallback behavior when ONNX model is missing."""
    engine = AcceleratedInferenceEngine(onnx_path="/non/existent/model.onnx")
    res = engine.infer_batch(["نص تجريبي رائع"])
    assert len(res) == 1
    assert res[0]["label"] in {"Negative", "Neutral", "Positive"}
    assert "probabilities" in res[0]
    assert engine.infer_batch([]) == []


def test_accelerated_api_endpoints() -> None:
    """Verify FastAPI endpoints using TestClient."""
    app = create_accelerated_app()
    with TestClient(app) as client:
        # Health check
        h_resp = client.get("/health")
        assert h_resp.status_code == 200
        assert h_resp.json()["status"] == "ok"
        assert "provider" in h_resp.json()

        # Metadata
        m_resp = client.get("/metadata")
        assert m_resp.status_code == 200
        assert m_resp.json()["framework"] == "ONNX Runtime"
        assert m_resp.json()["dynamic_batching"] is True

        # Single prediction
        p_resp = client.post("/predict", json={"text": "المنتج رائع جدا وأنصح به بشدة"})
        assert p_resp.status_code == 200
        pred = p_resp.json()
        assert pred["label"] in {"Negative", "Neutral", "Positive"}
        assert 0.0 <= pred["confidence"] <= 1.0

        # Batch prediction
        b_resp = client.post("/predict/batch", json={"texts": ["ممتاز جدا", "سيء للغاية"]})
        assert b_resp.status_code == 200
        assert len(b_resp.json()) == 2
