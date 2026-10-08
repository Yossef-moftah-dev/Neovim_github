"""Test suite for Model Registry loading and code-free model swap."""

from __future__ import annotations

import os

import pytest

from prodml.predict import SentimentPredictor


def _is_server_reachable(uri: str) -> bool:
    import socket
    from urllib.parse import urlparse

    try:
        parsed = urlparse(uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@pytest.mark.skipif(
    os.getenv("SKIP_DOCKER_TESTS") == "1",
    reason="Skipping tests requiring local MLflow server",
)
def test_load_from_model_registry_stage() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    if not _is_server_reachable(tracking_uri):
        pytest.skip(f"MLflow server not reachable at {tracking_uri}")

    stage_uri = "models:/arabic-sentiment-model/Production"

    try:
        predictor = SentimentPredictor.load(stage_uri, tracking_uri=tracking_uri)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MLflow server or registry not reachable: {exc}")

    if predictor.model_version is None:
        pytest.skip("Model not found in registry (fallback to local artifact occurred)")

    assert predictor.is_loaded()
    assert predictor.model_version is not None
    assert predictor.model_stage == "Production"

    res = predictor.predict_one("المنتج جميل جدا ومريح")
    assert res.label in ["Negative", "Neutral", "Positive"]
    assert 0.0 <= res.confidence <= 1.0
