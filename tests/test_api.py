"""Tests for FastAPI endpoints and middleware."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_endpoint_healthy(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["memory_resident"] is True
    assert "X-Request-ID" in response.headers


def test_health_endpoint_unloaded(client: TestClient) -> None:
    # Simulate unready/unloaded model state
    client.app.state.predictor = None
    response = client.get("/health")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["model_loaded"] is False


def test_metadata_endpoint(client: TestClient) -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "0.4.0"
    assert data["framework"] == "PyTorch / HuggingFace Transformers"
    assert data["num_classes"] == 3
    assert data["classes"] == ["Negative", "Neutral", "Positive"]
    assert "artifact_hash" in data


def test_predict_endpoint_success(client: TestClient) -> None:
    payload = {"text": "هذا المنتج رائع ومميز جداً"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert len(data["probabilities"]) == 3


def test_predict_endpoint_validation_errors(client: TestClient) -> None:
    # Blank string
    res_blank = client.post("/predict", json={"text": "   "})
    assert res_blank.status_code == 422

    # Empty string
    res_empty = client.post("/predict", json={"text": ""})
    assert res_empty.status_code == 422

    # Missing field
    res_missing = client.post("/predict", json={})
    assert res_missing.status_code == 422


def test_predict_batch_endpoint(client: TestClient) -> None:
    payload = {"texts": ["المنتج رائع", "الخدمة سيئة"]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all("label" in item for item in data)


def test_predict_batch_validation_error(client: TestClient) -> None:
    # Batch containing blank item
    payload = {"texts": ["المنتج رائع", "   "]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


def test_correlation_id_passthrough(client: TestClient) -> None:
    custom_id = "test-custom-req-id-9999"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id


def test_metrics_endpoint(client: TestClient) -> None:
    # Trigger a predict request to ensure metrics get populated
    client.post("/predict", json={"text": "المنتج رائع جدا"})
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "prodml_http_requests_total" in text
    assert "prodml_http_request_duration_seconds" in text
    assert "prodml_predictions_total" in text
