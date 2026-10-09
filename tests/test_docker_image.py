"""Integration tests for production Docker image (prodml-service:latest).

Validates static image configurations, container healthcheck lifecycle,
and live HTTP inference endpoints.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Generator

import httpx
import pytest

IMAGE_NAME = os.getenv("PRODML_IMAGE", "prodml-service:latest")
TEST_PORT = 8005
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


def is_docker_available() -> bool:
    """Return True if docker CLI is available and operational."""
    if not shutil.which("docker"):
        return False
    try:
        res = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def is_image_available(image: str) -> bool:
    """Return True if the target image is available locally."""
    try:
        res = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return res.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


pytestmark = [
    pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not available"),
    pytest.mark.skipif(
        not is_image_available(IMAGE_NAME),
        reason=f"Docker image '{IMAGE_NAME}' is not built locally",
    ),
]


def test_docker_image_static_metadata() -> None:
    """Verify static security and deployment metadata of the Docker image."""
    # Inspect user configuration
    proc = subprocess.run(
        ["docker", "inspect", "--format", "{{.Config.User}}", IMAGE_NAME],
        capture_output=True,
        text=True,
        check=True,
    )
    user = proc.stdout.strip()
    assert user in ("appuser", "10001", "10001:10001"), f"Unexpected non-root user: {user}"

    # Inspect exposed ports
    proc_ports = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Config.ExposedPorts}}", IMAGE_NAME],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "8000/tcp" in proc_ports.stdout, "Port 8000/tcp must be exposed"

    # Inspect healthcheck directive
    proc_hc = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Config.Healthcheck}}", IMAGE_NAME],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "curl" in proc_hc.stdout, "Dockerfile must define a curl-based HEALTHCHECK"


@pytest.fixture(scope="module")
def running_test_container() -> Generator[str, None, None]:
    """Launch test container and yield once healthy, tearing down afterwards."""
    container_name = f"pytest-prodml-test-{int(time.time())}"
    model_dir = (
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        + "/outputs/final_model"
    )

    # Start container
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "-p",
        f"{TEST_PORT}:8000",
        "-v",
        f"{model_dir}:/app/outputs/final_model:ro",
        IMAGE_NAME,
    ]
    subprocess.run(cmd, check=True)

    # Poll for healthy status
    max_wait = 90
    poll_interval = 2
    started = time.time()
    is_healthy = False

    try:
        while time.time() - started < max_wait:
            res = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            status = res.stdout.strip()
            if status == "healthy":
                is_healthy = True
                break
            try:
                r = httpx.get(f"{BASE_URL}/health", timeout=1.0)
                if r.status_code == 200:
                    is_healthy = True
                    break
            except httpx.RequestError:
                pass
            time.sleep(poll_interval)

        if not is_healthy:
            err_logs = subprocess.run(
                ["docker", "logs", container_name], capture_output=True, text=True, check=False
            ).stdout
            raise AssertionError(
                f"Container failed to become healthy within {max_wait}s. Logs:\n{err_logs}"
            )
        yield container_name
    finally:
        # Tear down container cleanly
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, check=False)


def test_container_health_and_metadata(running_test_container: str) -> None:
    """Verify live HTTP /health and /metadata endpoints on running container."""
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # Health check
        res_health = client.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True
        assert health_data["memory_resident"] is True

        # Metadata check
        res_meta = client.get("/metadata")
        assert res_meta.status_code == 200
        meta_data = res_meta.json()
        assert meta_data["num_classes"] == 3
        assert "Negative" in meta_data["classes"]
        assert "Positive" in meta_data["classes"]


def test_container_prediction_endpoints(running_test_container: str) -> None:
    """Verify single and batch inference on the containerized service."""
    with httpx.Client(base_url=BASE_URL, timeout=15.0) as client:
        # Single inference
        single_payload = {"text": "هذا المنتج ممتاز جدا وتوصيل سريع"}
        res = client.post("/predict", json=single_payload)
        assert res.status_code == 200
        pred_data = res.json()
        assert "label" in pred_data
        assert "confidence" in pred_data
        assert "probabilities" in pred_data
        assert pred_data["confidence"] > 0.0

        # Batch inference
        batch_payload = {"texts": ["خدمة ممتازة", "سيء للغاية ولن أكررها"]}
        res_batch = client.post("/predict/batch", json=batch_payload)
        assert res_batch.status_code == 200
        batch_data = res_batch.json()
        assert len(batch_data) == 2
        assert all("label" in item for item in batch_data)


def test_container_validation_and_tracing(running_test_container: str) -> None:
    """Verify 422 error rejection and correlation ID passthrough."""
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # 422 validation
        res_invalid = client.post("/predict", json={"text": "   "})
        assert res_invalid.status_code == 422

        # Trace correlation ID
        test_trace_id = "smoke-pytest-trace-888"
        res_trace = client.get("/health", headers={"X-Request-ID": test_trace_id})
        assert res_trace.status_code == 200
        assert res_trace.headers.get("X-Request-ID") == test_trace_id
