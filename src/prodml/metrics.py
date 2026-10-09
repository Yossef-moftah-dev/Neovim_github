"""Prometheus telemetry instrumentation for ProdML serving tier.

Features:
- Multiprocess mode handling via PROMETHEUS_MULTIPROC_DIR
- Metric contract: request rate, latency histograms, prediction counters,
  confidence quantiles, batch size distributions, and statistical drift scores
- Support for both single-process and multi-worker (Uvicorn / Gunicorn / BentoML) deployments
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger("prodml.metrics")

# Default latency and confidence buckets
LATENCY_BUCKETS = (0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5, 5.0)
CONFIDENCE_BUCKETS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.0)
BATCH_SIZE_BUCKETS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)


def setup_multiproc_dir() -> Path | None:
    """Ensure PROMETHEUS_MULTIPROC_DIR exists if specified in environment."""
    multiproc_env = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if not multiproc_env:
        return None

    multiproc_path = Path(multiproc_env)
    multiproc_path.mkdir(parents=True, exist_ok=True)
    return multiproc_path


def cleanup_multiproc_dir() -> None:
    """Clean obsolete worker files in PROMETHEUS_MULTIPROC_DIR on master startup."""
    multiproc_env = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_env and Path(multiproc_env).exists():
        for item in Path(multiproc_env).glob("*.db"):
            try:
                item.unlink(missing_ok=True)
            except OSError:
                pass


# Primary metric registry
REGISTRY = CollectorRegistry(auto_describe=True)

# Metric Contract Instrumentation
HTTP_REQUESTS_TOTAL = Counter(
    "prodml_http_requests_total",
    "Total number of HTTP requests processed by endpoint and status",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "prodml_http_request_duration_seconds",
    "HTTP request latency distribution in seconds",
    ["method", "endpoint"],
    buckets=LATENCY_BUCKETS,
    registry=REGISTRY,
)

PREDICTIONS_TOTAL = Counter(
    "prodml_predictions_total",
    "Total sentiment predictions classified by model version and class label",
    ["model_version", "predicted_label"],
    registry=REGISTRY,
)

PREDICTION_CONFIDENCE = Histogram(
    "prodml_prediction_confidence",
    "Model confidence score distribution across predicted classes",
    ["model_version", "predicted_label"],
    buckets=CONFIDENCE_BUCKETS,
    registry=REGISTRY,
)

BATCH_SIZE_DISTRIBUTION = Histogram(
    "prodml_batch_size",
    "Distribution of processed dynamic micro-batch sizes",
    buckets=BATCH_SIZE_BUCKETS,
    registry=REGISTRY,
)

DRIFT_SCORE = Gauge(
    "prodml_drift_score",
    "Current statistical drift score for tracked feature or output",
    ["metric", "feature"],
    registry=REGISTRY,
    multiprocess_mode="mostrecent",
)

DRIFT_DETECTED = Gauge(
    "prodml_drift_detected",
    "Boolean indicator of statistical drift threshold breach (1=drifted, 0=stable)",
    ["metric", "feature"],
    registry=REGISTRY,
    multiprocess_mode="mostrecent",
)

MODEL_LOADED = Gauge(
    "prodml_model_loaded",
    "Model memory residency readiness status (1=resident, 0=unloaded)",
    ["model_version"],
    registry=REGISTRY,
    multiprocess_mode="mostrecent",
)


def record_prediction_metrics(
    label: str,
    confidence: float,
    model_version: str = "v1",
) -> None:
    """Record single prediction counter and confidence histogram observation."""
    safe_label = str(label)
    safe_version = str(model_version or "v1")
    PREDICTIONS_TOTAL.labels(model_version=safe_version, predicted_label=safe_label).inc()
    PREDICTION_CONFIDENCE.labels(model_version=safe_version, predicted_label=safe_label).observe(
        float(confidence)
    )


def record_batch_prediction_metrics(
    results: list[dict[str, Any]],
    model_version: str = "v1",
) -> None:
    """Record batch prediction counters, confidences, and batch size."""
    if not results:
        return
    BATCH_SIZE_DISTRIBUTION.observe(float(len(results)))
    for res in results:
        label = res.get("label", "unknown")
        confidence = float(res.get("confidence", 0.0))
        record_prediction_metrics(label=label, confidence=confidence, model_version=model_version)


def record_drift_metrics(
    metric_name: str,
    feature: str,
    score: float,
    is_drifted: bool,
) -> None:
    """Publish evaluated drift score and breach flag to Prometheus gauges."""
    DRIFT_SCORE.labels(metric=metric_name, feature=feature).set(float(score))
    DRIFT_DETECTED.labels(metric=metric_name, feature=feature).set(1.0 if is_drifted else 0.0)


def get_latest_metrics() -> tuple[bytes, str]:
    """Generate Prometheus exposition text format payload, handling multiprocess mode if set."""
    multiproc_env = os.getenv("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_env and Path(multiproc_env).exists():
        from prometheus_client import multiprocess

        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest(REGISTRY)

    return data, CONTENT_TYPE_LATEST
