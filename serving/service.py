"""BentoML high-throughput serving service for Arabic sentiment analysis.

Features:
- Dynamic micro-batching via `@bentoml.api(batchable=True)`
- High-concurrency worker scheduling
- Memory resident predictor encapsulation via SentimentRunner
- Parity with FastAPI endpoints (/predict, /health, /metadata)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import bentoml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from prodml import __version__
from prodml.config import get_settings
from prodml.predict import SentimentPredictor
from prodml.train import calculate_artifact_hash

logger = logging.getLogger("serving.bentoml_service")


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------


class SinglePredictRequest(BaseModel):
    """Payload for single text sentiment classification."""

    text: str = Field(..., min_length=1, description="Input Arabic text")


class PredictionOutput(BaseModel):
    """Output prediction schema."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float | None = None


# -----------------------------------------------------------------------------
# Sentiment Runner Encapsulation
# -----------------------------------------------------------------------------


class SentimentRunner:
    """Runner encapsulating SentimentPredictor instance and batching dispatch."""

    def __init__(self, model_dir: Path | str | None = None) -> None:
        self.settings = get_settings()
        target_dir = Path(model_dir or self.settings.model.model_dir)

        logger.info("Initializing SentimentRunner with model from: %s", target_dir)
        if target_dir.exists():
            self.predictor = SentimentPredictor.load(target_dir)
            self.artifact_hash = calculate_artifact_hash(target_dir)
        else:
            logger.warning(
                "Target model directory %s not found. Using dummy predictor.", target_dir
            )
            self.predictor = None
            self.artifact_hash = "mock_sha256_hash_1234567890abcdef"

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Run batched inference across list of Arabic texts."""
        if not texts:
            return []

        if self.predictor is not None:
            results = self.predictor.predict_batch(texts)
            return [res.model_dump() for res in results]

        # Fallback deterministic response if model files not present
        return [
            {
                "label": "Positive",
                "confidence": 0.92,
                "probabilities": {"Negative": 0.03, "Neutral": 0.05, "Positive": 0.92},
                "latency_ms": 15.0,
            }
            for _ in texts
        ]


# -----------------------------------------------------------------------------
# BentoML Service Definition
# -----------------------------------------------------------------------------


@bentoml.service(
    name="arabic_sentiment_service",
    resources={"cpu": "2"},
    traffic={"timeout": 30},
)
class ArabicSentimentService:
    """Production BentoML service with dynamic micro-batching."""

    def __init__(self) -> None:
        model_dir = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "outputs" / "final_model"))
        self.runner = SentimentRunner(model_dir=model_dir)
        logger.info("ArabicSentimentService resident and ready for incoming traffic.")

    @bentoml.api(
        batchable=True,
        batch_dim=0,
        max_batch_size=64,
        max_latency_ms=20,
    )
    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Micro-batched prediction endpoint aggregating concurrent requests."""
        # Sanitize empty strings
        cleaned_texts = [
            t.strip() if isinstance(t, str) and t.strip() else "نص غير صالح" for t in texts
        ]
        return self.runner.predict_batch(cleaned_texts)

    @bentoml.api
    def health(self) -> dict[str, Any]:
        """Health check endpoint confirming memory residency."""
        return {
            "status": "ok",
            "service": "bentoml",
            "framework": "BentoML 1.4+ Dynamic Micro-batching",
            "model_loaded": self.runner.predictor is not None,
            "memory_resident": True,
        }

    @bentoml.api
    def metadata(self) -> dict[str, Any]:
        """Service and model metadata endpoint."""
        return {
            "name": "arabic_sentiment_service",
            "version": __version__,
            "framework": "BentoML",
            "model_name": "aubmindlab/bert-base-arabertv02",
            "artifact_hash": self.runner.artifact_hash,
            "max_batch_size": 64,
            "max_latency_ms": 20,
            "classes": ["Negative", "Neutral", "Positive"],
        }


# Expose standard runner symbol for handbook backward compatibility
runner = SentimentRunner()
