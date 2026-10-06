"""FastAPI application serving production sentiment classification service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from prodml import __version__
from prodml.config import get_settings
from prodml.logging import configure_logging
from prodml.middleware import CorrelationIdMiddleware
from prodml.predict import PredictionResult, SentimentPredictor
from prodml.train import calculate_artifact_hash

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------


class PredictRequest(BaseModel):
    """Payload for single-item prediction."""

    text: str = Field(
        ...,
        min_length=1,
        description="Text content for sentiment classification",
        examples=["المنتج رائع جدا وأنصح به بشدة"],
    )

    @field_validator("text")
    @classmethod
    def validate_non_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input text must not be empty or whitespace-only")
        return v


class BatchPredictRequest(BaseModel):
    """Payload for batch predictions."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="List of text samples for batch sentiment classification",
    )

    @field_validator("texts")
    @classmethod
    def validate_items(cls, items: list[str]) -> list[str]:
        if not items:
            raise ValueError("Batch must contain at least 1 text item")
        for idx, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Item at index {idx} must not be empty or whitespace-only")
        return items


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status ('ok' or 'degraded')")
    model_loaded: bool = Field(..., description="True if model is resident in memory")
    memory_resident: bool = Field(..., description="Memory residency verification")
    device: str = Field(..., description="Inference compute device")


class MetadataResponse(BaseModel):
    """Service and model metadata schema."""

    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Package release version")
    framework: str = Field(..., description="ML inference framework")
    model_name: str = Field(..., description="Base model name")
    artifact_hash: str = Field(..., description="Deterministic SHA-256 hash of model weights")
    num_classes: int = Field(..., description="Number of target classes")
    classes: list[str] = Field(..., description="Class labels")


class PredictionResponse(BaseModel):
    """Prediction output schema."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float | None = None


# ---------------------------------------------------------
# Application Lifespan
# ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan by loading model into memory once at startup."""
    settings = get_settings()
    configure_logging(settings.service.log_level)
    if getattr(app.state, "predictor", None) is not None:
        logger.info("Predictor already initialized in application state.")
        yield
        return

    try:
        predictor = SentimentPredictor.load(
            model_dir=settings.model.model_dir,
            device=settings.model.device,
        )
        app.state.predictor = predictor
        app.state.artifact_hash = calculate_artifact_hash(settings.model.model_dir)
        logger.info(
            "Model successfully resident in memory. Artifact hash: %s",
            app.state.artifact_hash,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Model weights failed to load during startup (%s). Health checks will return 503 until resolved.",
            exc,
        )
        app.state.predictor = None
        app.state.artifact_hash = "not_loaded"

    yield

    logger.info("Cleaning up application resources during shutdown...")
    app.state.predictor = None


# ---------------------------------------------------------
# Application Instantiation & Routes
# ---------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    application = FastAPI(
        title="ProdML Arabic Sentiment Service",
        version=__version__,
        description="Production-grade Arabic Sentiment Classification API",
        lifespan=lifespan,
    )

    # Attach observability middleware
    application.add_middleware(CorrelationIdMiddleware)

    @application.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        responses={
            status.HTTP_200_OK: {"description": "Model is resident in memory and ready"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model not resident in memory"},
        },
    )
    def health_check(request: Request) -> Any:
        predictor: SentimentPredictor | None = getattr(request.app.state, "predictor", None)
        if predictor is None or not predictor.is_loaded():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "degraded",
                    "model_loaded": False,
                    "memory_resident": False,
                    "device": "none",
                },
            )

        return HealthResponse(
            status="ok",
            model_loaded=True,
            memory_resident=True,
            device=str(predictor.device),
        )

    @application.get(
        "/metadata",
        response_model=MetadataResponse,
        summary="Model & Artifact Metadata",
    )
    def get_metadata(request: Request) -> MetadataResponse:
        cfg = get_settings()
        predictor: SentimentPredictor | None = getattr(request.app.state, "predictor", None)
        artifact_hash: str = getattr(request.app.state, "artifact_hash", "unknown")

        classes = (
            list(predictor.id2label.values()) if predictor else list(cfg.model.id2label.values())
        )

        return MetadataResponse(
            name=cfg.service.app_name,
            version=__version__,
            framework="PyTorch / HuggingFace Transformers",
            model_name=cfg.model.model_name,
            artifact_hash=artifact_hash,
            num_classes=len(classes),
            classes=classes,
        )

    @application.post(
        "/predict",
        response_model=PredictionResponse,
        summary="Single Text Prediction",
        responses={
            status.HTTP_200_OK: {"description": "Successful sentiment prediction"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Payload validation error"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model is not loaded"},
        },
    )
    def predict(req: PredictRequest, request: Request) -> PredictionResponse:
        predictor: SentimentPredictor | None = getattr(request.app.state, "predictor", None)
        if predictor is None or not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not resident in memory",
            )

        result: PredictionResult = predictor.predict_one(req.text)
        return PredictionResponse(
            label=result.label,
            confidence=result.confidence,
            probabilities=result.probabilities,
            latency_ms=getattr(predictor.predict_one, "last_duration_ms", None),
        )

    @application.post(
        "/predict/batch",
        response_model=list[PredictionResponse],
        summary="Batch Text Prediction",
        responses={
            status.HTTP_200_OK: {"description": "Successful batch prediction list"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {"description": "Payload validation error"},
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Model is not loaded"},
        },
    )
    def predict_batch(req: BatchPredictRequest, request: Request) -> list[PredictionResponse]:
        predictor: SentimentPredictor | None = getattr(request.app.state, "predictor", None)
        if predictor is None or not predictor.is_loaded():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not resident in memory",
            )

        results: list[PredictionResult] = predictor.predict_batch(req.texts)
        duration_ms = getattr(predictor.predict_batch, "last_duration_ms", None)

        return [
            PredictionResponse(
                label=r.label,
                confidence=r.confidence,
                probabilities=r.probabilities,
                latency_ms=duration_ms,
            )
            for r in results
        ]

    return application


app = create_app()


def start() -> None:
    """Entry point to launch the uvicorn server."""
    settings = get_settings()
    uvicorn.run(
        "prodml.api:app",
        host=settings.service.host,
        port=settings.service.port,
        reload=False,
    )
