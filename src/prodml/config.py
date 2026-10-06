"""Configuration settings for ProdML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class ModelConfig(BaseModel):
    """Configuration for model artifacts and inference hyperparameters."""

    model_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv("MODEL_DIR", str(DEFAULT_PROJECT_ROOT / "outputs" / "final_model"))
        )
    )
    model_name: str = "aubmindlab/bert-base-arabertv02"
    max_length: int = 128
    device: str = Field(
        default_factory=lambda: os.getenv(
            "DEVICE", "cuda" if os.getenv("USE_CUDA") == "1" else "cpu"
        )
    )
    onnx_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "ONNX_PATH",
                str(DEFAULT_PROJECT_ROOT / "outputs" / "final_model" / "model.onnx"),
            )
        )
    )
    id2label: dict[int, str] = {
        0: "Negative",
        1: "Neutral",
        2: "Positive",
    }
    label2id: dict[str, int] = {
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2,
    }


class ServiceConfig(BaseModel):
    """Configuration for FastAPI service and logging."""

    app_name: str = "prodml-arabic-sentiment"
    app_version: str = "0.1.0"
    host: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_batch_size: int = 64


class AppSettings(BaseModel):
    """Unified application settings."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    project_root: Path = DEFAULT_PROJECT_ROOT


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings singleton."""
    return AppSettings()
