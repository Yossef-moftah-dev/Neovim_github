"""Tests for configuration settings."""

from __future__ import annotations

import os
from pathlib import Path

from prodml.config import AppSettings, ModelConfig, ServiceConfig, get_settings


def test_default_config() -> None:
    settings = get_settings()
    assert isinstance(settings, AppSettings)
    assert settings.service.app_name == "prodml-arabic-sentiment"
    assert settings.service.app_version == "0.2.0"
    assert settings.model.max_length == 128
    assert len(settings.model.id2label) == 3


def test_custom_model_config() -> None:
    custom_cfg = ModelConfig(
        model_dir=Path("/custom/path"),
        max_length=64,
        device="cpu",
    )
    assert custom_cfg.max_length == 64
    assert custom_cfg.device == "cpu"
    assert custom_cfg.model_dir == Path("/custom/path")


def test_service_config_defaults() -> None:
    svc_cfg = ServiceConfig()
    assert svc_cfg.host in ("0.0.0.0", "127.0.0.1", os.getenv("HOST", "0.0.0.0"))
    assert svc_cfg.max_batch_size == 64
