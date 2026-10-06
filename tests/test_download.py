"""Tests for Google Drive model downloader module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from prodml.download import download_model, is_model_present


def test_is_model_present_false(tmp_path: Path) -> None:
    assert not is_model_present(tmp_path / "nonexistent")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert not is_model_present(empty_dir)

    # Missing some required files
    (empty_dir / "config.json").write_text("{}", encoding="utf-8")
    assert not is_model_present(empty_dir)


def test_is_model_present_true(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for fname in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
        (model_dir / fname).write_text("sample content", encoding="utf-8")

    assert is_model_present(model_dir)


def test_download_model_skips_when_present(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for fname in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
        (model_dir / fname).write_text("sample content", encoding="utf-8")

    with patch("prodml.download.logger") as mock_logger:
        result = download_model(model_dir=model_dir, force=False)
        assert result == model_dir
        mock_logger.info.assert_called_with(
            "All model artifacts already present in %s. Skipping download.", model_dir
        )
