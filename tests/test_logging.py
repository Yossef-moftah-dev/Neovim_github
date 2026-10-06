"""Tests for structured JSON logging and correlation IDs."""

from __future__ import annotations

import json
import logging

from prodml.logging import (
    JSONFormatter,
    configure_logging,
    get_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)


def test_correlation_id_context() -> None:
    assert get_correlation_id() is None
    token = set_correlation_id("test-corr-123")
    assert get_correlation_id() == "test-corr-123"
    reset_correlation_id(token)
    assert get_correlation_id() is None


def test_json_formatter_output() -> None:
    formatter = JSONFormatter()
    token = set_correlation_id("test-corr-456")

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=25,
        msg="Sample log message with %s",
        args=("argument",),
        exc_info=None,
    )
    record.custom_field = "custom_value"  # type: ignore[attr-defined]

    output = formatter.format(record)
    reset_correlation_id(token)

    data = json.loads(output)
    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert data["message"] == "Sample log message with argument"
    assert data["correlation_id"] == "test-corr-456"
    assert data["extra"]["custom_field"] == "custom_value"
    assert "timestamp" in data


def test_configure_logging() -> None:
    configure_logging("DEBUG")
    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG
    assert any(isinstance(h.formatter, JSONFormatter) for h in root_logger.handlers)
