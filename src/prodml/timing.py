"""Execution timing utilities."""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def timed[F: Callable[..., Any]](func: F) -> F:
    """Decorator to measure and log function execution time in milliseconds."""
    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                async_wrapper.last_duration_ms = duration_ms  # type: ignore[attr-defined]
                logger.debug(
                    "Executed %s in %.2f ms",
                    func.__qualname__,
                    duration_ms,
                    extra={"duration_ms": duration_ms, "function": func.__qualname__},
                )

        async_wrapper.last_duration_ms = 0.0  # type: ignore[attr-defined]
        return async_wrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            sync_wrapper.last_duration_ms = duration_ms  # type: ignore[attr-defined]
            logger.debug(
                "Executed %s in %.2f ms",
                func.__qualname__,
                duration_ms,
                extra={"duration_ms": duration_ms, "function": func.__qualname__},
            )

    sync_wrapper.last_duration_ms = 0.0  # type: ignore[attr-defined]
    return sync_wrapper  # type: ignore[return-value]
