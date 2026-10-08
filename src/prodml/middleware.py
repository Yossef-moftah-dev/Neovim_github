"""FastAPI middleware for correlation ID tracking and request logging."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from prodml.logging import reset_correlation_id, set_correlation_id

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a correlation ID to every request and logs request metadata."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        # Extract or generate correlation ID
        correlation_id = (
            request.headers.get(REQUEST_ID_HEADER)
            or request.headers.get(CORRELATION_ID_HEADER)
            or str(uuid.uuid4())
        )

        token = set_correlation_id(correlation_id)
        start_time = time.perf_counter()

        logger.info(
            "Incoming request: %s %s",
            request.method,
            request.url.path,
            extra={
                "http_method": request.method,
                "http_path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        try:
            response: Response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0

            # Attach correlation ID to response headers
            response.headers[REQUEST_ID_HEADER] = correlation_id

            logger.info(
                "Completed request: %s %s - status %d in %.2f ms",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": duration_ms,
                },
            )
            return response
        except Exception:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception(
                "Unhandled exception during %s %s in %.2f ms",
                request.method,
                request.url.path,
                duration_ms,
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "latency_ms": duration_ms,
                },
            )
            raise
        finally:
            reset_correlation_id(token)
