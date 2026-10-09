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
from prodml.metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL

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
            duration_sec = time.perf_counter() - start_time
            duration_ms = duration_sec * 1000.0

            # Attach correlation ID to response headers
            response.headers[REQUEST_ID_HEADER] = correlation_id

            # Telemetry metrics collection
            try:
                HTTP_REQUESTS_TOTAL.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status=str(response.status_code),
                ).inc()
                HTTP_REQUEST_DURATION_SECONDS.labels(
                    method=request.method,
                    endpoint=request.url.path,
                ).observe(duration_sec)
            except Exception:  # noqa: BLE001, S110
                pass

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
            duration_sec = time.perf_counter() - start_time
            duration_ms = duration_sec * 1000.0

            try:
                HTTP_REQUESTS_TOTAL.labels(
                    method=request.method,
                    endpoint=request.url.path,
                    status="500",
                ).inc()
                HTTP_REQUEST_DURATION_SECONDS.labels(
                    method=request.method,
                    endpoint=request.url.path,
                ).observe(duration_sec)
            except Exception:  # noqa: BLE001, S110
                pass

            logger.exception(
                "Unhandled exception during %s %s in %.2f ms",
                request.method,
                request.url.path,
                duration_ms,
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "status_code": 500,
                    "latency_ms": duration_ms,
                },
            )
            raise
        finally:
            reset_correlation_id(token)
