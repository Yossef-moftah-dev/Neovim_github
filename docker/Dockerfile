# ==============================================================================
# Multi-Stage Dockerfile for ProdML Arabic Sentiment Service
# Stage 1: Build stage (compiles wheels and installs virtualenv)
# Stage 2: Runtime stage (lean, non-root user appuser, healthcheck enabled)
# ==============================================================================

# Stage 1: Builder
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package definition and install production dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install prod dependencies + CPU-optimized PyTorch
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install .

# ==============================================================================
# Stage 2: Runtime
# ==============================================================================
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app/src:$PYTHONPATH" \
    PORT=8000 \
    HOST=0.0.0.0 \
    MODEL_DIR="/app/outputs/final_model"

LABEL org.opencontainers.image.title="ProdML Arabic Sentiment Service" \
      org.opencontainers.image.description="Production Arabic sentiment analysis service powered by AraBERT with full observability and monitoring" \
      org.opencontainers.image.url="https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert" \
      org.opencontainers.image.source="https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert" \
      org.opencontainers.image.vendor="ProdML" \
      org.opencontainers.image.licenses="MIT"

# Install curl for Docker healthcheck and clean apt caches
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user and group (appuser:10001)
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -s /bin/bash appuser

WORKDIR /app

# Copy isolated virtual environment from builder stage
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv

# Copy source code and package files
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup pyproject.toml README.md ./

# Create artifact directory with proper permissions for mounted volumes
RUN mkdir -p /app/outputs/final_model && chown -R appuser:appgroup /app

# Configure Docker Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Enforce non-root execution
USER appuser

EXPOSE 8000

# Launch FastAPI service via Uvicorn
CMD ["uvicorn", "prodml.api:app", "--host", "0.0.0.0", "--port", "8000"]
