# Milestone 01 Report — Foundation, Packaging & Docker Service

> **Reference:** *The MLOps Practitioner Handbook*, Module 1, Pages 9–17  
> **Git Milestone Branch:** `module-1-packaging`  
> **Release Tag:** `v0.1.0`  
> **Author:** Yossef Moftah  
> **Date:** October 2026  

---

## 1. Executive Summary

Milestone 01 successfully transforms unstructured Arabic sentiment analysis prototypes and exploratory Jupyter notebooks into an industrial-grade, pip-installable Python package (`prodml`). The system is exposed as a fully asynchronous, tested FastAPI microservice, equipped with structured JSON observability and correlation tracking, containerized through a hardened multi-stage Docker build, and verifiable in **exactly 3 terminal commands**.

---

## 2. Transformation Overview: Prototype to Production

### 2.1 The Baseline Prototype State
Before this milestone, the project was organized as a collection of exploratory scripts and ad-hoc code:
- Prototype inference code was mixed with exploratory training logic (`src/DataPipline.py`, `src/inference.py`, `src/api.py`, `src/main.py`).
- Raw `print()` statements were scattered across scripts without timestamps or correlation contexts.
- Configuration parameters, model paths, and batch thresholds were hardcoded.
- Lacked formalized testing and test coverage metrics.
- Service dependencies and runtimes were coupled with local machine environments.

### 2.2 Refactored Modular Package Layout (`src/prodml/`)
The codebase was decomposed into single-responsibility modules adhering to strict typing and object-oriented principles:

```text
src/prodml/
├── __init__.py        # Package exports and version metadata (v0.1.0)
├── config.py          # Strongly typed Pydantic models & environment configuration
├── data.py            # CSV data ingestion, schema validation, and star-to-sentiment mappings
├── features.py        # Arabic text normalization, diacritics removal, and tokenization
├── timing.py          # Generic execution latency measurement decorator (@timed)
├── logging.py         # Structured JSON logging formatter with ContextVar correlation ID tracking
├── middleware.py      # Starlette/FastAPI correlation ID injection (uuid4) and latency logging
├── predict.py         # OOP SentimentPredictor (.load, .predict_one, .predict_batch, ONNX parity)
├── train.py           # Training metrics, loss weighting, and SHA-256 weight hashing
└── api.py             # Production FastAPI service with lifespan model resident management
```

### 2.3 Key Architectural Patterns Implemented
1. **Zero Raw Print Policy**: Every raw `print()` was purged from `src/`. Logging is exclusively routed through `prodml.logging.JSONFormatter`.
2. **OOP Predictor Pattern**: `SentimentPredictor` encapsulates model weights, tokenizer, and device management. The model is loaded once at server startup via FastAPI's `lifespan` manager, preventing memory leaks and per-request disk I/O.
3. **Execution Timing Decorator (`@timed`)**: Decorator capturing function latency in milliseconds, attaching metadata, and publishing telemetry.
4. **Strict Typing**: Enforced type hints (`mypy` / Python 3.12+ syntax) across all public methods and schemas.

---

## 3. Observability & Telemetry

### 3.1 Structured JSON Logging
All logging messages are formatted as uniform single-line JSON objects adhering to standard cloud ingestion formats (e.g., Datadog, CloudWatch, Google Cloud Logging):

```json
{
  "timestamp": "2026-10-06T13:10:19.452109+00:00",
  "level": "INFO",
  "logger": "prodml.middleware",
  "message": "Completed request: POST /predict - status 200 in 1902.86 ms",
  "correlation_id": "9199a9c7-0364-42b1-acf6-ca73cc51b572",
  "extra": {
    "http_method": "POST",
    "http_path": "/predict",
    "status_code": 200,
    "latency_ms": 1902.86
  }
}
```

### 3.2 Distributed Request Tracing & Correlation IDs
- `CorrelationIdMiddleware` inspects incoming HTTP requests for `X-Request-ID` or `X-Correlation-ID`.
- When omitted, a unique `uuid4` string is dynamically assigned.
- The ID is stored in Python's thread-safe / task-safe `ContextVar`.
- Responses return the `X-Request-ID` header, enabling end-to-end distributed trace propagation across client-service interactions.

---

## 4. API Endpoints & Health Verification

The microservice exposes four primary endpoints:

| Endpoint | Method | Specification |
| :--- | :--- | :--- |
| `/health` | `GET` | Validates model memory residency. Returns `200 OK` when model is loaded; `503 Service Unavailable` if uninitialized. |
| `/metadata` | `GET` | Reports package version, model architecture, target classes, and deterministic SHA-256 model weight checksum. |
| `/predict` | `POST` | Processes single review text with Pydantic validation (rejects blank/whitespace strings with HTTP `422 Unprocessable Content`). |
| `/predict/batch` | `POST` | Processes lists of review texts (up to 64 items per request) in a single tensorized forward pass. |

### Model Checksum & Provenance
Model weights in `outputs/final_model/model.safetensors` were deterministically hashed:
- **Algorithm**: SHA-256
- **Hash**: `f27e1ea07dcb5395e281e9f1c3ea202a23794c1e2db8ca0e70c5a2aac3407951`
- Exposing this through `/metadata` prevents model serving silent drift and ensures model artifact reproducibility.

---

## 5. Verification & Test Suite Parity

### 5.1 Test Coverage Analysis
The automated pytest suite (`pytest --cov=src/prodml --cov-report=term-missing --cov-fail-under=70`) achieved **95.92% total test coverage**, significantly exceeding the handbook's 70% threshold.

| Module | Statements | Missing Lines | Coverage (%) |
| :--- | :---: | :---: | :---: |
| `src/prodml/__init__.py` | 1 | 0 | **100%** |
| `src/prodml/config.py` | 28 | 0 | **100%** |
| `src/prodml/timing.py` | 32 | 0 | **100%** |
| `src/prodml/data.py` | 43 | 0 | **100%** |
| `src/prodml/features.py` | 29 | 0 | **100%** |
| `src/prodml/logging.py` | 40 | 1 | **98%** |
| `src/prodml/predict.py` | 99 | 3 | **97%** |
| `src/prodml/train.py` | 60 | 2 | **97%** |
| `src/prodml/api.py` | 105 | 9 | **91%** |
| `src/prodml/middleware.py` | 29 | 4 | **86%** |
| **TOTAL** | **466** | **19** | **95.92%** |

### 5.2 Logit Parity: PyTorch Native vs. ONNX Runtime
- In `tests/test_logit_parity.py`, native PyTorch logits were compared directly against the exported ONNX Runtime model using dynamic batching.
- **Tolerance threshold**: $\le 10^{-4}$ ($1e-4$)
- **Observed Max Absolute Difference**: $< 2.4 \times 10^{-6}$
- **Result**: PASSED ($100\%$). Proves that conversion from PyTorch to optimized inference engines yields numerically identical predictions.

---

## 6. Containerization & Security Hardening

The multi-stage `docker/Dockerfile` separates build-time dependencies from the final production runtime:
1. **Stage 1 (`builder`)**:
   - Uses `python:3.12-slim`.
   - Compiles package wheels and installs dependencies into an isolated virtual environment (`/opt/venv`).
2. **Stage 2 (`runtime`)**:
   - Clean `python:3.12-slim` image without compiler toolchains (`gcc`, `make`, etc.).
   - Copies solely `/opt/venv` and `src/` from the builder stage.
   - Creates a dedicated non-root user `appuser` (`UID 10001`, `GID 10001`) preventing container privilege escalation.
   - Configures native Docker `HEALTHCHECK` with curl polling `http://localhost:8000/health`.
   - Image tagged with `latest` and commit hash `sha-1611b4d`.

---

## 7. MLOps Maturity Self-Assessment

In accordance with the MLOps Practitioner Handbook maturity model (transitioning from Level 0 to Level 1):

| Dimension | Baseline (Level 0) | Milestone 01 State (Level 1) | Score | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Packaging & Modularity** | Monolithic notebooks and unstructured scripts in root directory. | Fully isolated, pip-installable `prodml` package with `pyproject.toml` and standard layout. | **5/5** | Completed |
| **Code Hygiene & Typing** | Raw prints, loose types, untracked latencies. | Strict typing, PEP 695 type parameters, `@timed` metrics, zero raw prints. | **5/5** | Completed |
| **Test Automation** | No automated tests; manual visual inspection in notebooks. | Automated pytest suite with 34 tests, 95.92% coverage, and ONNX logit parity verification ($<10^{-4}$). | **5/5** | Completed |
| **Observability** | Standard out print logs without timestamps or tracing. | Structured JSON log format with ISO-8601 timestamps and `uuid4` correlation IDs in headers/context. | **5/5** | Completed |
| **Containerization** | Host-dependent local environment. | Multi-stage Docker container with non-root security (`appuser:10001`), HEALTHCHECK, and Docker Compose. | **5/5** | Completed |
| **Reproducibility** | Ad-hoc package installations and model weights. | Lockfile pinning (`uv.lock`), SHA-256 weight provenance verification via `/metadata`. | **5/5** | Completed |

**Maturity Assessment Summary**: The repository has transitioned from **Level 0 (Manual, ad-hoc experimentation)** to **Level 1 (Automated Pipeline & Production Packaging Foundation)**.

---

## 8. Milestone Verification Evidence

```bash
# Command 1: Clone repo
git clone git@github.com:Yossef-moftah-dev/arabic-sentiment-arabert.git && cd arabic-sentiment-arabert

# Command 2: Start service
docker compose up -d

# Command 3: Send test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع جدا وأنصح به"}'
```

### Live Test Outputs
- `GET /health` -> `200 OK` (`{"status":"ok","model_loaded":true,"memory_resident":true,"device":"cpu"}`)
- `POST /predict` (Valid) -> `200 OK` (`{"label":"Positive","confidence":0.8675,"probabilities":{...}}`)
- `POST /predict` (Blank) -> `422 Unprocessable Content` (`{"detail":[{"type":"value_error","loc":["body","text"],"msg":"Value error, Input text must not be empty or whitespace-only"}]}`)
- `GET /metadata` -> `200 OK` (Artifact hash: `f27e1ea07dcb5395e281e9f1c3ea202a23794c1e2db8ca0e70c5a2aac3407951`)
- Test Coverage -> `pytest --cov=src/prodml --cov-fail-under=70` passed with **95.92%**.

---

## 9. Conclusion & Next Steps for Module 2

Milestone 01 has fulfilled all functional, packaging, architectural, and verification criteria.

### Next Steps for Module 2:
1. **Continuous Integration (CI)**: Implement GitHub Actions workflows running `ruff`, `pytest --cov`, and docker build on pull requests.
2. **Model Registry & Tracking**: Integrate MLflow / DVC for tracking experiment runs, metrics, and weight versioning.
3. **Drift Detection & Data Monitoring**: Set up continuous inference monitoring to track data drift and latency percentiles (P95/P99).
