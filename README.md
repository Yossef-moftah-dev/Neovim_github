# ProdML — Arabic Sentiment Analysis Service

[![CI Test Coverage](https://img.shields.io/badge/Coverage-95.9%25-brightgreen.svg)](#-local-development-workflow)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.142+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Multi--Stage-2496ED.svg)](docker/Dockerfile)

Production-ready Arabic sentiment classification service built upon fine-tuned AraBERT (`aubmindlab/bert-base-arabertv02`). Transforms prototype notebook exploration into a modular, pip-installable Python package, wrapped in a high-performance FastAPI server with structured JSON observability, tested with $\ge 95\%$ coverage, and containerized inside a multi-stage Docker image.

---

## 🚀 Quickstart: Exactly 3 Commands

Start and verify the containerized service in **exactly three commands**:

```bash
# Command 1: Clone repo
git clone git@github.com:Yossef-moftah-dev/Neovim_github.git && cd Neovim_github

# Command 2: Start service
docker compose up -d

# Command 3: Send test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع جدا وأنصح به بشدة"}'
```

Expected JSON response:
```json
{
  "label": "Positive",
  "confidence": 0.8675,
  "probabilities": {
    "Negative": 0.0355,
    "Neutral": 0.0968,
    "Positive": 0.8675
  },
  "latency_ms": 42.15
}
```

---

## 🛠️ System Architecture

```text
               +-------------------------------------------------------+
               |                   Client Request                      |
               +-------------------------------------------------------+
                                          |
                                          v  [X-Request-ID Header]
+-----------------------------------------------------------------------------------------+
| FastAPI Application (prodml.api)                                                        |
|                                                                                         |
|   +---------------------------------------------------------------------------------+   |
|   | CorrelationIdMiddleware (ContextVar uuid4 injection, request latency tracking) |   |
|   +---------------------------------------------------------------------------------+   |
|                                          |                                              |
|                                          v                                              |
|   +-----------------------+   +-----------------------+   +-------------------------+   |
|   |      GET /health      |   |     GET /metadata     |   |      POST /predict      |   |
|   | (Checks memory model) |   | (Weights SHA-256 hash)|   | (Pydantic 422 validator)|   |
|   +-----------------------+   +-----------------------+   +-------------------------+   |
|                                                                    |                    |
|                                                                    v                    |
|   +---------------------------------------------------------------------------------+   |
|   | SentimentPredictor (Singleton resident in memory via lifespan)                  |   |
|   |   - features.normalize_arabic()                                                 |   |
|   |   - @timed predict_one() / predict_batch()                                      |   |
|   |   - ONNX Runtime Export & Logit Parity (<1e-4 tolerance)                        |   |
|   +---------------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------------+
| Structured JSON Logger (ISO-8601, log level, correlation_id, execution latency)         |
+-----------------------------------------------------------------------------------------+
```

---

## 📡 API Reference

| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Memory residency health check | `200 OK` (or `503` if uninitialized) |
| `GET` | `/metadata` | Release version, framework, and SHA-256 artifact hash | `200 OK` |
| `POST` | `/predict` | Single Arabic review sentiment inference | `200 OK` / `422 Unprocessable` |
| `POST` | `/predict/batch` | Batch review inference (max 64 items) | `200 OK` / `422 Unprocessable` |
| `GET` | `/docs` | Interactive Swagger UI documentation | `200 OK` |

### Sample Commands

#### 1. Check Service Health
```bash
curl -i http://localhost:8000/health
```
Response:
```http
HTTP/1.1 200 OK
x-request-id: 21aa9f18-d23c-461a-99f9-751d19cf851f

{"status":"ok","model_loaded":true,"memory_resident":true,"device":"cpu"}
```

#### 2. Query Model Metadata & Provenance
```bash
curl -s http://localhost:8000/metadata
```
Response:
```json
{
  "name": "prodml-arabic-sentiment",
  "version": "0.1.0",
  "framework": "PyTorch / HuggingFace Transformers",
  "model_name": "aubmindlab/bert-base-arabertv02",
  "artifact_hash": "f27e1ea07dcb5395e281e9f1c3ea202a23794c1e2db8ca0e70c5a2aac3407951",
  "num_classes": 3,
  "classes": ["Negative", "Neutral", "Positive"]
}
```

#### 3. Test Validation Error Handling (HTTP 422)
```bash
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "   "}'
```
Response:
```http
HTTP/1.1 422 Unprocessable Content
x-request-id: b5a2f595-b466-4a2b-aa4e-ab3b2d446fb5

{"detail":[{"type":"value_error","loc":["body","text"],"msg":"Value error, Input text must not be empty or whitespace-only","input":"   "}]}
```

---

## 💻 Local Development Workflow

### Prerequisites
- Python $\ge 3.12$
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- Docker & Docker Compose

### 1. Install Dependencies
```bash
# Install with dev dependencies using uv
uv sync

# Or with pip in editable mode
pip install -e ".[dev]"
```

### 2. Run Tests & Coverage
```bash
# Run pytest with >=70% coverage assertion
uv run pytest --cov=src/prodml --cov-report=term-missing --cov-fail-under=70
```

### 3. Code Quality & Formatting
```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

### 4. Run API Server Locally
```bash
uv run uvicorn prodml.api:app --reload --port 8000
```

---

## 🐳 Docker Deployment

The Docker image employs a multi-stage build:
- **Build Stage**: Installs compilation tools, resolves Python wheels, creates an isolated virtualenv `/opt/venv`.
- **Runtime Stage**: Lean `python:3.12-slim` base, non-root user `appuser` (`UID 10001`), Docker `HEALTHCHECK` with curl, and no compilers or test tooling.

```bash
# Build Docker image locally
docker build -t prodml-service:latest -t prodml-service:sha-$(git rev-parse --short HEAD) -f docker/Dockerfile .

# Run with Docker Compose
docker compose up -d

# Stop container
docker compose down
```
