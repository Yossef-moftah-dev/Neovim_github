# ProdML — Arabic Sentiment Analysis Service

[![Tests](https://img.shields.io/badge/Tests-34%20Passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-95.9%25-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](pyproject.toml)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker/Dockerfile)

Production-ready Arabic sentiment classification API powered by AraBERT, packaged into an installable Python module (`prodml`), fully covered with tests, and containerized with Docker.

---

## ⚡ Quickstart (Run in 3 Commands)

```bash
# 1. Clone repository
git clone git@github.com:Yossef-moftah-dev/Neovim_github.git && cd Neovim_github

# 2. Build and start containerized service
docker compose up -d --build

# 3. Send test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع جدا وأنصح به بشدة"}'
```

**Response:**
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

## 📡 Using the API

Interactive API documentation is available at **`http://localhost:8000/docs`**.

### 1. Single Text Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "الخدمة ممتازة والتوصيل سريع"}'
```

### 2. Batch Text Prediction (`POST /predict/batch`)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["المنتج رائع جدا", "تجربة سيئة ولن أكررها", "عادي لا بأس به"]}'
```

### 3. Health Check (`GET /health`)
Verifies that model weights are resident in memory:
```bash
curl -i http://localhost:8000/health
```

### 4. Model Metadata (`GET /metadata`)
Returns framework versions, class mappings, and SHA-256 weight checksum:
```bash
curl -s http://localhost:8000/metadata
```

---

## 🧪 Testing the Project

### 1. Setup Environment
```bash
# Using uv (fastest)
uv sync

# Or using standard pip inside virtualenv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run Test Suite with Coverage
Runs all 34 unit, integration, and logit parity tests and verifies $\ge 70\%$ coverage:
```bash
uv run pytest --cov=src/prodml --cov-report=term-missing --cov-fail-under=70
```

### 3. Run Specific Tests
```bash
# Test ONNX vs PyTorch logit parity (< 1e-4 tolerance)
uv run pytest tests/test_logit_parity.py -v

# Test API endpoints
uv run pytest tests/test_api.py -v
```

### 4. Code Quality & Linting
```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

---

## 💻 Local Development (Without Docker)

Run the FastAPI development server directly:
```bash
uv run uvicorn prodml.api:app --reload --port 8000
```

To stop Docker:
```bash
docker compose down
```
