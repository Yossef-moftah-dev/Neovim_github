# Arabic Sentiment Analysis with AraBERT (ProdML)

[![Tests](https://img.shields.io/badge/Tests-37%20Passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-88%25-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](pyproject.toml)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](docker/Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-ready Arabic sentiment classification API and pipeline powered by **AraBERT**, packaged into a modular Python library (`prodml`), fully covered with unit and integration tests, and containerized for low-latency production deployment.

---

## 📌 Business Context & Motivation

In high-volume e-commerce platforms (such as Noon and Amazon.eg), customer feedback exceeds **50,000 Arabic reviews daily**. Automated sentiment classification enables:
- **Product Rankings & Search Relevance:** Incorporating real-time customer sentiment into discovery algorithms.
- **Merchant & Seller Ratings:** Tracking vendor performance and identifying systematic product defects.
- **Customer Support Prioritization:** Instantly triaging negative feedback to customer success teams.

### Challenges in Arabic NLP
Arabic text presents distinct linguistic complexities that break naive classification pipelines:
- **Morphological Richness & Dialects:** Modern Standard Arabic (MSA) mixed with regional dialects (Egyptian, Gulf, Levantine).
- **Orthographic Inconsistencies:** Multiple forms of Alif (`أ`, `إ`, `آ`, `ا`), Taa Marbuta vs Haa (`ة` / `ه`), and Yaa vs Alif Maqsura (`ي` / `ى`).
- **Diacritics (Tashkeel) & Elongation (Tatweel):** Repetitive character elongations (`راااائع`) and vowel marks that affect tokenization if not normalized.

This project tackles these challenges using domain-specific Arabic normalization, fine-tuned **AraBERTv02** transformer encoders, and high-performance ONNX Runtime inference.

---

## 🏗️ Architecture & Features

```
                                    +-----------------------+
                                    |   Client Requests     |
                                    +-----------+-----------+
                                                |
                                                v
                              +-----------------------------------+
                              |   FastAPI Application Gateway     |
                              |  - Correlation ID Middleware      |
                              |  - Structured JSON Logging        |
                              +-----------------+-----------------+
                                                |
                       +------------------------+------------------------+
                       |                                                 |
                       v                                                 v
           +-----------------------+                         +-----------------------+
           |  POST /predict (Single)|                         | POST /predict/batch   |
           +-----------+-----------+                         +-----------+-----------+
                       |                                                 |
                       +------------------------+------------------------+
                                                |
                                                v
                              +-----------------------------------+
                              |       Arabic Preprocessor         |
                              |  - Diacritic & Tatweel Removal    |
                              |  - Alif/Yaa Orthographic Mapping  |
                              |  - Emoji & Punctuation Clean      |
                              +-----------------+-----------------+
                                                |
                                                v
                              +-----------------------------------+
                              |      SentimentPredictor           |
                              |  - Fine-tuned AraBERTv02          |
                              |  - PyTorch / ONNX Runtime (<1e-4) |
                              |  - Sub-50ms CPU Latency           |
                              +-----------------+-----------------+
                                                |
                                                v
                              +-----------------------------------+
                              | Output: Label, Probabilities & ms |
                              +-----------------------------------+
```

- **Modular Package Layout (`prodml`):** Decoupled into `data`, `features`, `predict`, `train`, `logging`, and `api`.
- **Dual Inference Engine:** Native PyTorch + ONNX Runtime export validated via automated logit parity tests ($\text{tol} < 10^{-4}$).
- **Production API:** FastAPI service with lifespan model loading, Pydantic v2 schemas, correlation IDs, and structured JSON logs.
- **Hardened Containerization:** Multi-stage Dockerfile running non-root `appuser:10001`, volume-mounted model artifacts, and health checks.
- **Automated Artifact Ingestion:** Docker init container and CLI tools to fetch fine-tuned model weights directly from Google Drive.
- **Tested & Verified:** Comprehensive test suite with 37 tests covering unit logic, integration endpoints, and numerical parity.

---

## ⚡ Quickstart (Run in 3 Commands)

```bash
# 1. Clone the repository
git clone https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert.git && cd arabic-sentiment-arabert

# 2. Build and launch containerized service (auto-downloads model weights)
docker compose up -d --build

# 3. Send a test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع جدا وأنصح به بشدة"}'
```

**Sample Response:**
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

## 📡 API Reference

Interactive Swagger documentation is available at **`http://localhost:8000/docs`**.

### 1. Single Prediction (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "الخدمة ممتازة والتوصيل سريع للغاية"}'
```

### 2. Batch Prediction (`POST /predict/batch`)
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["المنتج رائع جدا", "تجربة سيئة ولن أكررها", "عادي لا بأس به"]}'
```

### 3. Health Check (`GET /health`)
Verifies service liveness and confirms model weights are resident in memory:
```bash
curl -i http://localhost:8000/health
```

### 4. Model Metadata (`GET /metadata`)
Returns runtime environment details, class labels, and SHA-256 weight checksum:
```bash
curl -s http://localhost:8000/metadata
```

---

## 🧪 Testing & Code Quality

### 1. Environment Setup
```bash
# Using uv (fastest)
uv sync

# Or using standard pip inside a virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Run Test Suite
Runs all 37 unit, integration, and logit parity tests:
```bash
uv run pytest --cov=src/prodml --cov-report=term-missing
```

### 3. Run Specific Tests
```bash
# Test ONNX vs PyTorch logit parity (< 1e-4 tolerance)
uv run pytest tests/test_logit_parity.py -v

# Test FastAPI endpoints and middleware
uv run pytest tests/test_api.py -v
```

### 4. Linting and Formatting
```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

---

## 📥 Model Artifacts & Data

- **Pretrained Model:** [aubmindlab/bert-base-arabertv02](https://huggingface.co/aubmindlab/bert-base-arabertv02)
- **Dataset:** [Ruqiya/Arabic_Reviews_of_SHEIN](https://huggingface.co/datasets/Ruqiya/Arabic_Reviews_of_SHEIN) (sample tracked with Git LFS in `data/`)
- **Fine-tuned Weights:** Automatically pulled on first startup via Docker Compose from [Google Drive](https://drive.google.com/drive/folders/1yeJcg-nrci3BDq01p9-pjtciv1c4g9rc).

To download weights manually:
```bash
./scripts/download_model.sh
# Or via package CLI:
uv run prodml-download
```

---

## 💻 Local Development & Deployment

### Run Service Directly
```bash
uv run uvicorn prodml.api:app --reload --port 8000
```

### Stop Docker Containers
```bash
docker compose down
```

### Cloud Deployment (AWS EC2)
An automated instance bootstrapping script is included for Ubuntu-based EC2 instances:
```bash
# Configures 4GB swap space, installs Docker, uv, and sets up dependencies
./EC2_instance_setup.sh
```

---

## 📂 Repository Layout

```
arabic-sentiment-arabert/
├── .dockerignore              # Docker build context exclusions
├── .gitattributes             # Git LFS & notebook filter configuration
├── .gitignore                 # Python, venv, and artifact ignore rules
├── docker-compose.yml         # Container orchestration with model download init
├── EC2_instance_setup.sh      # AWS EC2 host setup script
├── LICENSE                    # MIT License
├── pyproject.toml             # Hatchling package build configuration & dependencies
├── README.md                  # Project documentation & quickstart
├── requirements-api.txt       # Frozen requirements for container builds
├── uv.lock                    # Dependency lockfile
├── data/                      # Dataset documentation and LFS samples
├── docker/
│   └── Dockerfile             # Multi-stage non-root runtime container
├── notebooks/                 # Training and experimentation Colab notebooks
├── scripts/
│   └── download_model.sh      # Model weights retrieval script
├── src/prodml/                # Core Python package
│   ├── __init__.py
│   ├── api.py                 # FastAPI endpoints and lifespan handler
│   ├── config.py              # Configuration models
│   ├── data.py                # Dataset ingestion and label parsing
│   ├── download.py            # Weights downloader utility
│   ├── features.py            # Arabic text cleaning and normalization
│   ├── logging.py             # Structured JSON logger
│   ├── middleware.py          # Correlation ID tracing middleware
│   ├── predict.py             # OOP Predictor supporting PyTorch & ONNX
│   ├── timing.py              # Execution latency profiling decorators
│   └── train.py               # Fine-tuning loop and artifact hashing
└── tests/                     # Test suite (37 unit & integration tests)
    ├── conftest.py            # Pytest fixtures and mock models
    ├── test_api.py            # Endpoint testing
    ├── test_config.py         # Config validation
    ├── test_data.py           # Ingestion tests
    ├── test_download.py       # Weights download tests
    ├── test_features.py       # Arabic normalization tests
    ├── test_logging.py        # Logging tests
    ├── test_logit_parity.py   # PyTorch vs ONNX numerical parity
    ├── test_predict.py        # Inference tests
    ├── test_timing.py         # Latency profiling tests
    └── test_train.py          # Training and hashing tests
```

---

## 📄 License

Released under the [MIT License](LICENSE). Pretrained weights and dataset belong to their respective authors.
