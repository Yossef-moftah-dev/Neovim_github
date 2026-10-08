# ProdML — Arabic Sentiment Analysis & MLOps Platform

[![CI](https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert/actions/workflows/ci.yml/badge.svg)](https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/Release-v0.2.0-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/Tests-38%20Passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-73%25-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade Arabic sentiment classification service and end-to-end MLOps platform powered by AraBERT. Features an enterprise tracking topology (PostgreSQL 16, MinIO S3, MLflow), reproducible DVC data pipelines, automated CI/CD quality gates, zero-downtime model registry hot-reloading, and containerized deployment.

---

## ⚡ Quickstart

```bash
# 1. Clone repository
git clone https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert.git && cd arabic-sentiment-arabert

# 2. Launch full tracking and serving stack
docker compose up -d --build

# 3. Test real-time inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع جدا وأنصح به بشدة"}'
```

---

## 🌐 Platform Architecture & Services

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                      │
│      (FastAPI Serving / DVC Pipeline / Experiment Tracking)     │
└──────────────┬──────────────────────────────────┬───────────────┘
               │ HTTP (:5000)                     │ S3 API (:9000)
               ▼                                  ▼
┌──────────────────────────────┐    ┌────────────────────────────┐
│    MLflow Tracking Server    │    │        MinIO Object        │
│    (prodml-mlflow-server)    │    │           Storage          │
│                              │    │       (prodml-minio)       │
└──────────────┬───────────────┘    └─────────────┬──────────────┘
               │ SQL (:5432)                      │ S3 Buckets:
               ▼                                  │  • s3://mlflow/
┌──────────────────────────────┐                  │  • s3://dvc-storage/
│        PostgreSQL 16         │                  │
│      (prodml-postgres)       │◄─────────────────┘
└──────────────────────────────┘
```

| Service | Port / URL | Credentials | Purpose |
| :--- | :--- | :--- | :--- |
| **FastAPI Inference** | [`http://localhost:8000`](http://localhost:8000) | — | Real-time sentiment prediction and Swagger UI at `/docs` |
| **MLflow Tracking** | [`http://localhost:5000`](http://localhost:5000) | — | Experiment tracking, metric curves, and Model Registry |
| **MinIO Console** | [`http://localhost:9001`](http://localhost:9001) | `minioadmin` / `minioadmin` | Object storage browser and administrative management |
| **MinIO S3 API** | `http://localhost:9000` | `minioadmin` / `minioadmin` | S3-compatible remote storage for DVC and MLflow artifacts |
| **PostgreSQL 16** | `localhost:5432` | `mlflow` / `mlflow_password` | Relational backend store for experiment metadata |

---

## ☁️ EC2 Deployment & Automation Script

The repository includes [`EC2_instance_setup.sh`](EC2_instance_setup.sh), a single-command provisioning script for Ubuntu, Debian, Amazon Linux, and RHEL instances:

### What the Script Automates
1. **Docker Engine & Compose:** Installs Docker CE and the Compose plugin via official package repositories.
2. **2GB Swap Allocation:** Configures swap space to prevent Out-Of-Memory (OOM) failures during Docker builds on burstable instances (`t2.micro`, `t3.small`).
3. **Environment Tooling:** Installs Astral `uv` for high-speed Python package execution.
4. **Stack Launch:** Clones or updates the repo and starts all services in detached mode (`docker compose up -d --build`).
5. **Network Output:** Automatically detects the EC2 public IP and outputs direct browser links.

### Running the Setup Script on EC2
```bash
# Option A: Run directly on EC2
curl -fsSL https://raw.githubusercontent.com/Yossef-moftah-dev/arabic-sentiment-arabert/main/EC2_instance_setup.sh | bash

# Option B: Run locally via SSH
ssh -i ~/Downloads/kk.pem ubuntu@<EC2_PUBLIC_DNS> "bash -s" < EC2_instance_setup.sh
```

### Accessing Remote Services
- **Direct Public Access:** Open `http://<EC2_PUBLIC_DNS>:8000/docs`, `:5000`, or `:9001` in your browser (requires AWS Security Group Inbound TCP rules for ports `8000`, `5000`, `9001`, and `9000`).
- **Encrypted SSH Tunnel:** If ports are restricted to SSH (Port 22), run the tunnel helper locally:
  ```bash
  ./scripts/tunnel_services.sh [path/to/kk.pem] [EC2_PUBLIC_DNS]
  ```
  *(Forwards remote ports to `localhost:8000`, `localhost:5000`, `localhost:9001`, and `localhost:9000`).*

### Populating MLflow Experiments on EC2
To populate the 5 model exploration runs and register the Production model inside the EC2 MLflow database:
```bash
ssh -i ~/Downloads/kk.pem ubuntu@<EC2_PUBLIC_DNS> \
  "cd ~/arabic-sentiment-arabert && uv run python scripts/run_experiments.py"
```

---

## 📡 API Usage & Endpoints

Interactive OpenAPI Swagger docs are available at `http://localhost:8000/docs`.

### 1. Single & Batch Predictions
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "الخدمة ممتازة والتوصيل سريع جدا"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["المنتج رائع جدا", "تجربة سيئة ولن أكررها", "عادي لا بأس به"]}'
```

### 2. Code-Free Model Hot-Reload (`POST /model/reload`)
Reloads weights from the MLflow Model Registry into memory with **zero downtime** and without redeploying containers:
```bash
curl -X POST "http://localhost:8000/model/reload?model_uri=models:/arabic-sentiment-model/Production"
```

### 3. Health & Lineage Metadata (`GET /health`, `GET /metadata`)
```bash
# Memory residency health check
curl -s http://localhost:8000/health

# SHA-256 weight hash, active registry version, and class mappings
curl -s http://localhost:8000/metadata
```

---

## 🔬 MLflow Tracking & Model Registry

Systematic model exploration across 5 runs spanning 3 distinct model architectures (`AraBERT`, `CAMeLBERT`, `MARBERT`):

```bash
# Run experiment matrix and log to MLflow
uv run python scripts/run_experiments.py

# Demonstrate zero-downtime hot-swap between model versions
uv run python scripts/demonstrate_model_swap.py
```

| Run Name | Model Family | Chunk / Strategy | Epochs | Val Loss | Val Acc | Macro F1 | Registry Lifecycle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `run-1-arabert-chunk64` | AraBERT | 64 / Head Truncate | 3 | 0.395 | 84.2% | 0.825 | — |
| `run-2-arabert-chunk128` | AraBERT | 128 / Sliding Window | 4 | 0.334 | 87.6% | 0.862 | Version 1 (Archived) |
| `run-3-camelbert-chunk128` | CAMeLBERT | 128 / Head-Tail | 3 | 0.362 | 85.8% | 0.841 | — |
| `run-4-marbert-chunk64` | MARBERT | 64 / Head Truncate | 3 | 0.378 | 84.9% | 0.834 | — |
| `run-5-arabert-champion-chunk128` | AraBERT | 128 / Window Overlap | 5 | **0.281** | **89.8%** | **0.887** | **Version 2 (Production)** |

Comparative UI screenshot captured in [`reports/mlflow_comparison.png`](reports/mlflow_comparison.png).

---

## 🔄 DVC Data Versioning

Pipelines and dataset splits are hashed and cached via DVC backed by MinIO S3 storage:

```bash
# Execute and cache reproducible stages (prepare_data -> evaluate)
uv run dvc repro

# Push/pull dataset artifacts to/from MinIO remote
uv run dvc push
uv run dvc pull
```

---

## 🛡️ CI/CD Quality Gates & Retraining

### 1. Model Quality Gate (`scripts/model_quality_gate.py`)
Blocks PR merges if candidate metrics regress below operational thresholds (Accuracy $\ge 0.70$, Macro F1 $\ge 0.65$, Latency $\le 100\text{ ms}$):
```bash
# Nominal evaluation (PASSED / Exit 0)
uv run python scripts/model_quality_gate.py --metrics reports/eval_metrics.json

# Simulated regression check (BLOCKS / Exit 1)
uv run python scripts/model_quality_gate.py --simulate-regression
```

### 2. GitHub Actions Workflows
- **CI Pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):** Runs `ruff`, `pytest` with coverage enforcement ($\ge 70\%$), model quality gate, and multi-stage Docker build/push.
- **Continuous Retraining ([`.github/workflows/continuous-training.yml`](.github/workflows/continuous-training.yml)):** Supports cron schedule, manual workflow dispatch, and repository webhooks, with human approval required for `Production` deployment.

---

## 🧪 Testing & Code Quality

```bash
# Install dependencies
uv sync --all-extras --dev

# Run full test suite with coverage
uv run pytest --cov=src/prodml --cov-report=term-missing --cov-fail-under=70

# ONNX vs PyTorch logit parity check (< 1e-4 tolerance)
uv run pytest tests/test_logit_parity.py -v

# Code linting & formatting checks
uv run ruff check src/ tests/ scripts/
uv run ruff format --check src/ tests/ scripts/
```

---

## 📄 Documentation & Reports

- **Module 1 Report:** [`reports/module-1.md`](reports/module-1.md) (Foundation, Packaging & Service)
- **Module 2 Report:** [`reports/module-2.md`](reports/module-2.md) (Tracking, Versioning & Quality Gates)
- **License:** [MIT License](LICENSE)
