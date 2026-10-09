# ProdML — Arabic Sentiment Analysis & MLOps Platform

[![CI](https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert/actions/workflows/ci.yml/badge.svg)](https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/Release-v0.4.0-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/Tests-68%20Passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-71%25-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Production-grade Arabic sentiment classification service and end-to-end MLOps platform powered by AraBERT. Features Apache Airflow 2.9+ pipeline orchestration, high-throughput BentoML dynamic micro-batching, NVIDIA Triton & ONNX Runtime accelerated serving, Nginx canary routing with automated sub-second rollback, reproducible DVC pipelines, full MLflow experiment governance, and complete production observability (Prometheus, Grafana as code, Alertmanager, Evidently drift monitoring, and closed-loop retraining).

---

## ⚡ Quickstart

```bash
# 1. Clone repository
git clone https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert.git && cd arabic-sentiment-arabert

# 2. Launch full platform stack (Airflow, BentoML, MLflow, Postgres, MinIO, Prometheus, Grafana, Alertmanager)
docker compose up -d --build

# 3. Test real-time inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "المنتج رائع جدا وأنصح به بشدة"}'
```

---

## 🌐 Platform Architecture & Services

| Service | Port / URL | Credentials | Purpose |
| :--- | :--- | :--- | :--- |
| **BentoML / FastAPI** | [`http://localhost:8000`](http://localhost:8000) | — | Production serving with dynamic micro-batching (`max_batch=64`) |
| **Grafana Observability** | [`http://localhost:3000`](http://localhost:3000) | `admin` / `admin` | Dashboards as code (Health/Ingress, Stage Latency, Data/Drift, System) |
| **Prometheus Telemetry** | [`http://localhost:9090`](http://localhost:9090) | — | TSDB scraping `/metrics`, SLA evaluation, and quantile histograms |
| **Alertmanager Routing** | [`http://localhost:9093`](http://localhost:9093) | — | Alert routing, grouping, and immediate on-call action runbooks |
| **Canary Reverse Proxy** | [`http://localhost:80`](http://localhost:80) | — | Nginx weighted traffic splitting (95% Prod / 5% Canary) |
| **Apache Airflow UI** | [`http://localhost:8080`](http://localhost:8080) | `admin` / `admin` | DAG orchestration, automated training & quality gate branching |
| **MLflow Tracking** | [`http://localhost:5000`](http://localhost:5000) | — | Experiment tracking, metric curves, and Model Registry |
| **MinIO Console** | [`http://localhost:9001`](http://localhost:9001) | `minioadmin` / `minioadmin` | Object storage browser and administrative management |
| **MinIO S3 API** | `http://localhost:9000` | `minioadmin` / `minioadmin` | S3-compatible remote storage for DVC and MLflow artifacts |
| **PostgreSQL 16** | `localhost:5432` | `mlflow` / `mlflow_password` | Backend store for MLflow (`mlflow`), Airflow (`airflow`), & Drift (`monitoring_drift_records`) |


---

## 📦 GitHub Container Registry (GHCR) & Deployment

The serving service is automatically verified, built, and published to **GitHub Container Registry (GHCR)** on every release and commit:

### 1. Pull Pre-Built Image
```bash
# Pull by specific Git commit SHA (deterministic reproducible artifact)
docker pull ghcr.io/yossef-moftah-dev/arabic-sentiment-arabert:<commit_sha>

# Pull latest stable release
docker pull ghcr.io/yossef-moftah-dev/arabic-sentiment-arabert:latest
```

### 2. Standalone Container Execution
```bash
docker run -d --name prodml-service \
  -p 8000:8000 \
  -v $(pwd)/outputs/final_model:/app/outputs/final_model:ro \
  ghcr.io/yossef-moftah-dev/arabic-sentiment-arabert:latest
```

### 3. Zero-Downtime Deployment & Automated Rollback (`scripts/deploy.sh`)
The repository includes [`scripts/deploy.sh`](scripts/deploy.sh), an idempotent deployment script featuring **automated healthcheck validation and sub-second rollback**:
```bash
# Deploy a specific Git commit SHA or release tag
./scripts/deploy.sh --tag 28d5d6e13e819e1f68e23ef39e0f516606036ec2

# Deploy the latest GHCR package
./scripts/deploy.sh --latest

# Immediately rollback to previous active container
./scripts/deploy.sh --rollback
```

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

---

## 📊 Production Observability & Closed Retraining Loop

The platform features an end-to-end production monitoring and closed-loop retraining architecture:

### 1. Real-Time Telemetry & Prometheus Metrics
- **Metrics Exposition:** Exposes live request rates, p50/p95/p99 latency quantiles, sentiment class distributions, prediction confidences, and micro-batch distributions at `GET /metrics`.
- **Multiprocess Concurrency Trap:** Handled via `PROMETHEUS_MULTIPROC_DIR` and `MultiProcessCollector` across Uvicorn/BentoML worker processes.

### 2. Grafana Dashboard as Code (Zero-Click Recovery)
- Declarative 4-row dashboard at `http://localhost:3000` (`admin`/`admin`):
  1. **Row 1: Health & Ingress** (Throughput, 2xx/4xx/5xx status rates, model residency status)
  2. **Row 2: Stage Latency** (p50, p95, p99 latency heatmaps and time-series)
  3. **Row 3: Data & Drift Telemetry** (PSI scores, Wasserstein distance, sentiment class shares)
  4. **Row 4: System Resources** (CPU utilization, resident memory RSS)

### 3. Statistical Drift Simulation & Detection
- **4 Drift Typologies (`monitoring/simulate_drift.py`):** Sudden, Gradual, Incremental, and Periodic/Seasonal drift.
- **5 Detection Algorithms (`monitoring/drift_detector.py`):** Chi-Square ($\chi^2$), Wasserstein Distance ($W_1$), Population Stability Index (PSI), Jensen-Shannon (JS) Divergence, and Maximum Mean Discrepancy (MMD).
- **Evidently AI & PostgreSQL Store (`monitoring/evidently_monitor.py`):** Generates interactive HTML reports (`reports/evidently_drift_report.html`) and persists structured drift records to PostgreSQL table `monitoring_drift_records`.

### 4. Closed Retraining Loop & Storm Protections
- **Defensive Storm Gates (`monitoring/retraining_trigger.py`):** Enforces Dwell Time Cooldown, 24-hour Rate Limiting, and Sample Volume/Quality sanity checks.
- **End-to-End Automated Demonstration:**
  ```bash
  # Execute full automated closed retraining loop
  uv run python scripts/demonstrate_closed_loop.py

  # Deliberately fire and verify 3 production alert scenarios
  uv run python scripts/test_alerts.py
  ```

---

## 🧪 Testing & Code Quality

```bash
# Install dependencies
uv sync --all-extras --dev

# Run full test suite with coverage
uv run pytest --cov=src/prodml --cov=monitoring --cov-report=term-missing --cov-fail-under=70

# ONNX vs PyTorch logit parity check (< 1e-4 tolerance)
uv run pytest tests/test_logit_parity.py -v

# Code linting & formatting checks
uv run ruff check src/ tests/ scripts/ monitoring/
uv run ruff format --check src/ tests/ scripts/ monitoring/
```

---

## 📄 Documentation & Reports

- **Module 1 Report:** [`reports/module-1.md`](reports/module-1.md) (Foundation, Packaging & Service)
- **Module 2 Report:** [`reports/module-2.md`](reports/module-2.md) (Tracking, Versioning & Quality Gates)
- **Module 3 Report:** [`reports/module-3.md`](reports/module-3.md) (Serving, Orchestration & Load Testing)
- **Module 4 Report:** [`reports/module-4.md`](reports/module-4.md) (Observability, Monitoring & Retraining)
- **On-Call Runbooks:** [`docs/runbooks/`](docs/runbooks/) (Actionable Alert Incident Guides)
- **License:** [MIT License](LICENSE)
