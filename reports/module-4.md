# Milestone 04 Report — Observability, Monitoring & Retraining

> **Reference:** *The MLOps Practitioner Handbook*, Module 5, Pages 41–61  
> **Track:** Track A — Deep Learning: Arabic Sentiment Analysis  
> **Git Milestone Branch:** `module-4-observability`  
> **Release Tag:** `v0.4.0`  
> **Author:** Yossef Moftah  
> **Date:** October 2026  

---

## 1. Executive Summary

Milestone 04 completes the operational evolution of the `prodml` platform, transitioning the system from Level 3 high-throughput micro-batching into a **self-healing, observable Level 4 production ecosystem**. 

By coupling real-time telemetry with rigorous statistical drift analysis and automated closed retraining loops, the platform guarantees that regressions, concept drift, and performance bottlenecks are detected, surfaced, and mitigated in real time:

1. **Prometheus Telemetry & Multiprocess Trap Mitigation:** Instrumented FastAPI and BentoML serving layers with `prometheus-client`, resolving the multiprocess concurrency trap via `PROMETHEUS_MULTIPROC_DIR` and exposing structured request, latency, batching, and model lifecycle metrics at `GET /metrics`.
2. **Grafana Dashboard as Code (Zero-Click Recovery):** Declaratively provisioned a production 4-row dashboard (`docker/grafana/dashboards/prodml_observability.json`) displaying Health/Ingress, Stage Latency, Data/Drift, and System Resources. Verified that complete teardown and re-launch restores all panels, datasource connections, and release annotations with zero manual UI interaction.
3. **Actionable Alertmanager Rules & On-Call Runbooks:** Configured 6 production alert rules with strict thresholds, explicit immediate on-call actions, and direct links to comprehensive triage runbooks (`docs/runbooks/*.md`). Deliberately fired and verified 3 live alert scenarios.
4. **Statistical Drift Detection (5 Methods across 4 Typologies):** Implemented mathematical detectors for Chi-Square ($\chi^2$), Wasserstein Distance ($W_1$), Population Stability Index (PSI), Jensen-Shannon (JS) Divergence, and Maximum Mean Discrepancy (MMD). Authored `monitoring/simulate_drift.py` covering Sudden, Gradual, Incremental, and Periodic/Seasonal drift patterns in Arabic review distributions.
5. **Evidently AI & PostgreSQL Metrics Store:** Orchestrated periodic Evidently batch evaluation pipelines, generating interactive HTML and structured JSON summaries while persisting historical drift scores into the `monitoring_drift_records` PostgreSQL table.
6. **Closed Retraining Loop with 3-Tier Storm Protections:** Engineered `monitoring/retraining_trigger.py` featuring Dwell Time Cooldown, 24-hour Rate Limiting, and Data Quality/Volume sanity checks. Successfully verified the full automated chain: *Drift Injection → Statistical Breach → Alertmanager Fired → Storm Gates Evaluated → Airflow DAG → Model Quality Gate → MLflow Staging Registry*.

---

## 2. Platform Observability Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Live Client Ingress                                    │
│             (Single reviews, Batch payloads, or Drift Simulator traffic)              │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ HTTP / JSON
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                ProdML Serving Tier                                     │
│  FastAPI / BentoML Service (Port 8000)                                                 │
│   ├── Middleware: Correlation ID, Request Timer & Status Tracking                      │
│   ├── Predictor: AraBERT ONNX/PyTorch Sentiment Classifier                             │
│   ├── Prometheus Metrics: prodml_http_requests_total, latency_seconds, drift_score     │
│   └── Multiprocess Collector: PROMETHEUS_MULTIPROC_DIR (/tmp/prometheus_multiproc)    │
│   └── Endpoint: GET /metrics (Prometheus Text Exposition)                              │
└────────────────────┬──────────────────────────────────────┬────────────────────────────┘
                     │ Scrapes :8000/metrics (5s)           │ Logs Inference Payloads
                     ▼                                      ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────────────┐
│       Prometheus (Port 9090)         │  │     Evidently AI & Drift Engine              │
│  • TSDB Metric Scrapes (5s)          │  │  • monitoring/simulate_drift.py (4 Typologies│
│  • PromQL Quantile Evaluations       │  │  • monitoring/drift_detector.py (5 Stats)    │
│  • alert_rules.yml (6 Rules)         │  │  • monitoring/evidently_monitor.py           │
└──────────────────┬───────────────────┘  └───────────────────────┬──────────────────────┘
                   │ Alert Dispatch                               │ Persists Drift Metrics
                   ▼                                              ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────────────────┐
│     Alertmanager (Port 9093)         │  │         PostgreSQL 16 (Port 5432)            │
│  • Routing & Severity Grouping       │  │  • Table: monitoring_drift_records           │
│  • Immediate Actions & Runbook URLs  │  │  • Historical p-values, PSI, test status     │
└──────────────────┬───────────────────┘  └──────────────────────────────────────────────┘
                   │ Webhook Trigger / Critical Breach
                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    Closed Retraining Loop & Storm Protection                           │
│  monitoring/retraining_trigger.py                                                      │
│   ├── Gate 1: Cooldown Dwell Time Gate (Reject if retrained < 60m ago)                │
│   ├── Gate 2: Rate Limiting Gate (Max 5 triggers per 24-hour rolling window)          │
│   └── Gate 3: Data Quality & Volume Gate (Min 50 non-null validated samples)           │
│                                           │ Dispatches Trigger                         │
│                                           ▼                                            │
│  Apache Airflow 2.9+ / Local Pipeline: arabic_sentiment_training_pipeline              │
│   ├── sensor_upstream_data >> extract_data >> validate_data >> train_model             │
│   ├── evaluate_model >> branch_quality_gate >> [register_model | model_rejected]       │
│   └── Promotes Verified Candidate to MLflow Model Registry Staging                     │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               Grafana Dashboard as Code                                │
│  Grafana (Port 3000) — Auto-Provisioned Dashboards & Datasources                       │
│   ├── Row 1: Health & Ingress (RPS, 2xx/4xx/5xx Rates, Model Residency, Uptime)        │
│   ├── Row 2: Stage Latency (p50, p95, p99 Latency Heatmaps, Batch Distributions)       │
│   ├── Row 3: Data & Drift Telemetry (PSI, Wasserstein Curves, Class Shifts)            │
│   └── Row 4: System Resources (Process CPU, Resident Memory, Worker Concurrency)       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Formal Metric Contract & Prometheus Telemetry

### 3.1 Formal Metric Contract Specification

The metrics contract strictly defines all telemetry exported by the serving layer:

| Metric Name | Type | Labels / Dimensions | Bucket Boundaries / Units | Operational Purpose |
| :--- | :---: | :--- | :--- | :--- |
| `prodml_http_requests_total` | Counter | `method`, `endpoint`, `status` | Count (integer) | Ingress throughput monitoring, 4xx client errors, and 5xx server error rate tracking. |
| `prodml_http_request_duration_seconds` | Histogram | `method`, `endpoint` | `[0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5, 5.0]` | End-to-end request latency percentiles (p50, p95, p99) against operational SLAs. |
| `prodml_predictions_total` | Counter | `model_version`, `predicted_label` | Count (integer) | Volume of classified sentiment samples per class label (`Negative`, `Neutral`, `Positive`). |
| `prodml_prediction_confidence` | Histogram | `model_version`, `predicted_label` | `[0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.0]` | Detection of model calibration decay or prediction uncertainty shifts. |
| `prodml_batch_size` | Histogram | None | `[1, 2, 4, 8, 16, 32, 64]` | Saturation and window efficiency tracking for dynamic micro-batching. |
| `prodml_drift_score` | Gauge | `metric`, `feature` | Dimensionless float | Real-time statistical drift score (PSI, Wasserstein, $\chi^2$, JS divergence). |
| `prodml_drift_detected` | Gauge | `metric`, `feature` | Binary (`0` or `1`) | Alerting trigger flag for statistical SLA threshold breach. |
| `prodml_model_loaded` | Gauge | `model_version` | Binary (`0` or `1`) | Continuous verification of model weight residency in memory. |

### 3.2 The Multiprocess Concurrency Trap & Resolution

In enterprise Python ML serving architectures (such as Uvicorn running multiple `--workers` or BentoML process runners), standard in-memory Prometheus metric registries encounter a severe architectural flaw:
- Each OS process maintains an isolated in-memory metric state.
- Successive scrapes hitting different worker processes return erratic, non-monotonic counter jumps and missing observations.
- Gauge values oscillate violently depending on which worker handles the `/metrics` scrape request.

**The Solution:**  
1. Configured `PROMETHEUS_MULTIPROC_DIR` environment variable pointing to a shared memory-mapped directory (`/tmp/prometheus_multiproc`).
2. At container startup, the supervisor cleans stale `.db` files from crashed workers (`cleanup_multiproc_dir()`).
3. Registered `prometheus_client.multiprocess.MultiProcessCollector(registry)` to aggregate and merge binary shared-memory dictionaries across all active Uvicorn worker PIDs into an accurate consolidated exposition document during each scrape.

---

## 4. PromQL Query Catalog & Telemetry Mathematics

Production PromQL expressions powering dashboards and Alertmanager rules:

### 4.1 p95 Tail Latency SLA

$$
q_{0.95} = \operatorname{histogram\_quantile}\left(0.95, \sum \operatorname{rate}(H_{\text{bucket}}[2\text{m}])\right)
$$

```promql
histogram_quantile(0.95, sum(rate(prodml_http_request_duration_seconds_bucket{endpoint=~"/predict.*"}[2m])) by (le))
```
*Evaluates whether the 95th percentile inference latency remains strictly under the 100ms operational budget.*

### 4.2 HTTP 5xx Error Ratio

$$
\text{Error Rate} = \frac{\sum \operatorname{rate}(\text{HTTP}_{5\text{xx}}[2\text{m}])}{\sum \operatorname{rate}(\text{HTTP}_{\text{total}}[2\text{m}])}
$$
```promql
sum(rate(prodml_http_requests_total{status=~"5.."}[2m])) / clamp_min(sum(rate(prodml_http_requests_total[2m])), 0.001)
```
*Triggers critical incident severity if server errors exceed 2% of total traffic.*

### 4.3 Predicted Class Distribution Shifts
```promql
sum(rate(prodml_predictions_total[5m])) by (predicted_label) / ignoring(predicted_label) group_left sum(rate(prodml_predictions_total[5m]))
```
*Calculates the real-time proportion of Positive, Neutral, and Negative predictions over a 5-minute rolling window to detect prior probability shifts.*

### 4.4 Dynamic Micro-Batch Saturation
```promql
histogram_quantile(0.90, sum(rate(prodml_batch_size_bucket[5m])) by (le))
```
*Identifies whether incoming requests consistently fill the micro-batch window to the maximum configured ceiling (`max_batch_size=64`).*

---

## 5. Grafana Dashboard as Code & Zero-Click Recovery

### 5.1 Declarative Provisioning Architecture

All Grafana infrastructure is managed strictly as code:
- **Datasource (`docker/grafana/provisioning/datasources/prometheus.yml`):** Connects to `http://prometheus:9090` with automated 5s intervals.
- **Provider (`docker/grafana/provisioning/dashboards/dashboards.yml`):** Directs Grafana to scan `/etc/grafana/dashboards` on startup.
- **Dashboard (`docker/grafana/dashboards/prodml_observability.json`):** 4-row dashboard layout (`uid: prodml-observability-v1`).

### 5.2 The 4-Row Dashboard Layout

| Row | Panel Name | Visualization Type | Metric / PromQL Query |
| :---: | :--- | :---: | :--- |
| **Row 1: Health & Ingress** | Total Request Rate (RPS) | Time Series | `sum(rate(prodml_http_requests_total[1m]))` |
| | HTTP Status Breakdown | Time Series | `sum(rate(prodml_http_requests_total{status=~"[245].."}[1m]))` |
| | Service & Model Residency | Stat Gauge | `prodml_model_loaded` |
| | Total Lifetime Inferences | Stat Counter | `sum(prodml_predictions_total)` |
| **Row 2: Stage Latency** | Latency Percentiles (p50 / p95 / p99) | Time Series | `histogram_quantile([0.5, 0.95, 0.99], ...)` |
| | Micro-Batch Size Distribution | Time Series | `sum(rate(prodml_batch_size_bucket[2m])) by (le)` |
| **Row 3: Data & Drift** | Population Stability Index (PSI) | Time Series | `prodml_drift_score{metric="psi"}` |
| | Wasserstein Distance (EMD) | Time Series | `prodml_drift_score{metric="wasserstein"}` |
| | Class Prediction Share (%) | Time Series Stacked | Normalized class rates |
| **Row 4: System Resources** | Process CPU Utilization | Time Series | `rate(process_cpu_seconds_total[1m])` |
| | Memory Residency (RSS) | Time Series | `process_resident_memory_bytes` |

### 5.3 Zero-Click Recovery Proof

To verify that the monitoring stack adheres to zero-click disaster recovery:
1. Grafana container and persistent volume were destroyed:
   ```bash
   docker compose down -v
   ```
2. The platform stack was relaunched from cold storage:
   ```bash
   docker compose up -d
   ```
3. Navigating to `http://localhost:3000` immediately presented the complete dashboard with all 11 panels, Prometheus data links, and template variables (`$model_version`, `$instance`) without any UI imports or manual clicks.

---

## 6. Actionable Alerting Rules & Alertmanager Routing

### 6.1 Actionable Alerting Matrix

Alert rules are defined in `docker/prometheus/alert_rules.yml`. In accordance with production best practices, **every alert contains an explicit immediate first action and a link to a dedicated on-call runbook**:

| Alert Name | Severity | Breach Condition | For | Immediate First Action | Runbook Link |
| :--- | :---: | :--- | :---: | :--- | :--- |
| `HighInferenceLatencyP95` | Critical | p95 latency > 100 ms | 1m | Execute `serving/canary_promote.sh rollback` if canary is live; check CPU quota. | [`docs/runbooks/high_latency.md`](../docs/runbooks/high_latency.md) |
| `HighHTTP5xxErrorRate` | Critical | 5xx error rate > 2% | 1m | Inspect `docker compose logs -n 100 prodml-service`; restart service if needed. | [`docs/runbooks/http_5xx_errors.md`](../docs/runbooks/http_5xx_errors.md) |
| `CriticalPredictionDrift` | Warning | PSI > 0.20 or W₁ > 0.25 | 1m | Review `reports/evidently_drift_report.html`; inspect closed-loop retraining status. | [`docs/runbooks/prediction_drift.md`](../docs/runbooks/prediction_drift.md) |
| `ModelUnloadedOrServiceDegraded` | Critical | `prodml_model_loaded == 0` or target down | 30s | Trigger dynamic model reload via `POST /model/reload`; verify `/health` probe. | [`docs/runbooks/service_unloaded.md`](../docs/runbooks/service_unloaded.md) |
| `LowPredictionConfidenceAnomaly` | Warning | Mean confidence < 65% | 2m | Sample review logs for Franco-Arabic, emoji saturation, or out-of-domain slang. | [`docs/runbooks/confidence_anomaly.md`](../docs/runbooks/confidence_anomaly.md) |
| `BatchQueueSaturation` | Warning | 90th percentile batch size ≥ 60 | 2m | Scale worker replica count or adjust `max_batch_size` in `serving/service.py`. | [`docs/runbooks/batch_saturation.md`](../docs/runbooks/batch_saturation.md) |

### 6.2 Deliberate Alert Testing Verification

Authored `scripts/test_alerts.py` to deliberately fire 3 production alerts into the Alertmanager v2 API (`http://localhost:9093/api/v2/alerts`):
1. **CriticalPredictionDrift:** Verified routing to warning receiver with immediate runbook link.
2. **HighInferenceLatencyP95:** Verified critical alert dispatching with sub-second rollback mitigation instruction.
3. **HighHTTP5xxErrorRate:** Verified critical alert dispatching with log inspection guidance.

---

## 7. Statistical Drift Detection Algorithms & The 4 Drift Typologies

### 7.1 Statistical Detection Algorithms

Authored in `monitoring/drift_detector.py`:

#### 1. Chi-Square Test of Independence ($\chi^2$)
For categorical discrete outcomes (sentiment labels: Negative, Neutral, Positive):

$$
\chi^2 = \sum_{i=1}^k \frac{(O_i - E_i)^2}{E_i}, \quad p = 1 - F_{\chi^2}(\chi^2, k - 1)
$$

*Detects prior probability shifts in class distribution ($p \lt 0.05$).*

#### 2. 1-Wasserstein Distance ($W_1$, Earth Mover's Distance)
For continuous 1D features (prediction confidence, character lengths):

$$
W_1(u, v) = \int_{-\infty}^{\infty} |U(x) - V(x)| \, \mathrm{d}x
$$

*Measures the minimum work required to transform the reference cumulative distribution into the target distribution ($W_1 > 0.25$).*

#### 3. Population Stability Index (PSI)
Binned risk metric comparing baseline reference $R_i$ against target batch $T_i$:

$$
\mathrm{PSI} = \sum_{i=1}^k (T_i - R_i) \cdot \ln\left(\frac{T_i}{R_i}\right)
$$

- $\mathrm{PSI} \lt 0.10$: Stable / No significant drift.
- $0.10 \le \mathrm{PSI} \lt 0.20$: Moderate shift / Monitoring recommended.
- $\mathrm{PSI} \ge 0.20$: Significant drift / Retraining trigger mandatory.

#### 4. Jensen-Shannon Divergence ($\mathrm{JSD}$)
Symmetric, smoothed relative entropy bounded in $[0, 1]$:

$$
\mathrm{JSD}(P \parallel Q) = \frac{1}{2} D_{\mathrm{KL}}(P \parallel M) + \frac{1}{2} D_{\mathrm{KL}}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)
$$

#### 5. Maximum Mean Discrepancy ($\mathrm{MMD}$)
Kernel two-sample test in Reproducing Kernel Hilbert Space (RKHS) using Gaussian RBF kernel $k(x, y) = \exp(-\gamma \|x - y\|^2)$:

$$
\mathrm{MMD}^2(X, Y) = \frac{1}{m^2}\sum_{i=1}^m \sum_{j=1}^m k(x_i, x_j) - \frac{2}{mn}\sum_{i=1}^m \sum_{j=1}^n k(x_i, y_j) + \frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n k(y_i, y_j)
$$

### 7.2 The 4 Production Drift Typologies

Implemented in `monitoring/simulate_drift.py`:

```
1. Sudden Drift:            2. Gradual Drift:
   Distribution                 Distribution
   ▲    ┌──────────             ▲        . - - -
   │    │                       │      /
   │────┘                       │──── '
   └─────────────► Time         └─────────────► Time

3. Incremental Drift:       4. Periodic / Seasonal:
   Distribution                 Distribution
   ▲      ┌──────               ▲    ╭─╮   ╭─╮
   │    ┌─┘                     │    │ │   │ │
   │────┘                       │────╯ ╰───╯ ╰► Time
   └─────────────► Time
```

1. **Sudden Drift (Abrupt Shift):** Abrupt step change at $t_0$, simulating viral boycott events or slang shock where negative reviews jump from 15% to 85% of traffic.
2. **Gradual Drift:** Continuous sigmoidal transition $P(\mathrm{drift} \mid t) = \frac{1}{1 + e^{-k(t - t_{\mathrm{mid}})}}$, simulating smooth language evolution across seasons.
3. **Incremental Drift:** Discrete multi-stage degradation across 4 distinct phases (10% → 30% → 60% → 90%).
4. **Periodic / Seasonal Drift:** Sinusoidal oscillation $P(t) = 0.5 + 0.45 \sin(2\pi t / T)$, simulating holiday/Ramadan shopping surges with high positive sentiment and promotional language.

---

## 8. Evidently AI & PostgreSQL Metrics Store Integration

### 8.1 Batch Evaluation Pipeline

`monitoring/evidently_monitor.py` extracts lightweight Arabic NLP features:
- Character length (`char_length`)
- Word count (`word_count`)
- Arabic character density (`arabic_ratio`)
- Class sentiment distribution (`label`)

Executes the statistical drift suite against reference splits, generating an interactive HTML dashboard in `reports/evidently_drift_report.html` and serializing results to `reports/drift_summary.json`.

### 8.2 PostgreSQL Schema & Persistence

Every evaluation batch writes structured metrics to PostgreSQL (`prodml-postgres:5432`) in the `monitoring_drift_records` table:

```sql
CREATE TABLE IF NOT EXISTS monitoring_drift_records (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    batch_id VARCHAR(100),
    feature_name VARCHAR(100),
    metric_type VARCHAR(50),
    statistic_value DOUBLE PRECISION,
    threshold_value DOUBLE PRECISION,
    p_value DOUBLE PRECISION,
    drift_detected BOOLEAN,
    reference_sample_count INT,
    current_sample_count INT,
    details JSONB
);
```

Sample query for active drift status:
```sql
SELECT timestamp, feature_name, metric_type, statistic_value, threshold_value, drift_detected
FROM monitoring_drift_records
WHERE drift_detected = true
ORDER BY timestamp DESC
LIMIT 5;
```

---

## 9. Closed Retraining Loop Demonstration & Storm Protections

### 9.1 Retraining Storm Protections

Frequent or repeated drift alerts in production can trigger "retraining storms" — runaway training cascades that exhaust GPU/CPU compute, degrade database pools, and cause thrashing in Model Registry versions.

`monitoring/retraining_trigger.py` enforces **3 mandatory defensive gates**:
1. **Dwell Time / Cooldown Window Gate:** Rejects any retraining trigger if the previous run executed less than 60 minutes ago (`min_cooldown_seconds=3600`).
2. **Rate Limiting Gate:** Restricts triggers to a maximum of 5 runs within any rolling 24-hour window (`max_triggers_per_day=5`).
3. **Data Quality & Volume Gate:** Blocks retraining unless the candidate drift dataset contains at least 50 records (`min_samples_threshold=50`), zero null text entries, and at least 2 distinct class labels.

### 9.2 End-to-End Evidence Chain Verification

The complete automated demonstration was executed via `scripts/demonstrate_closed_loop.py` and logged to `reports/closed_loop_execution.log`:

```
============================================================
CLOSED RETRAINING LOOP EXECUTION SUMMARY:
============================================================
Step 1: Baseline Reference Verification -> PASSED
  Verified baseline dataset at data/processed/train.csv with 1932 samples and class balance.
Step 2: Drift Injection (Typology 1: Sudden Shift) -> COMPLETED
  Synthesized 180 reviews with abrupt negative polarity inversion and dialect slang.
Step 3: Evidently TestSuite & Statistical Evaluation -> FAILED_QUALITY_GATE
  Evaluations complete. Overall drift detected: True. (PSI = 6.98, Chi-Square p = 0.0). Report saved to reports/evidently_drift_report.html.
Step 4: Alertmanager Notification Trigger -> ALERT_DISPATCHED
  Fired CriticalPredictionDrift (severity: warning) with immediate action: 'Trigger closed-loop retraining'.
Step 5: Retraining Storm Protection Gates -> PASSED
  Gate verification outcome: PASSED (Dwell time, rate limiting, and sample volume verified).
Step 6: Retraining Pipeline & Model Quality Gate -> PROMOTED_TO_STAGING
  Pipeline executed: Extract -> Validate -> Train -> Evaluate (Acc=0.73, F1=0.71) -> Quality Gate -> MLflow Model Registry Staging.
============================================================
```

---

## 10. Track A Scoping Rationale

Per project requirements, tooling was applied strictly based on relevance to **Track A (Deep Learning: Arabic Sentiment Analysis)**:

- **Track B Exclusions (GenAI / LLM Tracing):** Self-hosted Langfuse, RAGAS 4-metric evaluation, and Guardrails PII redaction apply strictly to generative retrieval-augmented generation (RAG) and LLM chatbots. Because ProdML is a supervised deep learning sequence classification platform, these tools were omitted to avoid architectural bloat.
- **Track C Exclusions (Tabular Detector Scorecard):** Tabular Step 0 corruption suites apply exclusively to structured columnar feature stores. ProdML instead implements statistical drift detection specifically engineered for natural language tokens, character lengths, and sentiment class distributions.

---

## 11. Verification Checklist & Acceptance Proof

- [x] Service exposes valid Prometheus metrics at `GET /metrics`.
- [x] Multiprocess mode trap handled cleanly via `PROMETHEUS_MULTIPROC_DIR`.
- [x] Grafana dashboard provisions automatically with zero-click recovery.
- [x] Alertmanager configured with ≥ 6 actionable rules, runbooks, and deliberate test triggers.
- [x] Statistical drift detection implemented across all 5 algorithms and 4 typologies.
- [x] Evidently AI and PostgreSQL persistence operational.
- [x] Closed retraining loop with storm protections verified end-to-end.
- [x] Full test suite passes with code coverage ≥ 70%.
- [x] Report authored and committed to branch `module-4-observability`.
