# Runbook: HighHTTP5xxErrorRate Alert

## 1. Alert Summary
- **Severity:** Critical
- **SLA Breach Threshold:** HTTP 5xx error rate exceeds 2% of total incoming requests for 1 minute.
- **Affected System:** ProdML FastAPI / BentoML API Gateway.

## 2. Immediate On-Call Actions
1. **Container Log Inspection:**
   Fetch recent exception traces and error messages:
   ```bash
   docker compose logs --tail=100 prodml-service
   ```
2. **Health Probe Verification:**
   Verify whether `/health` returns 503 Service Unavailable or 500:
   ```bash
   curl -i http://localhost:8000/health
   ```
3. **Emergency Service Restart:**
   If unhandled PyTorch runtime exceptions are crashing request workers:
   ```bash
   docker compose restart prodml-service
   ```

## 3. Triage & Root Cause Analysis
- **Model Weight Deserialization Error:** If model weights in `/outputs/final_model` were corrupted or missing, trigger re-download:
  ```bash
  uv run python -m prodml.download
  ```
- **Out of Memory (OOM):** Check dmesg or kernel logs for OOM killer: `dmesg -T | grep -i oom`.

## 4. Verification
- Confirm 5xx error rate returns to 0%:
  ```bash
  curl -s http://localhost:8000/metrics | grep 'prodml_http_requests_total{.*status="5.."'
  ```
