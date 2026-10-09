# Runbook: ModelUnloadedOrServiceDegraded Alert

## 1. Alert Summary
- **Severity:** Critical
- **SLA Breach Threshold:** Model memory residency indicator `prodml_model_loaded == 0` or Prometheus scrape target `up == 0`.
- **Affected System:** ProdML Model Predictor / Process State.

## 2. Immediate On-Call Actions
1. **Dynamic Model Hot-Reload:**
   Attempt a zero-downtime hot-swap from registry or local weights without restarting container:
   ```bash
   curl -X POST "http://localhost:8000/model/reload?model_uri=/outputs/final_model"
   ```
2. **Container Status Check:**
   Verify if Docker container is running:
   ```bash
   docker ps -f name=prodml-service
   ```
3. **Restart Service:**
   If unresponsive:
   ```bash
   docker compose restart prodml-service
   ```

## 3. Verification
- Verify `/health` responds with `status: ok` and `model_loaded: true`:
  ```bash
  curl -s http://localhost:8000/health
  ```
