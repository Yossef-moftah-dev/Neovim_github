# Runbook: HighInferenceLatencyP95 Alert

## 1. Alert Summary
- **Severity:** Critical
- **SLA Breach Threshold:** 95th percentile inference latency $> 100\text{ ms}$ sustained over 2 minutes.
- **Affected System:** ProdML Serving Layer (`prodml-service` / BentoML / Triton).

## 2. Immediate On-Call Actions
1. **Canary Status Assessment:**
   If a canary deployment is currently in progress, trigger instant sub-second rollback:
   ```bash
   ./serving/canary_promote.sh rollback
   ```
2. **Resource Saturation Inspection:**
   Check whether CPU, memory, or thread limits are saturated:
   ```bash
   docker stats --no-stream prodml-service
   ```
3. **Queue & Batch Concurrency:**
   Verify current batch queue size and latency metrics on Grafana dashboard:
   `http://localhost:3000/d/prodml-observability-v1` (Row 2).

## 3. Triage & Root Cause Analysis
- **CPU Throttling:** Check Docker CPU quota: `docker inspect prodml-service | grep -i cpu`.
- **Large Input Payload:** Check if incoming batch requests contain unusually long text inputs exceeding sequence length limits.
- **Worker Lockup:** If worker threads are blocking on I/O, restart the serving container:
  ```bash
  docker compose restart prodml-service
  ```

## 4. Verification & Escalation
- Confirm p95 latency drops back below 50ms:
  ```bash
  curl -s http://localhost:8000/health
  ```
- If latency remains $> 100\text{ ms}$, scale serving replicas or activate Triton GPU inference cluster.
