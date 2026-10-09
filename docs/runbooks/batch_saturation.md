# Runbook: BatchQueueSaturation Alert

## 1. Alert Summary
- **Severity:** Warning
- **SLA Breach Threshold:** 90th percentile dynamic batch size $\ge 60$ (out of 64 maximum limit) sustained over 5 minutes.
- **Affected System:** BentoML Runner Micro-Batch Queue.

## 2. Immediate On-Call Actions
1. **Traffic Ingress Assessment:**
   Inspect current request RPS on Grafana dashboard (`http://localhost:3000`).
2. **Scale Serving Capacity:**
   Increase the number of worker replicas in `docker-compose.yml` or launch BentoML worker pool:
   ```bash
   docker compose up -d --scale prodml-service=2
   ```
3. **Tune Batch Window Parameters:**
   Adjust `max_batch_size` or decrease `max_latency_ms` in `serving/service.py` to drain the batch queue faster.
