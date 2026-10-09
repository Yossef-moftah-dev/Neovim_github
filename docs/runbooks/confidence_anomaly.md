# Runbook: LowPredictionConfidenceAnomaly Alert

## 1. Alert Summary
- **Severity:** Warning
- **SLA Breach Threshold:** Mean prediction confidence $< 65\%$ sustained over 5 minutes.
- **Affected System:** Classification Uncertainty / Model Calibration.

## 2. Immediate On-Call Actions
1. **Sample Uncertainty Inferences:**
   Extract low-confidence logs from application stdout:
   ```bash
   docker compose logs prodml-service | grep -i "confidence" | tail -n 50
   ```
2. **Inspect Language / Dialect Input:**
   Check if incoming traffic contains English/Franco-Arabic, broken Unicode, or heavy emoji strings bypassing sanitization.
3. **Trigger Statistical Drift Suite:**
   ```bash
   uv run python monitoring/evidently_monitor.py
   ```
