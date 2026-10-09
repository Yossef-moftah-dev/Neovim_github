# Runbook: CriticalPredictionDrift Alert

## 1. Alert Summary
- **Severity:** Warning / Action Required
- **SLA Breach Threshold:** Population Stability Index (PSI) $> 0.20$ or Wasserstein Distance $> 0.25$ or Chi-Square $p < 0.05$.
- **Affected System:** ProdML Model Inference Outputs & Feature Pipeline.

## 2. Immediate On-Call Actions
1. **Review Drift Report:**
   Open the latest generated HTML report or inspect the JSON summary:
   ```bash
   cat reports/drift_summary.json
   ```
2. **Check Closed Retraining Loop Status:**
   Verify if the automated retraining trigger fired and check cooldown status:
   ```bash
   cat outputs/retraining_state.json
   ```
3. **Inspect Airflow DAG Runs:**
   Check Airflow Webserver UI at `http://localhost:8080` or execute manual trigger:
   ```bash
   uv run python monitoring/retraining_trigger.py --force
   ```

## 3. Triage & Root Cause Analysis
- **Sudden Event / Slang Shift:** Check recent review texts for external events (e.g. promotional holiday, regional dialect surge).
- **Data Pipeline Ingestion Bug:** Verify upstream scraper or ingestion formatting has not altered Arabic token encodings.

## 4. Verification
- Verify that a fresh model run completes training, passes the quality gate, and is promoted to Model Registry Staging.
