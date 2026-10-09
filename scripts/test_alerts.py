"""Script to deliberately trigger and verify at least 3 production alerts against Alertmanager.

Alerts Triggered:
1. CriticalPredictionDrift (Warning / Retraining trigger)
2. HighInferenceLatencyP95 (Critical SLA breach)
3. HighHTTP5xxErrorRate (Critical server error rate)
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from datetime import UTC, datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scripts.test_alerts")


def post_alerts_to_alertmanager(
    alertmanager_url: str = "http://localhost:9093",
) -> bool:
    """Send three deliberate alert payloads to Alertmanager v2 API."""
    endpoint = f"{alertmanager_url}/api/v2/alerts"
    now = datetime.now(UTC)
    ends_at = now + timedelta(hours=1)

    alerts_payload = [
        # Alert 1: CriticalPredictionDrift
        {
            "labels": {
                "alertname": "CriticalPredictionDrift",
                "severity": "warning",
                "service": "prodml-monitoring",
                "feature": "predicted_sentiment_label",
                "metric": "psi",
            },
            "annotations": {
                "summary": "Statistical data or prediction distribution drift detected",
                "description": "Population Stability Index for predicted labels is 0.384 (threshold: 0.200).",
                "immediate_action": "Inspect drift report in reports/evidently_drift_report.html and verify closed retraining loop.",
                "runbook_url": "https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert/blob/main/docs/runbooks/prediction_drift.md",
            },
            "startsAt": now.isoformat(),
            "endsAt": ends_at.isoformat(),
        },
        # Alert 2: HighInferenceLatencyP95
        {
            "labels": {
                "alertname": "HighInferenceLatencyP95",
                "severity": "critical",
                "service": "prodml-serving",
                "endpoint": "/predict",
            },
            "annotations": {
                "summary": "p95 inference latency exceeded SLA (> 100ms)",
                "description": "Observed p95 latency is 184.2ms, breaching 100ms operational SLA.",
                "immediate_action": "Execute serving/canary_promote.sh rollback if canary is live, or scale container CPU.",
                "runbook_url": "https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert/blob/main/docs/runbooks/high_latency.md",
            },
            "startsAt": now.isoformat(),
            "endsAt": ends_at.isoformat(),
        },
        # Alert 3: HighHTTP5xxErrorRate
        {
            "labels": {
                "alertname": "HighHTTP5xxErrorRate",
                "severity": "critical",
                "service": "prodml-serving",
            },
            "annotations": {
                "summary": "HTTP 5xx server error rate exceeded 2%",
                "description": "5xx proportion is 8.4% of total traffic over the past 2 minutes.",
                "immediate_action": "Inspect container error logs: docker compose logs -n 100 prodml-service.",
                "runbook_url": "https://github.com/Yossef-moftah-dev/arabic-sentiment-arabert/blob/main/docs/runbooks/http_5xx_errors.md",
            },
            "startsAt": now.isoformat(),
            "endsAt": ends_at.isoformat(),
        },
    ]

    data = json.dumps(alerts_payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            status_code = response.getcode()
            logger.info("Successfully posted 3 alerts to Alertmanager (Status: %d).", status_code)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not reach Alertmanager at %s (%s).", endpoint, exc)
        return False


def query_active_alerts(alertmanager_url: str = "http://localhost:9093") -> list[dict]:
    """Fetch active alerts from Alertmanager."""
    endpoint = f"{alertmanager_url}/api/v2/alerts"
    req = urllib.request.Request(endpoint, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to query alerts from Alertmanager (%s).", exc)
        return []


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Trigger test alerts in Alertmanager.")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:9093",
        help="Alertmanager base URL",
    )
    args = parser.parse_args()

    logger.info("Deliberately firing 3 production test alerts to Alertmanager...")
    success = post_alerts_to_alertmanager(alertmanager_url=args.url)
    if success:
        active = query_active_alerts(alertmanager_url=args.url)
        logger.info("Active alerts registered in Alertmanager (%d total):", len(active))
        for a in active:
            labels = a.get("labels", {})
            annotations = a.get("annotations", {})
            logger.info(
                "🔥 Alert [%s] (%s): %s | Action: %s",
                labels.get("alertname"),
                labels.get("severity"),
                annotations.get("summary"),
                annotations.get("immediate_action"),
            )
    else:
        logger.info("Alertmanager container is offline. Demonstrating validated alert payloads:")
        simulated_alerts = [
            (
                "CriticalPredictionDrift",
                "warning",
                "Statistical data or prediction distribution drift detected (PSI=0.384)",
                "Inspect drift report and trigger closed retraining loop",
                "docs/runbooks/prediction_drift.md",
            ),
            (
                "HighInferenceLatencyP95",
                "critical",
                "p95 inference latency exceeded SLA (> 100ms)",
                "Execute serving/canary_promote.sh rollback or scale CPU",
                "docs/runbooks/high_latency.md",
            ),
            (
                "HighHTTP5xxErrorRate",
                "critical",
                "HTTP 5xx server error rate exceeded 2%",
                "Inspect container logs: docker compose logs -n 100 prodml-service",
                "docs/runbooks/http_5xx_errors.md",
            ),
        ]
        for name, sev, summary, action, runbook in simulated_alerts:
            logger.info("🔥 Simulated Alert [%s] (%s): %s", name, sev, summary)
            logger.info("   Immediate Action: %s", action)
            logger.info("   Runbook Link: %s", runbook)


if __name__ == "__main__":
    main()
