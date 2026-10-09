"""Evidently AI and statistical drift monitoring pipeline with PostgreSQL persistence.

Evaluates reference datasets (training/validation) against incoming batch scoring traffic:
1. Feature Extraction: Character lengths, word counts, Arabic token density, and class distributions.
2. Statistical Evaluation: Chi-Square, Wasserstein Distance, PSI, and Jensen-Shannon Divergence.
3. Evidently AI TestSuite & Report generation (HTML & JSON).
4. PostgreSQL metrics store persistence to table `monitoring_drift_records`.
5. Live Prometheus telemetry gauge publishing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from monitoring.drift_detector import (
    calculate_chi_square,
    calculate_jensen_shannon_divergence,
    calculate_psi,
    calculate_wasserstein_distance,
)
from prodml.metrics import record_drift_metrics

logger = logging.getLogger("monitoring.evidently_monitor")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ARABIC_REGEX = re.compile(r"[\u0600-\u06FF]")


def extract_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract lightweight numeric NLP features from raw text for drift analysis."""
    featured = df.copy()
    if "text" not in featured.columns:
        return featured

    texts = featured["text"].astype(str)
    featured["char_length"] = texts.apply(len)
    featured["word_count"] = texts.apply(lambda t: len(t.split()))

    def arabic_ratio(t: str) -> float:
        if not t:
            return 0.0
        arabic_chars = len(ARABIC_REGEX.findall(t))
        return arabic_chars / max(len(t), 1)

    featured["arabic_ratio"] = texts.apply(arabic_ratio)
    return featured


def run_statistical_drift_suite(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict[str, Any]:
    """Execute full statistical drift suite across categorical and continuous features."""
    ref_feat = extract_nlp_features(reference_df)
    cur_feat = extract_nlp_features(current_df)

    evaluations: list[dict[str, Any]] = []
    overall_drift_detected = False

    # 1. Categorical Label Drift (Chi-Square & JS Divergence)
    if "label" in ref_feat.columns and "label" in cur_feat.columns:
        ref_label_counts = dict(ref_feat["label"].value_counts())
        cur_label_counts = dict(cur_feat["label"].value_counts())

        chi_res = calculate_chi_square(
            ref_label_counts, cur_label_counts, feature_name="label", alpha=0.05
        )
        evaluations.append(chi_res.to_dict())
        if chi_res.drift_detected:
            overall_drift_detected = True

        all_lbls = sorted(set(ref_label_counts.keys()) | set(cur_label_counts.keys()))
        ref_probs = [ref_label_counts.get(k, 0) for k in all_lbls]
        cur_probs = [cur_label_counts.get(k, 0) for k in all_lbls]
        js_res = calculate_jensen_shannon_divergence(
            ref_probs, cur_probs, feature_name="label", threshold=0.20
        )
        evaluations.append(js_res.to_dict())
        if js_res.drift_detected:
            overall_drift_detected = True

    # 2. Continuous Features: char_length, word_count, arabic_ratio
    continuous_features = ["char_length", "word_count", "arabic_ratio"]
    for feat in continuous_features:
        if feat in ref_feat.columns and feat in cur_feat.columns:
            ref_vals = ref_feat[feat].dropna().to_numpy()
            cur_vals = cur_feat[feat].dropna().to_numpy()

            wass_res = calculate_wasserstein_distance(
                ref_vals, cur_vals, feature_name=feat, threshold=0.25
            )
            evaluations.append(wass_res.to_dict())

            psi_res = calculate_psi(ref_vals, cur_vals, feature_name=feat, threshold=0.20)
            evaluations.append(psi_res.to_dict())

            if wass_res.drift_detected or psi_res.drift_detected:
                overall_drift_detected = True

    # Publish Prometheus metrics for each evaluation
    for ev in evaluations:
        record_drift_metrics(
            metric_name=ev["method"],
            feature=ev["feature_name"],
            score=ev["statistic"],
            is_drifted=ev["drift_detected"],
        )

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
        "overall_drift_detected": overall_drift_detected,
        "evaluations": evaluations,
    }

    return summary


def generate_evidently_html_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_html_path: Path | str,
) -> bool:
    """Generate interactive Evidently HTML report if evidently is available."""
    ref_feat = extract_nlp_features(reference_df)
    cur_feat = extract_nlp_features(current_df)

    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from evidently.legacy.metric_preset import DataDriftPreset
        from evidently.legacy.report import Report

        # Select columns available in both datasets
        shared_cols = [
            c
            for c in ["char_length", "word_count", "arabic_ratio", "label"]
            if c in ref_feat.columns and c in cur_feat.columns
        ]
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=ref_feat[shared_cols],
            current_data=cur_feat[shared_cols],
        )
        report.save_html(str(out_path))
        logger.info("Evidently HTML report generated at %s", out_path)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Evidently report generation fallback: %s", exc)
        # Author an elegant fallback HTML report with full stats
        stats = run_statistical_drift_suite(reference_df, current_df)
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ProdML Drift Evaluation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #0d1117; color: #c9d1d9; }}
        h1 {{ color: #58a6ff; }}
        .badge {{ padding: 6px 12px; border-radius: 6px; font-weight: bold; }}
        .badge-danger {{ background: #da3633; color: white; }}
        .badge-success {{ background: #238636; color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ border: 1px solid #30363d; padding: 12px; text-align: left; }}
        th {{ background: #161b22; color: #8b949e; }}
        tr:nth-child(even) {{ background: #161b22; }}
    </style>
</head>
<body>
    <h1>ProdML Arabic Sentiment — Observability Drift Report</h1>
    <p>Generated: <strong>{stats["timestamp"]}</strong> | Reference: <strong>{stats["reference_rows"]} samples</strong> | Current: <strong>{stats["current_rows"]} samples</strong></p>
    <p>Status: <span class="badge {"badge-danger" if stats["overall_drift_detected"] else "badge-success"}">
        {"CRITICAL DRIFT DETECTED" if stats["overall_drift_detected"] else "STABLE (NO DRIFT)"}
    </span></p>
    <table>
        <tr><th>Feature</th><th>Method</th><th>Statistic</th><th>Threshold</th><th>P-Value</th><th>Drift Status</th></tr>
        {"".join(f"<tr><td>{e['feature_name']}</td><td>{e['method']}</td><td>{e['statistic']}</td><td>{e['threshold']}</td><td>{e['p_value']}</td><td>{'DRIFT' if e['drift_detected'] else 'PASS'}</td></tr>" for e in stats["evaluations"])}
    </table>
</body>
</html>"""
        out_path.write_text(html_content, encoding="utf-8")
        return True


def persist_drift_to_postgres(
    eval_summary: dict[str, Any],
    db_url: str | None = None,
) -> bool:
    """Persist structured drift summary records to PostgreSQL database."""
    default_db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://mlflow:mlflow_password@localhost:5432/mlflow",
    )
    target_url = db_url or default_db_url

    try:
        import psycopg2

        # Extract connection components
        conn = psycopg2.connect(target_url)
        with conn, conn.cursor() as cur:
            # 1. Ensure monitoring table exists
            cur.execute(
                """
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
                    """
            )

            batch_id = f"batch_{int(datetime.now(UTC).timestamp())}"
            ref_count = int(eval_summary.get("reference_rows", 0))
            cur_count = int(eval_summary.get("current_rows", 0))

            for ev in eval_summary.get("evaluations", []):
                cur.execute(
                    """
                        INSERT INTO monitoring_drift_records (
                            batch_id, feature_name, metric_type, statistic_value,
                            threshold_value, p_value, drift_detected,
                            reference_sample_count, current_sample_count, details
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                    (
                        batch_id,
                        ev["feature_name"],
                        ev["method"],
                        ev["statistic"],
                        ev["threshold"],
                        ev["p_value"],
                        ev["drift_detected"],
                        ref_count,
                        cur_count,
                        json.dumps(ev["details"]),
                    ),
                )

        conn.close()
        logger.info(
            "Persisted %d drift evaluations to PostgreSQL table monitoring_drift_records.",
            len(eval_summary.get("evaluations", [])),
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist drift records to PostgreSQL (%s). Continuing.", exc)
        return False


def run_drift_monitoring(
    reference_path: Path | str,
    current_path: Path | str,
    output_report_path: Path | str = "reports/evidently_drift_report.html",
    output_summary_path: Path | str = "reports/drift_summary.json",
    persist_db: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end drift monitoring workflow."""
    ref_p = Path(reference_path)
    cur_p = Path(current_path)

    if not ref_p.exists():
        raise FileNotFoundError(f"Reference dataset not found: {ref_p}")
    if not cur_p.exists():
        raise FileNotFoundError(f"Current dataset not found: {cur_p}")

    ref_df = pd.read_csv(ref_p) if ref_p.suffix == ".csv" else pd.read_parquet(ref_p)
    cur_df = pd.read_csv(cur_p) if cur_p.suffix == ".csv" else pd.read_parquet(cur_p)

    logger.info("Evaluating drift: Ref (%d rows) vs Cur (%d rows)", len(ref_df), len(cur_df))

    # 1. Statistical suite
    summary = run_statistical_drift_suite(ref_df, cur_df)

    # 2. Save JSON summary
    sum_p = Path(output_summary_path)
    sum_p.parent.mkdir(parents=True, exist_ok=True)
    sum_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Drift summary saved to %s", sum_p)

    # 3. HTML Report
    generate_evidently_html_report(ref_df, cur_df, output_report_path)

    # 4. PostgreSQL Persistence
    if persist_db:
        persist_drift_to_postgres(summary)

    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evidently AI & Statistical Drift Monitor")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/processed/train.csv",
        help="Path to reference baseline dataset",
    )
    parser.add_argument(
        "--current",
        type=str,
        default="data/drift_simulated.csv",
        help="Path to current production or drifted batch dataset",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/evidently_drift_report.html",
        help="Path to output HTML report",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="reports/drift_summary.json",
        help="Path to output JSON summary",
    )
    parser.add_argument("--no-db", action="store_true", help="Skip PostgreSQL persistence")

    args = parser.parse_args()
    run_drift_monitoring(
        reference_path=args.reference,
        current_path=args.current,
        output_report_path=args.report,
        output_summary_path=args.summary,
        persist_db=not args.no_db,
    )


if __name__ == "__main__":
    main()
