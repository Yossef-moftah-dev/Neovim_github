"""Script demonstrating code-free model swap via MLflow Model Registry.

Proves that transitioning stages in the Model Registry changes the active serving
model without requiring redeployment or modifying application code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
from mlflow.tracking import MlflowClient

from prodml.predict import SentimentPredictor


def main() -> None:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    model_name = "arabic-sentiment-model"
    stage_uri = f"models:/{model_name}/Production"
    client = MlflowClient(tracking_uri=tracking_uri)

    print("=" * 80)
    print("DEMONSTRATION: Code-Free Dynamic Model Swap via Model Registry")
    print(f"Tracking Server: {tracking_uri}")
    print(f"Target Stage URI: {stage_uri}")
    print("=" * 80)

    # 1. Inspect current Production version
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not prod_versions:
        print("Error: No Production model found in registry.")
        sys.exit(1)

    initial_version = prod_versions[0].version
    print(f"\n[Step 1] Initial State: Production is Version {initial_version}")

    # Load predictor via Stage URI
    predictor_v1 = SentimentPredictor.load(stage_uri, tracking_uri=tracking_uri)
    print(
        f"  Loaded Predictor Model Version: {predictor_v1.model_version} (Stage: {predictor_v1.model_stage})"
    )
    sample_text = "الخدمة ممتازة والتوصيل سريع جدا"
    pred1 = predictor_v1.predict_one(sample_text)
    print(
        f"  Prediction under v{predictor_v1.model_version}: label={pred1.label}, confidence={pred1.confidence:.4f}"
    )

    # 2. Pick an alternate version to swap into Production
    all_versions = [v.version for v in client.search_model_versions(f"name='{model_name}'")]
    alternate_version = [v for v in all_versions if v != initial_version]
    if not alternate_version:
        print("Note: Only one version exists, creating alias test.")
        target_swap_version = initial_version
    else:
        target_swap_version = alternate_version[0]

    print("\n[Step 2] Executing Code-Free Stage Swap in Model Registry...")
    print(f"  Promoting Version {target_swap_version} to Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=target_swap_version,
        stage="Production",
        archive_existing_versions=True,
    )

    # 3. Reload model via unchanged Stage URI
    print(f"\n[Step 3] Reloading serving predictor from identical URI: {stage_uri}")
    predictor_v2 = SentimentPredictor.load(stage_uri, tracking_uri=tracking_uri)
    print(
        f"  Loaded Predictor Model Version: {predictor_v2.model_version} (Stage: {predictor_v2.model_stage})"
    )
    pred2 = predictor_v2.predict_one(sample_text)
    print(
        f"  Prediction under v{predictor_v2.model_version}: label={pred2.label}, confidence={pred2.confidence:.4f}"
    )

    # 4. Verify swap succeeded without code change
    assert predictor_v2.model_version == target_swap_version, "Model version did not update!"
    print(
        f"\n✓ Verification Successful: Model swapped from v{initial_version} to v{target_swap_version} with zero code changes!"
    )

    # 5. Restore champion version (v4) to Production
    print(f"\n[Step 5] Restoring Champion Version {initial_version} to Production...")
    client.transition_model_version_stage(
        name=model_name,
        version=initial_version,
        stage="Production",
        archive_existing_versions=True,
    )
    restored_prod = client.get_latest_versions(model_name, stages=["Production"])
    print(f"  Restored Production Model Version: {restored_prod[0].version}")
    print("=" * 80)


if __name__ == "__main__":
    main()
