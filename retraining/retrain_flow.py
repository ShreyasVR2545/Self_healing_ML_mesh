"""
Automated Retraining Pipeline — Prefect Flow.

Orchestrates the full retraining lifecycle:
  1. Check drift scores
  2. Generate fresh training data
  3. Retrain XGBoost model
  4. Register new model version in MLflow
  5. Deploy as canary (10% traffic)

Can be triggered by drift detection or on a schedule.
"""

import os
import sys
import json
import time
import logging
from datetime import timedelta

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@task(name="check_drift", retries=2, retry_delay_seconds=10)
def check_drift() -> dict:
    """Query drift detector and decide if retraining is needed."""
    logger = get_run_logger()

    from feature_store.store import FeatureStore
    from feature_store.drift import DriftDetector
    from training.feature_engineering import FEATURE_COLUMNS

    store = FeatureStore()
    detector = DriftDetector()

    # Load reference data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv")
    if os.path.exists(data_path):
        detector.set_reference_data_from_csv(data_path, FEATURE_COLUMNS)
    else:
        logger.warning("No reference data found, skipping drift check")
        return {"drift_detected": True, "reason": "no_reference_data"}

    # Get current live features
    current_features = store.get_feature_arrays(FEATURE_COLUMNS, window_minutes=60)

    if not current_features:
        logger.info("No recent feature data available")
        return {"drift_detected": False, "reason": "no_live_data"}

    # Run drift detection
    results = detector.check_drift(current_features)
    summary = detector.get_drift_summary(results)

    logger.info(f"Drift check: {summary['drifted_features']}/{summary['total_features_checked']} "
                f"features drifted")

    store.close()
    return {
        "drift_detected": summary["overall_drift"],
        "summary": summary,
        "reason": "drift_detected" if summary["overall_drift"] else "no_drift",
    }


@task(name="generate_fresh_data", retries=1)
def generate_fresh_data(seed: int = None) -> str:
    """Generate a new batch of synthetic training data."""
    logger = get_run_logger()

    from data.generate_dataset import generate_transactions

    seed = seed or int(time.time()) % 10000
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"transactions_retrain_{seed}.csv"
    )

    df = generate_transactions(n_samples=50000, fraud_ratio=0.02, seed=seed)
    df.to_csv(output_path, index=False)

    logger.info(f"Generated fresh dataset: {output_path} ({len(df)} samples)")
    return output_path


@task(name="retrain_model", retries=1, timeout_seconds=600)
def retrain_model(data_path: str) -> dict:
    """Retrain XGBoost model on fresh data."""
    logger = get_run_logger()

    from training.train_xgboost import train_xgboost

    model_output = os.path.join(os.path.dirname(__file__), "..", "models")
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")

    logger.info(f"Retraining XGBoost on {data_path}...")
    model, metrics = train_xgboost(data_path, model_output, mlflow_uri)

    logger.info(f"Retrained model metrics: AUC={metrics['auc_roc']:.4f}, F1={metrics['f1']:.4f}")

    return {
        "metrics": metrics,
        "model_path": os.path.join(model_output, "xgboost_v1.json"),
    }


@task(name="register_model")
def register_model(training_result: dict) -> dict:
    """Register new model version in MLflow."""
    logger = get_run_logger()

    # MLflow registration already happens in train_xgboost
    # This task validates the registration
    import mlflow

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)

    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions("name='fraud-xgboost'")
        latest = max(versions, key=lambda v: int(v.version)) if versions else None

        if latest:
            logger.info(f"Latest registered version: {latest.version} "
                       f"(run_id={latest.run_id})")
            return {
                "model_name": "fraud-xgboost",
                "version": latest.version,
                "run_id": latest.run_id,
                "status": "registered",
            }
    except Exception as e:
        logger.warning(f"Could not query model registry: {e}")

    return {
        "model_name": "fraud-xgboost",
        "status": "training_complete",
        "metrics": training_result.get("metrics"),
    }


@task(name="deploy_canary")
def deploy_canary(registration_result: dict) -> dict:
    """Update gateway to route 10% traffic to new model version."""
    logger = get_run_logger()

    import httpx

    gateway_url = os.environ.get("GATEWAY_URL", "http://localhost:8000")

    try:
        response = httpx.post(
            f"{gateway_url}/api/v1/traffic",
            json={"xgboost": 0.9, "pytorch": 0.1},
            timeout=10.0
        )
        response.raise_for_status()
        logger.info("Canary deployment: shifted 10% traffic to retrained model")
        return {"deployed": True, "traffic_split": {"xgboost": 0.9, "pytorch": 0.1}}

    except Exception as e:
        logger.error(f"Failed to update gateway traffic: {e}")
        return {"deployed": False, "error": str(e)}


@flow(name="automated-retraining", log_prints=True)
def retraining_flow(force: bool = False):
    """Main retraining pipeline flow.

    Args:
        force: If True, skip drift check and retrain anyway
    """
    logger = get_run_logger()
    logger.info("Starting automated retraining pipeline...")

    # Step 1: Check drift
    if not force:
        drift_result = check_drift()
        if not drift_result["drift_detected"]:
            logger.info(f"No drift detected ({drift_result['reason']}). Skipping retraining.")
            return {"status": "skipped", "reason": drift_result["reason"]}
        logger.info(f"Drift detected: {drift_result['reason']}")

    # Step 2: Generate fresh data
    data_path = generate_fresh_data()

    # Step 3: Retrain
    training_result = retrain_model(data_path)

    # Step 4: Register
    registration = register_model(training_result)

    # Step 5: Deploy canary
    deployment = deploy_canary(registration)

    summary = {
        "status": "completed",
        "training_metrics": training_result.get("metrics"),
        "registration": registration,
        "deployment": deployment,
    }

    logger.info(f"Retraining pipeline complete: {json.dumps(summary, indent=2, default=str)}")
    return summary


if __name__ == "__main__":
    retraining_flow(force=True)
