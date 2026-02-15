"""
XGBoost Fraud Detection Model Training.

Trains an XGBoost classifier on synthetic fraud data,
logs parameters/metrics to MLflow, and registers the model.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.feature_engineering import extract_features, extract_labels, get_feature_stats, FEATURE_COLUMNS


def train_xgboost(data_path: str, model_output_dir: str, mlflow_tracking_uri: str = None):
    """Train XGBoost fraud detection model."""

    # ── Setup MLflow ──
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment("fraud-detection")

    # ── Load data ──
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = extract_features(df)
    y = extract_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples ({y_train.sum():.0f} fraud)")
    print(f"Test set:     {X_test.shape[0]} samples ({y_test.sum():.0f} fraud)")

    # ── XGBoost parameters ──
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run(run_name="xgboost-baseline") as run:
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("feature_names", json.dumps(FEATURE_COLUMNS))
        mlflow.log_param("training_samples", X_train.shape[0])

        # ── Train ──
        print("Training XGBoost model...")
        model = xgb.XGBClassifier(**params, use_label_encoder=False)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50
        )

        # ── Evaluate ──
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = {
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"  {name}: {value:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # ── Feature importance ──
        importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
        mlflow.log_dict(importance, "feature_importance.json")

        # ── Save reference stats for drift detection ──
        ref_stats = get_feature_stats(df)
        mlflow.log_dict(ref_stats, "reference_stats.json")

        # ── Save model ──
        os.makedirs(model_output_dir, exist_ok=True)
        model_path = os.path.join(model_output_dir, "xgboost_v1.json")
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

        # Log and register model in MLflow
        mlflow.xgboost.log_model(
            model, "model",
            registered_model_name="fraud-xgboost"
        )

        # Save reference stats locally too
        stats_path = os.path.join(model_output_dir, "reference_stats.json")
        with open(stats_path, "w") as f:
            json.dump(ref_stats, f, indent=2)

        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"Model registered as 'fraud-xgboost'")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost fraud model")
    parser.add_argument("--data", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv"))
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "..", "models"))
    parser.add_argument("--mlflow-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()

    train_xgboost(args.data, args.output, args.mlflow_uri)


if __name__ == "__main__":
    main()
