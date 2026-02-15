"""
PyTorch Shallow NN Fraud Detection Model Training.

Trains a 3-layer MLP on synthetic fraud data,
logs parameters/metrics to MLflow, and registers the model.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.feature_engineering import extract_features, extract_labels, FEATURE_COLUMNS


class FraudMLP(nn.Module):
    """3-layer MLP for fraud detection."""

    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def train_pytorch(data_path: str, model_output_dir: str, mlflow_tracking_uri: str = None):
    """Train PyTorch MLP fraud detection model."""

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

    # ── Standardize ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler params for inference
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }

    # ── DataLoaders ──
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)

    # ── Model ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims = [128, 64, 32]
    model = FraudMLP(input_dim=len(FEATURE_COLUMNS), hidden_dims=hidden_dims).to(device)

    # Class-weighted loss
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Override model's final sigmoid for proper BCEWithLogitsLoss
    # Actually, let's just use BCELoss since model has sigmoid
    criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    params = {
        "hidden_dims": str(hidden_dims),
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 30,
        "optimizer": "Adam",
        "weight_decay": 1e-4,
        "device": str(device),
    }

    with mlflow.start_run(run_name="pytorch-mlp") as run:
        mlflow.log_params(params)

        # ── Train ──
        print("Training PyTorch MLP...")
        n_epochs = 30
        best_auc = 0

        for epoch in range(n_epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # Evaluate
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)
                    preds = model(X_batch).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(y_batch.numpy())

            epoch_auc = roc_auc_score(all_labels, all_preds)
            avg_loss = train_loss / len(train_loader)

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_auc": epoch_auc,
            }, step=epoch)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs} | Loss: {avg_loss:.4f} | AUC: {epoch_auc:.4f}")

            if epoch_auc > best_auc:
                best_auc = epoch_auc

        # ── Final evaluation ──
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        y_pred_proba = np.array(all_preds)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        y_true = np.array(all_labels)

        metrics = {
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"  {name}: {value:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        # ── Save model ──
        os.makedirs(model_output_dir, exist_ok=True)
        model_path = os.path.join(model_output_dir, "pytorch_v1.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": len(FEATURE_COLUMNS),
            "hidden_dims": hidden_dims,
            "scaler_params": scaler_params,
        }, model_path)
        print(f"Model saved to {model_path}")

        # Save scaler separately for inference
        scaler_path = os.path.join(model_output_dir, "pytorch_scaler.json")
        with open(scaler_path, "w") as f:
            json.dump(scaler_params, f, indent=2)

        # Log model to MLflow
        mlflow.pytorch.log_model(
            model, "model",
            registered_model_name="fraud-pytorch"
        )

        print(f"\nMLflow Run ID: {run.info.run_id}")
        print("Model registered as 'fraud-pytorch'")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch MLP fraud model")
    parser.add_argument("--data", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data", "transactions.csv"))
    parser.add_argument("--output", type=str, default=os.path.join(os.path.dirname(__file__), "..", "models"))
    parser.add_argument("--mlflow-uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"))
    args = parser.parse_args()

    train_pytorch(args.data, args.output, args.mlflow_uri)


if __name__ == "__main__":
    main()
