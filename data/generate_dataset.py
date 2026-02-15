"""
Synthetic Fraud Transaction Dataset Generator.

Generates realistic transactional data with controllable fraud ratio
for training fraud detection models.
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_transactions(n_samples: int = 50000, fraud_ratio: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic fraud transaction data.

    Features:
        - amount: Transaction amount (USD)
        - merchant_category: Category code (0-9)
        - hour_of_day: Hour when transaction occurred (0-23)
        - day_of_week: Day of week (0=Mon, 6=Sun)
        - distance_from_home: Distance in km from cardholder's home
        - distance_from_last_txn: Distance from last transaction location
        - is_foreign: Whether transaction is international
        - velocity_last_1h: Number of transactions in last hour
        - velocity_last_24h: Number of transactions in last 24 hours
        - avg_amount_last_7d: Average transaction amount over last 7 days
        - amount_to_avg_ratio: Current amount / avg last 7 days
        - is_weekend: Whether transaction is on weekend
        - is_night: Whether transaction is between 23:00-05:00
        - card_age_days: Age of card in days
        - is_fraud: Target label (0 or 1)
    """
    rng = np.random.RandomState(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # ── Legitimate transactions ──
    legit = pd.DataFrame({
        "amount": rng.lognormal(mean=3.5, sigma=1.0, size=n_legit).clip(0.50, 5000),
        "merchant_category": rng.randint(0, 10, size=n_legit),
        "hour_of_day": rng.choice(range(24), size=n_legit, p=_hour_distribution(rng, fraud=False)),
        "day_of_week": rng.randint(0, 7, size=n_legit),
        "distance_from_home": rng.exponential(scale=5.0, size=n_legit).clip(0, 100),
        "distance_from_last_txn": rng.exponential(scale=3.0, size=n_legit).clip(0, 80),
        "is_foreign": rng.binomial(1, 0.05, size=n_legit),
        "velocity_last_1h": rng.poisson(lam=1.0, size=n_legit),
        "velocity_last_24h": rng.poisson(lam=5.0, size=n_legit),
        "avg_amount_last_7d": rng.lognormal(mean=3.5, sigma=0.8, size=n_legit).clip(5, 3000),
        "card_age_days": rng.randint(30, 3650, size=n_legit),
        "is_fraud": 0,
    })

    # ── Fraudulent transactions ──
    fraud = pd.DataFrame({
        "amount": rng.lognormal(mean=5.5, sigma=1.5, size=n_fraud).clip(50, 25000),
        "merchant_category": rng.choice([1, 3, 7, 9], size=n_fraud),  # Concentrated categories
        "hour_of_day": rng.choice(range(24), size=n_fraud, p=_hour_distribution(rng, fraud=True)),
        "day_of_week": rng.randint(0, 7, size=n_fraud),
        "distance_from_home": rng.exponential(scale=50.0, size=n_fraud).clip(5, 500),
        "distance_from_last_txn": rng.exponential(scale=40.0, size=n_fraud).clip(5, 400),
        "is_foreign": rng.binomial(1, 0.35, size=n_fraud),
        "velocity_last_1h": rng.poisson(lam=4.0, size=n_fraud),
        "velocity_last_24h": rng.poisson(lam=12.0, size=n_fraud),
        "avg_amount_last_7d": rng.lognormal(mean=3.0, sigma=0.8, size=n_fraud).clip(5, 2000),
        "card_age_days": rng.randint(1, 365, size=n_fraud),
        "is_fraud": 1,
    })

    # ── Combine and derive features ──
    df = pd.concat([legit, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Derived features
    df["amount_to_avg_ratio"] = (df["amount"] / df["avg_amount_last_7d"]).clip(0, 50)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)

    # Round floats for cleanliness
    float_cols = ["amount", "distance_from_home", "distance_from_last_txn",
                  "avg_amount_last_7d", "amount_to_avg_ratio"]
    df[float_cols] = df[float_cols].round(2)

    return df


def _hour_distribution(rng, fraud: bool) -> np.ndarray:
    """Generate hour-of-day probability distribution."""
    if fraud:
        # Fraud peaks at night (0-5 AM)
        weights = np.array([6, 6, 5, 5, 4, 3, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5], dtype=float)
    else:
        # Normal activity peaks during business hours
        weights = np.array([1, 1, 1, 1, 1, 1, 2, 3, 5, 6, 6, 5,
                            5, 5, 5, 5, 5, 5, 4, 3, 3, 2, 2, 1], dtype=float)
    return weights / weights.sum()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud dataset")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--fraud-ratio", type=float, default=0.02, help="Fraud ratio (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    output_path = args.output or os.path.join(os.path.dirname(__file__), "transactions.csv")

    print(f"Generating {args.n_samples} transactions (fraud ratio: {args.fraud_ratio})...")
    df = generate_transactions(args.n_samples, args.fraud_ratio, args.seed)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    fraud_count = df["is_fraud"].sum()
    print(f"Dataset saved to {output_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Fraud samples: {fraud_count} ({fraud_count/len(df)*100:.1f}%)")
    print(f"  Features: {list(df.columns)}")


if __name__ == "__main__":
    main()
