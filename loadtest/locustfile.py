"""
Locust Load Test for ML Mesh Gateway.

Simulates concurrent prediction traffic to validate
latency and throughput under load.

Usage:
    # Headless mode (30s, 50 users):
    locust -f locustfile.py --headless -u 50 -r 10 -t 30s --host http://localhost:8000

    # Web UI mode:
    locust -f locustfile.py --host http://localhost:8000
"""

import random
from locust import HttpUser, task, between, events


def generate_transaction():
    """Generate a random transaction payload."""
    is_suspicious = random.random() < 0.1  # 10% suspicious

    if is_suspicious:
        return {
            "amount": round(random.uniform(1000, 15000), 2),
            "merchant_category": random.choice([1, 3, 7, 9]),
            "hour_of_day": random.choice([0, 1, 2, 3, 4, 23]),
            "day_of_week": random.randint(0, 6),
            "distance_from_home": round(random.uniform(50, 500), 2),
            "distance_from_last_txn": round(random.uniform(30, 300), 2),
            "is_foreign": random.choice([0, 1]),
            "velocity_last_1h": random.randint(3, 10),
            "velocity_last_24h": random.randint(10, 30),
            "avg_amount_last_7d": round(random.uniform(50, 200), 2),
            "card_age_days": random.randint(1, 60),
        }
    else:
        return {
            "amount": round(random.uniform(5, 500), 2),
            "merchant_category": random.randint(0, 9),
            "hour_of_day": random.randint(8, 20),
            "day_of_week": random.randint(0, 6),
            "distance_from_home": round(random.uniform(0, 20), 2),
            "distance_from_last_txn": round(random.uniform(0, 10), 2),
            "is_foreign": 0,
            "velocity_last_1h": random.randint(0, 3),
            "velocity_last_24h": random.randint(1, 8),
            "avg_amount_last_7d": round(random.uniform(30, 300), 2),
            "card_age_days": random.randint(100, 3000),
        }


class PredictionUser(HttpUser):
    """Simulated user sending prediction requests."""

    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self):
        """Send a prediction request."""
        payload = generate_transaction()
        with self.client.post(
            "/api/v1/predict",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "fraud_probability" not in data:
                    response.failure("Missing fraud_probability in response")
            elif response.status_code == 503:
                response.failure("Service unavailable")
            else:
                response.failure(f"Unexpected status: {response.status_code}")

    @task(1)
    def health_check(self):
        """Check gateway health."""
        self.client.get("/health")

    @task(1)
    def system_status(self):
        """Check system status."""
        self.client.get("/api/v1/status")


class BurstUser(HttpUser):
    """Simulates burst traffic patterns."""

    wait_time = between(0.01, 0.05)
    weight = 1  # Lower weight than normal user

    @task
    def burst_predict(self):
        """Rapid-fire prediction requests."""
        payload = generate_transaction()
        self.client.post("/api/v1/predict", json=payload)
