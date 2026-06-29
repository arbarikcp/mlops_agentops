"""Locust load test scenario for the Credit Risk Inference API.

Run locally:
    locust -f serving/locustfile.py --host http://localhost:8080

Run headless (CI mode):
    locust -f serving/locustfile.py --host http://localhost:8080 \
        --users 100 --spawn-rate 10 --run-time 2m --headless \
        --html reports/load_test.html

Pass/fail thresholds (CI):
    --exit-code-on-error 1 (Locust exits 1 if error rate > 0)
    Set via --config locust.conf or inline assertions in on_stop hook.

Traffic mix (reflects production):
    90% — single-row online predictions (/v1/predict)
     9% — health checks (/health)
     1% — model info (/v1/model/info)
"""
from __future__ import annotations

import random

# Locust is an optional dev dependency — import guarded so the module can be
# imported without locust installed (needed for test_load_test.py to import
# this file for structural checks).
try:
    from locust import HttpUser, between, task
    _LOCUST_AVAILABLE = True
except ImportError:
    _LOCUST_AVAILABLE = False
    # Provide stubs so the module is importable without locust
    class HttpUser:  # type: ignore[no-redef]
        wait_time = None

    def task(weight=1):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator

    def between(low, high):  # type: ignore[misc]
        return None


# Sample feature payloads — representative of real distribution
_SAMPLE_FEATURES = [
    {"LIMIT_BAL": 50000.0, "SEX": 2.0, "EDUCATION": 2.0, "MARRIAGE": 1.0, "AGE": 35.0,
     "PAY_0": 0.0, "PAY_2": 0.0, "PAY_3": 0.0, "PAY_4": 0.0, "PAY_5": 0.0, "PAY_6": 0.0,
     "BILL_AMT1": 20000.0, "BILL_AMT2": 18000.0, "BILL_AMT3": 15000.0,
     "BILL_AMT4": 12000.0, "BILL_AMT5": 10000.0, "BILL_AMT6": 8000.0,
     "PAY_AMT1": 2000.0, "PAY_AMT2": 2000.0, "PAY_AMT3": 2000.0,
     "PAY_AMT4": 2000.0, "PAY_AMT5": 2000.0, "PAY_AMT6": 2000.0},
    {"LIMIT_BAL": 200000.0, "SEX": 1.0, "EDUCATION": 1.0, "MARRIAGE": 2.0, "AGE": 45.0,
     "PAY_0": 1.0, "PAY_2": 0.0, "PAY_3": 0.0, "PAY_4": 0.0, "PAY_5": 0.0, "PAY_6": 0.0,
     "BILL_AMT1": 80000.0, "BILL_AMT2": 75000.0, "BILL_AMT3": 70000.0,
     "BILL_AMT4": 65000.0, "BILL_AMT5": 60000.0, "BILL_AMT6": 55000.0,
     "PAY_AMT1": 5000.0, "PAY_AMT2": 5000.0, "PAY_AMT3": 5000.0,
     "PAY_AMT4": 5000.0, "PAY_AMT5": 5000.0, "PAY_AMT6": 5000.0},
]


class CreditRiskUser(HttpUser):
    """Simulates a client calling the credit risk inference API.

    Traffic mix: 90% predict, 9% health, 1% model info.
    Wait time simulates 10–100ms think time between requests.
    """

    wait_time = between(0.01, 0.1)  # 10–100ms between requests per user
    _request_counter: int = 0

    @task(weight=90)
    def predict_single(self) -> None:
        """POST /v1/predict with a random feature payload."""
        features = random.choice(_SAMPLE_FEATURES)
        CreditRiskUser._request_counter += 1
        applicant_id = CreditRiskUser._request_counter

        with self.client.post(
            "/v1/predict",
            json={"applicant_id": applicant_id, "features": features},
            catch_response=True,
            name="/v1/predict",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if not (0.0 <= data.get("score", -1) <= 1.0):
                    response.failure("score out of range [0, 1]")
            elif response.status_code == 503:
                response.failure("Model not ready")
            else:
                response.failure(f"Unexpected status {response.status_code}")

    @task(weight=9)
    def health_check(self) -> None:
        """GET /health — liveness probe check."""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")

    @task(weight=1)
    def model_info(self) -> None:
        """GET /v1/model/info — periodic model metadata check."""
        with self.client.get(
            "/v1/model/info", catch_response=True, name="/v1/model/info"
        ) as response:
            if response.status_code not in (200, 503):
                response.failure(f"Unexpected status {response.status_code}")
