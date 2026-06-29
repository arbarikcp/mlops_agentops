"""Tests for serving/api.py — FastAPI endpoints using TestClient."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from serving.api import app, PredictRequest
from serving.inference import PredictionResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(applicant_id: int = 1, score: float = 0.75) -> PredictionResult:
    from datetime import datetime, timezone
    return PredictionResult(
        applicant_id=applicant_id,
        score=score,
        label=int(score >= 0.5),
        latency_ms=1.5,
        model_version="v_test",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client_not_ready():
    """TestClient where the model is NOT loaded (lifespan may fail gracefully)."""
    import serving.api as api_module
    with TestClient(app, raise_server_exceptions=False) as client:
        # Ensure not-ready state (lifespan failure leaves _model_ready=False)
        api_module._model_ready = False
        api_module._runner = None
        yield client
        api_module._model_ready = False
        api_module._runner = None


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.model_version = "v_test"
    runner.threshold = 0.5
    runner.model_path = "/models/credit_risk_model.onnx"
    runner.feature_names = ["feat_0", "feat_1"]
    runner.predict_single.side_effect = lambda features, applicant_id=0: _make_result(applicant_id)
    return runner


@pytest.fixture
def client_ready(mock_runner):
    """TestClient where the model IS loaded (set AFTER lifespan runs)."""
    import serving.api as api_module
    with TestClient(app, raise_server_exceptions=False) as client:
        # Lifespan has already run (and may have failed); override state now
        api_module._model_ready = True
        api_module._runner = mock_runner
        yield client
        api_module._model_ready = False
        api_module._runner = None


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_always_200(self, client_not_ready) -> None:
        resp = client_not_ready.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client_not_ready) -> None:
        resp = client_not_ready.get("/health")
        assert resp.json()["status"] == "ok"

    def test_uptime_present(self, client_not_ready) -> None:
        resp = client_not_ready.get("/health")
        assert "uptime_seconds" in resp.json()

    def test_uptime_non_negative(self, client_not_ready) -> None:
        resp = client_not_ready.get("/health")
        assert resp.json()["uptime_seconds"] >= 0


# ── /ready ────────────────────────────────────────────────────────────────────

class TestReady:
    def test_503_when_not_ready(self, client_not_ready) -> None:
        resp = client_not_ready.get("/ready")
        assert resp.status_code == 503

    def test_ready_false_when_not_loaded(self, client_not_ready) -> None:
        resp = client_not_ready.get("/ready")
        assert resp.json()["ready"] is False

    def test_200_when_ready(self, client_ready) -> None:
        resp = client_ready.get("/ready")
        assert resp.status_code == 200

    def test_ready_true_when_loaded(self, client_ready) -> None:
        resp = client_ready.get("/ready")
        assert resp.json()["ready"] is True

    def test_model_version_in_ready_response(self, client_ready) -> None:
        resp = client_ready.get("/ready")
        assert resp.json()["model_version"] == "v_test"


# ── /v1/predict ───────────────────────────────────────────────────────────────

class TestPredict:
    def _body(self, applicant_id: int = 1) -> dict:
        return {
            "applicant_id": applicant_id,
            "features": {"feat_0": 0.5, "feat_1": 1.2},
        }

    def test_503_when_not_ready(self, client_not_ready) -> None:
        resp = client_not_ready.post("/v1/predict", json=self._body())
        assert resp.status_code == 503

    def test_200_with_valid_input(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict", json=self._body())
        assert resp.status_code == 200

    def test_response_has_score(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict", json=self._body())
        data = resp.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_response_has_label(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict", json=self._body())
        assert resp.json()["label"] in (0, 1)

    def test_applicant_id_echoed(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict", json=self._body(applicant_id=42))
        assert resp.json()["applicant_id"] == 42

    def test_model_version_in_response(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict", json=self._body())
        assert resp.json()["model_version"] == "v_test"

    def test_422_for_missing_features(self, client_ready) -> None:
        # features key absent entirely
        resp = client_ready.post("/v1/predict", json={"applicant_id": 1})
        assert resp.status_code == 422

    def test_422_for_zero_applicant_id(self, client_ready) -> None:
        body = {"applicant_id": 0, "features": {"feat_0": 1.0}}
        resp = client_ready.post("/v1/predict", json=body)
        assert resp.status_code == 422

    def test_422_for_negative_applicant_id(self, client_ready) -> None:
        body = {"applicant_id": -5, "features": {"feat_0": 1.0}}
        resp = client_ready.post("/v1/predict", json=body)
        assert resp.status_code == 422

    def test_422_for_string_feature_value(self, client_ready) -> None:
        body = {"applicant_id": 1, "features": {"feat_0": "not_a_float"}}
        resp = client_ready.post("/v1/predict", json=body)
        assert resp.status_code == 422


# ── /v1/predict/batch ─────────────────────────────────────────────────────────

class TestPredictBatch:
    def _body(self, n: int = 3) -> dict:
        return {
            "rows": [
                {"applicant_id": i + 1, "features": {"feat_0": float(i), "feat_1": 1.0}}
                for i in range(n)
            ]
        }

    def test_200_with_valid_input(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict/batch", json=self._body())
        assert resp.status_code == 200

    def test_n_rows_matches_input(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict/batch", json=self._body(n=5))
        assert resp.json()["n_rows"] == 5

    def test_predictions_count_matches(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict/batch", json=self._body(n=4))
        assert len(resp.json()["predictions"]) == 4

    def test_503_when_not_ready(self, client_not_ready) -> None:
        resp = client_not_ready.post("/v1/predict/batch", json=self._body())
        assert resp.status_code == 503

    def test_422_for_empty_rows(self, client_ready) -> None:
        resp = client_ready.post("/v1/predict/batch", json={"rows": []})
        assert resp.status_code == 422


# ── /v1/model/info ────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_503_when_not_ready(self, client_not_ready) -> None:
        resp = client_not_ready.get("/v1/model/info")
        assert resp.status_code == 503

    def test_200_when_ready(self, client_ready) -> None:
        resp = client_ready.get("/v1/model/info")
        assert resp.status_code == 200

    def test_info_has_model_version(self, client_ready) -> None:
        resp = client_ready.get("/v1/model/info")
        assert resp.json()["model_version"] == "v_test"

    def test_info_has_threshold(self, client_ready) -> None:
        resp = client_ready.get("/v1/model/info")
        assert resp.json()["threshold"] == 0.5


# ── PredictRequest Pydantic validation (unit tests) ───────────────────────────

class TestPredictRequestSchema:
    def test_valid_request_builds(self) -> None:
        req = PredictRequest(applicant_id=1, features={"a": 1.0})
        assert req.applicant_id == 1

    def test_nan_raises(self) -> None:
        import math
        with pytest.raises(Exception):
            PredictRequest(applicant_id=1, features={"a": float("nan")})

    def test_inf_raises(self) -> None:
        with pytest.raises(Exception):
            PredictRequest(applicant_id=1, features={"a": float("inf")})

    def test_neg_inf_raises(self) -> None:
        with pytest.raises(Exception):
            PredictRequest(applicant_id=1, features={"a": float("-inf")})

    def test_zero_applicant_id_raises(self) -> None:
        with pytest.raises(Exception):
            PredictRequest(applicant_id=0, features={"a": 1.0})

    def test_empty_features_raises(self) -> None:
        with pytest.raises(Exception):
            PredictRequest(applicant_id=1, features={})

    def test_multiple_features_accepted(self) -> None:
        req = PredictRequest(applicant_id=99, features={"a": 1.0, "b": 2.0, "c": -3.5})
        assert len(req.features) == 3
