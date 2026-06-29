"""Tests for serving/load_test.py."""
from __future__ import annotations

import pytest

from serving.load_test import (
    LatencyProfiler,
    LoadTestConfig,
    LoadTestResult,
    LoadTestRunner,
)


# ── LatencyProfiler ───────────────────────────────────────────────────────────

class TestLatencyProfiler:
    def test_empty_percentile_zero(self) -> None:
        p = LatencyProfiler()
        assert p.percentile("missing_label", 99) == 0.0

    def test_record_and_retrieve(self) -> None:
        p = LatencyProfiler()
        p.record("inference", 10.0)
        p.record("inference", 20.0)
        assert p.percentile("inference", 50) == pytest.approx(15.0, rel=0.1)

    def test_measure_context_manager_records(self) -> None:
        p = LatencyProfiler()
        with p.measure("test_op"):
            pass  # instant
        assert p.n_samples("test_op") == 1

    def test_measure_multiple_times(self) -> None:
        p = LatencyProfiler()
        for _ in range(5):
            with p.measure("op"):
                pass
        assert p.n_samples("op") == 5

    def test_report_has_correct_keys(self) -> None:
        p = LatencyProfiler()
        for v in range(10):
            p.record("inference", float(v))
        report = p.report()
        assert set(report["inference"].keys()) == {"n", "mean", "p50", "p95", "p99"}

    def test_report_n_matches_samples(self) -> None:
        p = LatencyProfiler()
        for v in range(7):
            p.record("op", float(v))
        assert p.report()["op"]["n"] == 7

    def test_labels_returns_recorded_labels(self) -> None:
        p = LatencyProfiler()
        p.record("a", 1.0)
        p.record("b", 2.0)
        assert set(p.labels()) == {"a", "b"}

    def test_reset_specific_label(self) -> None:
        p = LatencyProfiler()
        p.record("a", 1.0)
        p.record("b", 2.0)
        p.reset("a")
        assert p.n_samples("a") == 0
        assert p.n_samples("b") == 1

    def test_reset_all_labels(self) -> None:
        p = LatencyProfiler()
        p.record("a", 1.0)
        p.record("b", 2.0)
        p.reset()
        assert p.labels() == []

    def test_p99_ge_p50(self) -> None:
        p = LatencyProfiler()
        import numpy as np
        rng = np.random.default_rng(0)
        for v in rng.uniform(1, 100, 200):
            p.record("op", float(v))
        assert p.percentile("op", 99) >= p.percentile("op", 50)


# ── LoadTestConfig ─────────────────────────────────────────────────────────────

class TestLoadTestConfig:
    def test_defaults(self) -> None:
        cfg = LoadTestConfig()
        assert cfg.target_rps == 50
        assert cfg.p95_threshold_ms == 200.0

    def test_invalid_rps_raises(self) -> None:
        with pytest.raises(ValueError, match="target_rps"):
            LoadTestConfig(target_rps=0)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            LoadTestConfig(p95_threshold_ms=-1.0)

    def test_invalid_error_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="error_rate"):
            LoadTestConfig(error_rate_threshold=0.0)


# ── LoadTestRunner ─────────────────────────────────────────────────────────────

class TestLoadTestRunner:
    @pytest.fixture
    def fast_predict(self):
        """Instant predict function that always succeeds."""
        return lambda features: {"score": 0.5, "label": 0}

    @pytest.fixture
    def failing_predict(self):
        """Predict function that always raises."""
        def _predict(features):
            raise RuntimeError("Model error")
        return _predict

    @pytest.fixture
    def config(self) -> LoadTestConfig:
        return LoadTestConfig(
            target_rps=10,
            hold_seconds=1,
            ramp_seconds=0,
            max_requests=20,  # fast for unit tests
        )

    def test_returns_load_test_result(self, config, fast_predict) -> None:
        runner = LoadTestRunner(config, fast_predict)
        result = runner.run({"feat": 1.0})
        assert isinstance(result, LoadTestResult)

    def test_n_requests_matches_max(self, config, fast_predict) -> None:
        runner = LoadTestRunner(config, fast_predict)
        result = runner.run({"feat": 1.0})
        assert result.n_requests == 20

    def test_zero_errors_on_success(self, config, fast_predict) -> None:
        runner = LoadTestRunner(config, fast_predict)
        result = runner.run({"feat": 1.0})
        assert result.n_errors == 0
        assert result.error_rate == 0.0

    def test_all_errors_on_failing_predict(self, config, failing_predict) -> None:
        runner = LoadTestRunner(config, failing_predict)
        result = runner.run({"feat": 1.0})
        assert result.n_errors == 20
        assert result.error_rate == 1.0

    def test_sla_passes_for_fast_predict(self, fast_predict) -> None:
        config = LoadTestConfig(
            max_requests=10,
            p95_threshold_ms=10_000,  # very loose SLA
            p99_threshold_ms=10_000,
        )
        runner = LoadTestRunner(config, fast_predict)
        result = runner.run({"feat": 1.0})
        assert result.passed_sla is True

    def test_sla_fails_when_all_errors(self, config, failing_predict) -> None:
        config_strict = LoadTestConfig(
            max_requests=10,
            error_rate_threshold=0.01,
        )
        runner = LoadTestRunner(config_strict, failing_predict)
        result = runner.run({"feat": 1.0})
        assert result.passed_sla is False
        assert any("error_rate" in v for v in result.violations)

    def test_dataframe_input_works(self, config, fast_predict) -> None:
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({"feat": np.random.rand(10)})
        runner = LoadTestRunner(config, fast_predict)
        result = runner.run(df)
        assert result.n_requests == 20

    def test_p99_ge_p50(self, config, fast_predict) -> None:
        runner = LoadTestRunner(config, fast_predict)
        result = runner.run({"feat": 1.0})
        assert result.p99_ms >= result.p50_ms

    def test_summary_is_string(self, config, fast_predict) -> None:
        runner = LoadTestRunner(config, fast_predict)
        result = runner.run({"feat": 1.0})
        summary = result.summary()
        assert isinstance(summary, str)
        assert "RPS" in summary


# ── Locustfile structure ──────────────────────────────────────────────────────

class TestLocustfileStructure:
    def test_locustfile_importable(self) -> None:
        from serving import locustfile  # noqa: F401

    def test_credit_risk_user_class_exists(self) -> None:
        from serving.locustfile import CreditRiskUser
        assert hasattr(CreditRiskUser, "predict_single")
        assert hasattr(CreditRiskUser, "health_check")
        assert hasattr(CreditRiskUser, "model_info")

    def test_sample_features_non_empty(self) -> None:
        from serving.locustfile import _SAMPLE_FEATURES
        assert len(_SAMPLE_FEATURES) >= 1
        assert all(isinstance(f, dict) for f in _SAMPLE_FEATURES)
