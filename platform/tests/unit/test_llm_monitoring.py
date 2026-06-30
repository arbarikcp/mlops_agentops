"""Unit tests for platform/llm/llm_monitoring.py (Day 107)."""

import pytest
from llm.llm_monitoring import (
    EvalEconomics,
    HallucinationDriftMonitor,
    OnlineEvalSampler,
    QualityDriftWindow,
    SamplingStrategy,
)


class TestQualityDriftWindow:
    def test_no_drift(self):
        w = QualityDriftWindow(metric_name="faithfulness", historical_mean=0.9, recent_scores=[0.88, 0.89, 0.9])
        assert w.has_drifted() is False

    def test_drift_detected(self):
        w = QualityDriftWindow(
            metric_name="faithfulness", historical_mean=0.9, recent_scores=[0.5, 0.5, 0.5], drift_threshold=0.1
        )
        assert w.has_drifted() is True

    def test_empty_metric_name_raises(self):
        with pytest.raises(ValueError, match="metric_name"):
            QualityDriftWindow(metric_name="", historical_mean=0.9, recent_scores=[0.5])

    def test_empty_recent_scores_raises(self):
        with pytest.raises(ValueError, match="recent_scores"):
            QualityDriftWindow(metric_name="x", historical_mean=0.9, recent_scores=[])

    def test_invalid_historical_mean_raises(self):
        with pytest.raises(ValueError, match="historical_mean"):
            QualityDriftWindow(metric_name="x", historical_mean=0, recent_scores=[0.5])

    def test_invalid_drift_threshold_raises(self):
        with pytest.raises(ValueError, match="drift_threshold"):
            QualityDriftWindow(metric_name="x", historical_mean=0.9, recent_scores=[0.5], drift_threshold=0)

    def test_recent_mean(self):
        w = QualityDriftWindow(metric_name="x", historical_mean=0.9, recent_scores=[0.8, 1.0])
        assert w.recent_mean() == 0.9

    def test_to_dict(self):
        w = QualityDriftWindow(metric_name="x", historical_mean=0.9, recent_scores=[0.9])
        assert "has_drifted" in w.to_dict()


class TestHallucinationDriftMonitor:
    def test_is_alerting_true(self):
        w = QualityDriftWindow(
            metric_name="faithfulness", historical_mean=0.9, recent_scores=[0.3, 0.3], drift_threshold=0.1
        )
        m = HallucinationDriftMonitor(faithfulness_window=w, alert_threshold=0.15)
        assert m.is_alerting() is True

    def test_is_alerting_false_no_drift(self):
        w = QualityDriftWindow(metric_name="faithfulness", historical_mean=0.9, recent_scores=[0.89])
        m = HallucinationDriftMonitor(faithfulness_window=w)
        assert m.is_alerting() is False

    def test_invalid_alert_threshold_raises(self):
        w = QualityDriftWindow(metric_name="x", historical_mean=0.9, recent_scores=[0.9])
        with pytest.raises(ValueError, match="alert_threshold"):
            HallucinationDriftMonitor(faithfulness_window=w, alert_threshold=0)

    def test_to_dict(self):
        w = QualityDriftWindow(metric_name="x", historical_mean=0.9, recent_scores=[0.9])
        m = HallucinationDriftMonitor(faithfulness_window=w)
        assert "is_alerting" in m.to_dict()


class TestOnlineEvalSampler:
    def test_every_nth(self):
        s = OnlineEvalSampler(strategy=SamplingStrategy.EVERY_NTH, every_nth=10)
        assert s.should_sample(10) is True
        assert s.should_sample(15) is False

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            OnlineEvalSampler(strategy=SamplingStrategy.FIXED_RATE, sample_rate=0)

    def test_invalid_every_nth_raises(self):
        with pytest.raises(ValueError, match="every_nth"):
            OnlineEvalSampler(strategy=SamplingStrategy.EVERY_NTH, every_nth=0)

    def test_fixed_rate_deterministic(self):
        s = OnlineEvalSampler(strategy=SamplingStrategy.FIXED_RATE, sample_rate=0.5)
        r1 = s.should_sample(7)
        r2 = s.should_sample(7)
        assert r1 == r2

    def test_adaptive_behaves_like_fixed(self):
        s = OnlineEvalSampler(strategy=SamplingStrategy.ADAPTIVE, sample_rate=0.5)
        assert isinstance(s.should_sample(3), bool)

    def test_to_dict(self):
        s = OnlineEvalSampler(strategy=SamplingStrategy.FIXED_RATE)
        assert s.to_dict()["strategy"] == "fixed_rate"


class TestEvalEconomics:
    def test_daily_eval_count(self):
        e = EvalEconomics(total_requests_per_day=10000, sample_rate=0.05, cost_per_eval_usd=0.02)
        assert e.daily_eval_count() == 500

    def test_daily_cost(self):
        e = EvalEconomics(total_requests_per_day=10000, sample_rate=0.05, cost_per_eval_usd=0.02)
        assert e.daily_cost_usd() == pytest.approx(10.0)

    def test_full_traffic_cost(self):
        e = EvalEconomics(total_requests_per_day=10000, sample_rate=0.05, cost_per_eval_usd=0.02)
        assert e.full_traffic_cost_usd() == pytest.approx(200.0)

    def test_savings(self):
        e = EvalEconomics(total_requests_per_day=10000, sample_rate=0.05, cost_per_eval_usd=0.02)
        assert e.savings_usd() == pytest.approx(190.0)

    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="sample_rate"):
            EvalEconomics(total_requests_per_day=100, sample_rate=0, cost_per_eval_usd=0.01)

    def test_negative_requests_raises(self):
        with pytest.raises(ValueError, match="total_requests_per_day"):
            EvalEconomics(total_requests_per_day=-1, sample_rate=0.1, cost_per_eval_usd=0.01)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_per_eval_usd"):
            EvalEconomics(total_requests_per_day=100, sample_rate=0.1, cost_per_eval_usd=-1)

    def test_to_dict(self):
        e = EvalEconomics(total_requests_per_day=100, sample_rate=0.1, cost_per_eval_usd=0.01)
        assert "savings_usd" in e.to_dict()
