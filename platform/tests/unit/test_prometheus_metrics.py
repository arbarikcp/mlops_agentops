"""Tests for monitoring/prometheus_metrics.py — MetricSnapshot, MLMetricsCollector."""
from __future__ import annotations

import pytest

from monitoring.prometheus_metrics import MLMetricsCollector, MetricSnapshot


# ── MetricSnapshot ─────────────────────────────────────────────────────────────

class TestMetricSnapshot:
    def test_basic(self) -> None:
        s = MetricSnapshot("requests", 42.0, help_text="Total requests")
        assert s.value == 42.0
        assert s.metric_type == "gauge"

    def test_labels(self) -> None:
        s = MetricSnapshot("drift", 0.25, labels={"feature": "pay_ratio"})
        assert s.labels["feature"] == "pay_ratio"


# ── MLMetricsCollector — record & collect ─────────────────────────────────────

class TestMLMetricsCollector:
    def test_empty_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="prefix"):
            MLMetricsCollector(prefix="")

    def test_prediction_count_increments(self) -> None:
        c = MLMetricsCollector()
        c.record_prediction(latency_ms=10.0)
        c.record_prediction(latency_ms=20.0)
        snaps = c.collect()
        total = next(s for s in snaps if "requests_total" in s.name)
        assert total.value == 2.0

    def test_error_count_increments(self) -> None:
        c = MLMetricsCollector()
        c.record_prediction(10.0, error=False)
        c.record_prediction(10.0, error=True)
        snaps = c.collect()
        err = next(s for s in snaps if "errors_total" in s.name)
        assert err.value == 1.0

    def test_avg_latency(self) -> None:
        c = MLMetricsCollector()
        c.record_prediction(10.0)
        c.record_prediction(30.0)
        snaps = c.collect()
        avg = next(s for s in snaps if "latency_ms_avg" in s.name)
        assert avg.value == pytest.approx(20.0)

    def test_latency_histogram_buckets_present(self) -> None:
        c = MLMetricsCollector()
        c.record_prediction(15.0)  # falls in 25ms bucket
        snaps = c.collect()
        bucket_snaps = [s for s in snaps if "latency_ms_bucket" in s.name]
        assert len(bucket_snaps) > 0
        inf_bucket = next(s for s in bucket_snaps if s.labels.get("le") == "+Inf")
        assert inf_bucket.value == 1.0

    def test_drift_score_recorded(self) -> None:
        c = MLMetricsCollector()
        c.record_drift("pay_ratio", psi=0.25, severity="high")
        snaps = c.collect()
        drift = next(s for s in snaps if "drift_score" in s.name)
        assert drift.value == pytest.approx(0.25)
        assert drift.labels["feature"] == "pay_ratio"
        assert drift.labels["severity"] == "high"

    def test_auc_recorded(self) -> None:
        c = MLMetricsCollector()
        c.record_auc(0.78)
        snaps = c.collect()
        auc = next(s for s in snaps if "model_auc" in s.name)
        assert auc.value == pytest.approx(0.78)

    def test_auc_absent_when_not_recorded(self) -> None:
        c = MLMetricsCollector()
        snaps = c.collect()
        names = [s.name for s in snaps]
        assert not any("model_auc" in n for n in names)

    def test_feature_freshness_recorded(self) -> None:
        c = MLMetricsCollector()
        c.record_feature_freshness("payment_features", 3.5)
        snaps = c.collect()
        fresh = next(s for s in snaps if "freshness" in s.name)
        assert fresh.value == pytest.approx(3.5)
        assert fresh.labels["view"] == "payment_features"

    def test_approval_rate(self) -> None:
        c = MLMetricsCollector()
        c.record_approval_rate(0.72)
        snaps = c.collect()
        ar = next(s for s in snaps if "approval_rate" in s.name)
        assert ar.value == pytest.approx(0.72)

    def test_default_rate(self) -> None:
        c = MLMetricsCollector()
        c.record_default_rate(0.22)
        snaps = c.collect()
        dr = next(s for s in snaps if "default_rate" in s.name)
        assert dr.value == pytest.approx(0.22)

    def test_reset_clears_all(self) -> None:
        c = MLMetricsCollector()
        c.record_prediction(10.0)
        c.record_auc(0.8)
        c.record_drift("pay_ratio", 0.25)
        c.reset()
        snaps = c.collect()
        total = next(s for s in snaps if "requests_total" in s.name)
        assert total.value == 0.0
        assert not any("model_auc" in s.name for s in snaps)
        assert not any("drift_score" in s.name for s in snaps)

    def test_multiple_drift_features(self) -> None:
        c = MLMetricsCollector()
        c.record_drift("pay_ratio", 0.25, "high")
        c.record_drift("util_rate", 0.05, "none")
        snaps = c.collect()
        drift_snaps = [s for s in snaps if "drift_score" in s.name]
        assert len(drift_snaps) == 2


# ── Text exposition format ─────────────────────────────────────────────────────

class TestTextExposition:
    def test_contains_help_lines(self) -> None:
        c = MLMetricsCollector()
        c.record_prediction(10.0)
        text = c.format_text_exposition()
        assert "# HELP" in text
        assert "# TYPE" in text

    def test_counter_type_declared(self) -> None:
        c = MLMetricsCollector()
        text = c.format_text_exposition()
        assert "# TYPE mlops_prediction_requests_total counter" in text

    def test_drift_labels_in_exposition(self) -> None:
        c = MLMetricsCollector()
        c.record_drift("pay_ratio", 0.31, "high")
        text = c.format_text_exposition()
        assert 'feature="pay_ratio"' in text
        assert 'severity="high"' in text

    def test_ends_with_newline(self) -> None:
        c = MLMetricsCollector()
        assert c.format_text_exposition().endswith("\n")

    def test_custom_prefix_in_names(self) -> None:
        c = MLMetricsCollector(prefix="credit_risk")
        text = c.format_text_exposition()
        assert "credit_risk_prediction_requests_total" in text
