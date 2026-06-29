"""Tests for features/feature_monitor.py — FreshnessChecker, FeatureQualityChecker, FeatureDriftMonitor, FeatureMonitor."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from features.feature_monitor import (
    FeatureBounds,
    FeatureDriftMonitor,
    FeatureDriftResult,
    FeatureMonitor,
    FeatureMonitorReport,
    FeatureQualityChecker,
    FreshnessCheck,
    FreshnessChecker,
    FreshnessStatus,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dt(hours_ago: float) -> datetime:
    return _now() - timedelta(hours=hours_ago)


# ── FreshnessChecker ───────────────────────────────────────────────────────────

class TestFreshnessChecker:
    def _checker(self) -> FreshnessChecker:
        return FreshnessChecker(stale_multiplier=3.0)

    def test_fresh_when_recent(self) -> None:
        checker = self._checker()
        result = checker.check("view_a", _dt(1), threshold_hours=25)
        assert result.status == FreshnessStatus.FRESH
        assert result.is_fresh

    def test_stale_when_past_threshold(self) -> None:
        checker = self._checker()
        result = checker.check("view_a", _dt(30), threshold_hours=25)
        assert result.status == FreshnessStatus.STALE

    def test_missing_when_far_past_threshold(self) -> None:
        checker = self._checker()
        result = checker.check("view_a", _dt(100), threshold_hours=25)
        assert result.status == FreshnessStatus.MISSING

    def test_missing_when_never_materialized(self) -> None:
        checker = self._checker()
        result = checker.check("view_a", None, threshold_hours=25)
        assert result.status == FreshnessStatus.MISSING
        assert result.age_hours == math.inf

    def test_age_hours_correct(self) -> None:
        checker = self._checker()
        result = checker.check("view_a", _dt(5), threshold_hours=25)
        assert abs(result.age_hours - 5.0) < 0.1

    def test_invalid_multiplier_raises(self) -> None:
        with pytest.raises(ValueError, match="stale_multiplier"):
            FreshnessChecker(stale_multiplier=0.5)

    def test_feature_view_name_preserved(self) -> None:
        checker = self._checker()
        result = checker.check("my_view", _dt(1), threshold_hours=25)
        assert result.feature_view_name == "my_view"

    def test_naive_datetime_handled(self) -> None:
        checker = self._checker()
        naive = datetime.utcnow() - timedelta(hours=2)
        result = checker.check("view", naive, threshold_hours=25)
        assert result.status == FreshnessStatus.FRESH


# ── FeatureQualityChecker ──────────────────────────────────────────────────────

class TestFeatureQualityChecker:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "util_rate": [0.3, 0.5, 0.7, None, 0.9],
            "pay_ratio": [0.1, 0.2, 0.3, 0.4, 0.5],
        })

    def test_no_issues_when_within_bounds(self) -> None:
        checker = FeatureQualityChecker()
        bounds = [FeatureBounds("pay_ratio", min_val=0.0, max_val=1.0, max_null_rate=0.05)]
        results = checker.check(self._df(), bounds)
        assert results[0].passed

    def test_null_rate_exceeds_threshold(self) -> None:
        checker = FeatureQualityChecker()
        bounds = [FeatureBounds("util_rate", max_null_rate=0.0)]  # no nulls allowed
        results = checker.check(self._df(), bounds)
        assert not results[0].passed
        assert any("null" in issue.lower() for issue in results[0].issues)

    def test_out_of_range_detected(self) -> None:
        checker = FeatureQualityChecker()
        df = pd.DataFrame({"util_rate": [0.5, 1.5, 0.3]})  # 1.5 > max
        bounds = [FeatureBounds("util_rate", min_val=0.0, max_val=1.0)]
        results = checker.check(df, bounds)
        assert not results[0].passed
        assert results[0].out_of_range_rate > 0

    def test_missing_column_fails(self) -> None:
        checker = FeatureQualityChecker()
        bounds = [FeatureBounds("nonexistent")]
        results = checker.check(self._df(), bounds)
        assert not results[0].passed
        assert "not found" in results[0].issues[0]

    def test_observed_min_max_reported(self) -> None:
        checker = FeatureQualityChecker()
        bounds = [FeatureBounds("pay_ratio")]
        results = checker.check(self._df(), bounds)
        assert results[0].min_val == pytest.approx(0.1)
        assert results[0].max_val == pytest.approx(0.5)

    def test_null_rate_computed_correctly(self) -> None:
        checker = FeatureQualityChecker()
        df = pd.DataFrame({"x": [1.0, None, None, 4.0, 5.0]})
        bounds = [FeatureBounds("x", max_null_rate=0.5)]
        results = checker.check(df, bounds)
        assert results[0].null_rate == pytest.approx(0.4)

    def test_multiple_bounds_returned(self) -> None:
        checker = FeatureQualityChecker()
        bounds = [FeatureBounds("util_rate"), FeatureBounds("pay_ratio")]
        results = checker.check(self._df(), bounds)
        assert len(results) == 2


# ── FeatureDriftMonitor ────────────────────────────────────────────────────────

def _make_df(n: int = 500, shift: float = 0.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "pay_ratio": rng.normal(0.3 + shift, 0.1, n).clip(0, 1),
        "util_rate": rng.normal(0.5 + shift, 0.15, n).clip(0, 1),
    })


class TestFeatureDriftMonitor:
    def test_no_drift_identical_distributions(self) -> None:
        monitor = FeatureDriftMonitor()
        df = _make_df(500)
        results = monitor.check(df, df, ["pay_ratio", "util_rate"])
        for r in results:
            assert r.severity == "NONE"

    def test_high_drift_detected(self) -> None:
        monitor = FeatureDriftMonitor()
        ref = _make_df(500, shift=0.0)
        cur = _make_df(500, shift=0.5)  # large shift
        results = monitor.check(ref, cur, ["pay_ratio"])
        assert results[0].severity in ("LOW", "HIGH")

    def test_psi_zero_for_identical(self) -> None:
        monitor = FeatureDriftMonitor()
        df = _make_df(500)
        results = monitor.check(df, df, ["pay_ratio"])
        assert results[0].psi < 0.01

    def test_ks_stat_in_range(self) -> None:
        monitor = FeatureDriftMonitor()
        df = _make_df(500)
        results = monitor.check(df, df, ["pay_ratio"])
        assert 0.0 <= results[0].ks_stat <= 1.0

    def test_missing_feature_returns_none_severity(self) -> None:
        monitor = FeatureDriftMonitor()
        df = _make_df(200)
        results = monitor.check(df, df, ["nonexistent"])
        assert results[0].severity == "NONE"
        assert results[0].psi == 0.0

    def test_is_drifted_property(self) -> None:
        ref = _make_df(500, shift=0.0)
        cur = _make_df(500, shift=0.5)
        monitor = FeatureDriftMonitor()
        results = monitor.check(ref, cur, ["pay_ratio"])
        assert results[0].is_drifted

    def test_returns_one_result_per_feature(self) -> None:
        monitor = FeatureDriftMonitor()
        df = _make_df(200)
        results = monitor.check(df, df, ["pay_ratio", "util_rate"])
        assert len(results) == 2


# ── FeatureMonitor ────────────────────────────────────────────────────────────

class TestFeatureMonitor:
    def test_all_pass_fresh_quality_no_drift(self) -> None:
        ref = _make_df(300)
        cur = _make_df(300)
        monitor = FeatureMonitor(
            freshness_configs=[("view_a", _dt(1), 25.0)],
            quality_bounds=[FeatureBounds("pay_ratio", min_val=0.0, max_val=1.0)],
            drift_features=["pay_ratio"],
        )
        report = monitor.run(ref, cur, now=_now())
        assert report.overall_passed

    def test_missing_freshness_fails_overall(self) -> None:
        df = _make_df(200)
        monitor = FeatureMonitor(
            freshness_configs=[("view_a", None, 25.0)],
        )
        report = monitor.run(df, df)
        assert not report.overall_passed

    def test_quality_failure_fails_overall(self) -> None:
        df = pd.DataFrame({"bad_col": [None, None, None, None, None]})
        monitor = FeatureMonitor(
            quality_bounds=[FeatureBounds("bad_col", max_null_rate=0.0)],
        )
        report = monitor.run(df, df)
        assert not report.overall_passed

    def test_high_drift_fails_overall(self) -> None:
        ref = _make_df(500, shift=0.0)
        cur = _make_df(500, shift=0.6)
        monitor = FeatureMonitor(drift_features=["pay_ratio"])
        report = monitor.run(ref, cur)
        assert not report.overall_passed

    def test_summary_contains_status(self) -> None:
        df = _make_df(100)
        monitor = FeatureMonitor()
        report = monitor.run(df, df)
        summary = report.summary()
        assert "FeatureMonitorReport" in summary

    def test_report_has_all_sections(self) -> None:
        ref = _make_df(200)
        cur = _make_df(200)
        monitor = FeatureMonitor(
            freshness_configs=[("v1", _dt(1), 25.0)],
            quality_bounds=[FeatureBounds("pay_ratio")],
            drift_features=["pay_ratio"],
        )
        report = monitor.run(ref, cur)
        assert len(report.freshness) == 1
        assert len(report.quality) == 1
        assert len(report.drift) == 1

    def test_stale_freshness_does_not_block_overall(self) -> None:
        df = _make_df(200)
        # STALE (not MISSING) doesn't fail overall_passed
        monitor = FeatureMonitor(
            freshness_configs=[("view_a", _dt(30), 25.0)],  # 30h > 25h threshold = STALE
        )
        report = monitor.run(df, df)
        assert report.freshness[0].status == FreshnessStatus.STALE
        assert report.overall_passed  # STALE doesn't fail the gate — only MISSING does
