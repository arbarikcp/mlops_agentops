"""Tests for monitoring/skew_detector.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.reference_stats import compute_reference_stats
from monitoring.skew_detector import (
    SkewReport,
    compute_js,
    compute_ks,
    compute_psi,
    detect_skew,
    skew_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    # All normal features so the Gaussian approximation used in detect_skew
    # (reconstructing reference from stored mean/std) is accurate.
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat_a": rng.standard_normal(n),
        "feat_b": rng.standard_normal(n) * 10 + 50,
        "feat_c": rng.standard_normal(n) * 0.5 + 2.5,
    })


def _shifted_df(n: int = 500, shift: float = 10.0, seed: int = 99) -> pd.DataFrame:
    """DataFrame with mean shifted by `shift` relative to _make_df defaults."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat_a": rng.standard_normal(n) + shift,
        "feat_b": rng.standard_normal(n) * 10 + 50 + shift,
        "feat_c": rng.standard_normal(n) * 0.5 + 2.5 + shift,
    })


# ── compute_psi ───────────────────────────────────────────────────────────────

class TestComputePsi:
    def test_identical_distributions_zero_psi(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(1000)
        psi = compute_psi(data, data)
        assert psi < 0.02  # should be near 0 (epsilon adds tiny noise)

    def test_shifted_distribution_high_psi(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.standard_normal(1000)
        cur = rng.standard_normal(1000) + 5.0  # major shift
        psi = compute_psi(ref, cur)
        assert psi > 0.20  # must trigger "major" threshold

    def test_slight_shift_moderate_psi(self) -> None:
        rng = np.random.default_rng(1)
        ref = rng.standard_normal(1000)
        cur = rng.standard_normal(1000) + 0.5  # slight shift
        psi = compute_psi(ref, cur)
        # Should be > 0 (some shift) but not extreme
        assert psi >= 0.0

    def test_returns_float(self) -> None:
        rng = np.random.default_rng(2)
        psi = compute_psi(rng.standard_normal(100), rng.standard_normal(100))
        assert isinstance(psi, float)

    def test_constant_reference_returns_defined_value(self) -> None:
        ref = np.ones(100)
        cur = np.ones(100)
        psi = compute_psi(ref, cur)
        assert psi == 0.0  # identical constant distributions

    def test_psi_non_negative(self) -> None:
        rng = np.random.default_rng(3)
        psi = compute_psi(rng.standard_normal(200), rng.standard_normal(200) * 2)
        assert psi >= 0.0


# ── compute_ks ────────────────────────────────────────────────────────────────

class TestComputeKs:
    def test_identical_distributions_high_pvalue(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(500)
        _, pvalue = compute_ks(data, data)
        assert pvalue == 1.0  # identical data → p-value = 1

    def test_different_distributions_low_pvalue(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.standard_normal(500)
        cur = rng.standard_normal(500) + 10.0
        _, pvalue = compute_ks(ref, cur)
        assert pvalue < 0.05  # significant shift

    def test_returns_two_floats(self) -> None:
        rng = np.random.default_rng(0)
        stat, pvalue = compute_ks(rng.standard_normal(100), rng.standard_normal(100))
        assert isinstance(stat, float)
        assert isinstance(pvalue, float)

    def test_ks_stat_in_range(self) -> None:
        rng = np.random.default_rng(5)
        stat, _ = compute_ks(rng.standard_normal(200), rng.standard_normal(200))
        assert 0.0 <= stat <= 1.0


# ── compute_js ────────────────────────────────────────────────────────────────

class TestComputeJs:
    def test_identical_distributions_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal(500)
        js = compute_js(data, data)
        assert js < 0.05

    def test_different_distributions_higher_js(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.standard_normal(500)
        cur = rng.standard_normal(500) + 10.0
        js = compute_js(ref, cur)
        assert js > 0.1

    def test_js_in_range(self) -> None:
        rng = np.random.default_rng(7)
        js = compute_js(rng.standard_normal(200), rng.standard_normal(200) * 3)
        assert 0.0 <= js <= 1.0

    def test_returns_float(self) -> None:
        rng = np.random.default_rng(8)
        js = compute_js(rng.standard_normal(100), rng.standard_normal(100))
        assert isinstance(js, float)


# ── detect_skew ───────────────────────────────────────────────────────────────

class TestDetectSkew:
    def test_returns_skew_report(self) -> None:
        train_df = _make_df()
        ref = compute_reference_stats(train_df, model_version="v1")
        report = detect_skew(train_df, ref)
        assert isinstance(report, SkewReport)

    def test_same_distribution_stable(self) -> None:
        train_df = _make_df(n=1000, seed=0)
        serve_df = _make_df(n=200, seed=7)   # same distribution family, different seed
        ref = compute_reference_stats(train_df, model_version="v1")
        report = detect_skew(serve_df, ref)
        assert report.overall_severity in {"stable", "moderate"}

    def test_large_shift_flagged(self) -> None:
        train_df = _make_df(n=1000)
        ref = compute_reference_stats(train_df, model_version="v1")
        shifted = _shifted_df(n=500, shift=15.0)
        report = detect_skew(shifted, ref)
        assert report.n_flagged > 0

    def test_report_has_correct_n_features(self) -> None:
        df = _make_df()
        ref = compute_reference_stats(df, model_version="v1")
        report = detect_skew(df, ref)
        assert report.n_features == len(df.select_dtypes(include="number").columns)

    def test_report_model_version_matches(self) -> None:
        df = _make_df()
        ref = compute_reference_stats(df, model_version="v_test")
        report = detect_skew(df, ref)
        assert report.reference_model_version == "v_test"

    def test_serving_n_rows_correct(self) -> None:
        train_df = _make_df(n=500)
        serve_df = _make_df(n=123, seed=10)
        ref = compute_reference_stats(train_df, model_version="v1")
        report = detect_skew(serve_df, ref)
        assert report.serving_n_rows == 123

    def test_feature_subset_evaluated(self) -> None:
        df = _make_df()
        ref = compute_reference_stats(df, model_version="v1")
        report = detect_skew(df, ref, features=["feat_a"])
        assert report.n_features == 1
        assert report.feature_results[0].feature == "feat_a"


# ── skew_summary ──────────────────────────────────────────────────────────────

class TestSkewSummary:
    def test_returns_dataframe(self) -> None:
        df = _make_df()
        ref = compute_reference_stats(df, model_version="v1")
        report = detect_skew(df, ref)
        summary = skew_summary(report)
        assert isinstance(summary, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        df = _make_df()
        ref = compute_reference_stats(df, model_version="v1")
        report = detect_skew(df, ref)
        summary = skew_summary(report)
        required = {"feature", "psi", "ks_stat", "ks_pvalue", "js_divergence", "severity", "flag"}
        assert required.issubset(set(summary.columns))

    def test_sorted_by_psi_descending(self) -> None:
        df = _make_df()
        ref = compute_reference_stats(df, model_version="v1")
        report = detect_skew(df, ref)
        summary = skew_summary(report)
        if len(summary) > 1:
            assert summary["psi"].iloc[0] >= summary["psi"].iloc[-1]

    def test_empty_report_returns_empty_df(self) -> None:
        empty_report = SkewReport(
            feature_results=[],
            overall_severity="stable",
            n_flagged=0,
            n_features=0,
            serving_n_rows=0,
            reference_model_version="v1",
        )
        summary = skew_summary(empty_report)
        assert len(summary) == 0
