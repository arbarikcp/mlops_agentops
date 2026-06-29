"""Tests for monitoring/drift.py — DriftDetector, FeatureDriftResult, DriftReport."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    DriftDetector,
    DriftMetric,
    DriftReport,
    FeatureDriftResult,
)


def _ref_cur(shift: float = 0.0, n: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    ref = rng.normal(0.5, 0.1, n)
    cur = rng.normal(0.5 + shift, 0.1, n)
    return ref, cur


# ── FeatureDriftResult ─────────────────────────────────────────────────────────

class TestFeatureDriftResult:
    def test_is_high(self) -> None:
        r = FeatureDriftResult("f", DriftMetric.PSI, 0.25, 0.20, True, "HIGH")
        assert r.is_high

    def test_is_not_high(self) -> None:
        r = FeatureDriftResult("f", DriftMetric.PSI, 0.05, 0.20, False, "NONE")
        assert not r.is_high


# ── DriftReport ────────────────────────────────────────────────────────────────

class TestDriftReport:
    def test_drifted_features_unique(self) -> None:
        results = [
            FeatureDriftResult("f1", DriftMetric.PSI, 0.25, 0.20, True, "HIGH"),
            FeatureDriftResult("f1", DriftMetric.KS, 0.12, 0.10, True, "HIGH"),
            FeatureDriftResult("f2", DriftMetric.PSI, 0.05, 0.20, False, "NONE"),
        ]
        report = DriftReport(results=results, overall_drifted=True)
        assert report.drifted_features() == ["f1"]

    def test_severity_counts(self) -> None:
        results = [
            FeatureDriftResult("a", DriftMetric.PSI, 0.25, 0.20, True, "HIGH"),
            FeatureDriftResult("b", DriftMetric.KS, 0.07, 0.10, True, "LOW"),
            FeatureDriftResult("c", DriftMetric.MMD, 0.02, 0.10, False, "NONE"),
        ]
        report = DriftReport(results=results)
        counts = report.severity_counts()
        assert counts["HIGH"] == 1
        assert counts["LOW"] == 1
        assert counts["NONE"] == 1

    def test_by_metric(self) -> None:
        results = [
            FeatureDriftResult("f", DriftMetric.PSI, 0.05, 0.20, False, "NONE"),
            FeatureDriftResult("f", DriftMetric.KS, 0.12, 0.10, True, "HIGH"),
        ]
        report = DriftReport(results=results)
        psi_only = report.by_metric(DriftMetric.PSI)
        assert len(psi_only) == 1
        assert psi_only[0].metric == DriftMetric.PSI

    def test_summary_stable(self) -> None:
        report = DriftReport()
        assert "STABLE" in report.summary()

    def test_summary_drifted(self) -> None:
        r = FeatureDriftResult("f", DriftMetric.PSI, 0.25, 0.20, True, "HIGH")
        report = DriftReport(results=[r], overall_drifted=True)
        assert "DRIFTED" in report.summary()


# ── DriftDetector — PSI ───────────────────────────────────────────────────────

class TestComputePSI:
    def test_zero_for_identical(self) -> None:
        ref, _ = _ref_cur()
        d = DriftDetector()
        assert d.compute_psi(ref, ref) == pytest.approx(0.0, abs=1e-6)

    def test_high_for_large_shift(self) -> None:
        ref, cur = _ref_cur(shift=0.5)
        d = DriftDetector()
        assert d.compute_psi(ref, cur) > 0.20

    def test_zero_for_small_arrays(self) -> None:
        d = DriftDetector()
        assert d.compute_psi(np.array([0.1, 0.2]), np.array([0.1, 0.2])) == 0.0


# ── DriftDetector — KS ────────────────────────────────────────────────────────

class TestComputeKS:
    def test_zero_for_identical(self) -> None:
        ref, _ = _ref_cur()
        d = DriftDetector()
        assert d.compute_ks(ref, ref) == pytest.approx(0.0, abs=1e-6)

    def test_high_for_large_shift(self) -> None:
        ref, cur = _ref_cur(shift=0.5)
        d = DriftDetector()
        assert d.compute_ks(ref, cur) > 0.10

    def test_zero_for_small_arrays(self) -> None:
        d = DriftDetector()
        assert d.compute_ks(np.array([0.1]), np.array([0.1])) == 0.0


# ── DriftDetector — MMD ───────────────────────────────────────────────────────

class TestComputeMMD:
    def test_near_zero_for_identical(self) -> None:
        ref, _ = _ref_cur(n=100)
        d = DriftDetector()
        assert d.compute_mmd(ref, ref) < 0.05

    def test_positive_for_shifted(self) -> None:
        ref, cur = _ref_cur(shift=0.5, n=100)
        d = DriftDetector()
        assert d.compute_mmd(ref, cur) > 0.05

    def test_zero_for_small_arrays(self) -> None:
        d = DriftDetector()
        assert d.compute_mmd(np.array([0.1, 0.2]), np.array([0.1, 0.2])) == 0.0


# ── DriftDetector — Classifier AUC ────────────────────────────────────────────

class TestComputeClassifierAUC:
    def test_near_half_for_identical(self) -> None:
        ref, _ = _ref_cur(n=200)
        d = DriftDetector()
        auc = d.compute_classifier_auc(ref, ref)
        assert 0.4 <= auc <= 0.6

    def test_high_for_large_shift(self) -> None:
        ref, cur = _ref_cur(shift=0.8, n=200)
        d = DriftDetector()
        auc = d.compute_classifier_auc(ref, cur)
        assert auc > 0.65

    def test_returns_half_for_small_arrays(self) -> None:
        d = DriftDetector()
        assert d.compute_classifier_auc(np.array([0.1] * 5), np.array([0.2] * 5)) == 0.5


# ── DriftDetector — check_feature ─────────────────────────────────────────────

class TestCheckFeature:
    def test_returns_four_results(self) -> None:
        ref, cur = _ref_cur()
        d = DriftDetector()
        results = d.check_feature("f", ref, cur)
        assert len(results) == 4

    def test_all_metrics_present(self) -> None:
        ref, cur = _ref_cur()
        d = DriftDetector()
        results = d.check_feature("f", ref, cur)
        metrics = {r.metric for r in results}
        assert metrics == {DriftMetric.PSI, DriftMetric.KS, DriftMetric.MMD, DriftMetric.CLASSIFIER}

    def test_feature_name_preserved(self) -> None:
        ref, cur = _ref_cur()
        d = DriftDetector()
        results = d.check_feature("pay_ratio", ref, cur)
        assert all(r.feature_name == "pay_ratio" for r in results)


# ── DriftDetector — run ───────────────────────────────────────────────────────

class TestDriftDetectorRun:
    def _make_dfs(self, shift: float = 0.0, n: int = 300):
        rng = np.random.default_rng(7)
        cols = ["pay_ratio", "util_rate"]
        ref = pd.DataFrame({c: rng.normal(0.5, 0.1, n) for c in cols})
        cur = pd.DataFrame({c: rng.normal(0.5 + shift, 0.1, n) for c in cols})
        return ref, cur

    def test_stable_on_identical_data(self) -> None:
        ref, _ = self._make_dfs()
        d = DriftDetector()
        report = d.run(ref, ref)
        assert not report.overall_drifted

    def test_drifted_on_large_shift(self) -> None:
        ref, cur = self._make_dfs(shift=0.6)
        d = DriftDetector()
        report = d.run(ref, cur)
        assert report.overall_drifted

    def test_results_count(self) -> None:
        ref, cur = self._make_dfs()
        d = DriftDetector()
        report = d.run(ref, cur, feature_names=["pay_ratio"])
        assert len(report.results) == 4  # 4 metrics × 1 feature

    def test_skips_non_numeric(self) -> None:
        ref = pd.DataFrame({"cat": ["a", "b"], "num": [1.0, 2.0]})
        cur = pd.DataFrame({"cat": ["x", "y"], "num": [1.1, 2.1]})
        d = DriftDetector()
        report = d.run(ref, cur)
        names = {r.feature_name for r in report.results}
        assert "cat" not in names

    def test_custom_feature_names(self) -> None:
        ref, cur = self._make_dfs()
        d = DriftDetector()
        report = d.run(ref, cur, feature_names=["pay_ratio"])
        names = {r.feature_name for r in report.results}
        assert names == {"pay_ratio"}
