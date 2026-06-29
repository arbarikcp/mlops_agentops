"""Tests for features/skew_checker.py — SkewEvidence, TrainServeSkewReport, TrainServeSkewChecker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.skew_checker import (
    SkewEvidence,
    SkewType,
    TrainServeSkewChecker,
    TrainServeSkewReport,
)


def _make_df(
    n: int = 500,
    shift: float = 0.0,
    cols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = cols or ["pay_ratio", "util_rate"]
    return pd.DataFrame({c: rng.normal(0.5 + shift, 0.1, n).clip(0, 1) for c in cols})


# ── SkewEvidence ───────────────────────────────────────────────────────────────

class TestSkewEvidence:
    def test_is_high(self) -> None:
        e = SkewEvidence("f", SkewType.FEATURE, 0.5, 0.8, 0.3, "HIGH")
        assert e.is_high

    def test_is_not_high(self) -> None:
        e = SkewEvidence("f", SkewType.FEATURE, 0.5, 0.55, 0.05, "LOW")
        assert not e.is_high


# ── TrainServeSkewReport ───────────────────────────────────────────────────────

class TestTrainServeSkewReport:
    def test_zero_skew_default(self) -> None:
        report = TrainServeSkewReport()
        assert report.zero_skew

    def test_high_severity_count(self) -> None:
        e1 = SkewEvidence("a", SkewType.FEATURE, 0.5, 0.9, 0.4, "HIGH")
        e2 = SkewEvidence("b", SkewType.FEATURE, 0.5, 0.55, 0.05, "LOW")
        report = TrainServeSkewReport(evidences=[e1, e2], zero_skew=False)
        assert report.high_severity_count() == 1

    def test_by_type(self) -> None:
        e1 = SkewEvidence("a", SkewType.SCHEMA, 1.0, 0.0, 1.0, "HIGH")
        e2 = SkewEvidence("b", SkewType.FEATURE, 0.5, 0.9, 0.4, "HIGH")
        report = TrainServeSkewReport(evidences=[e1, e2])
        assert len(report.by_type(SkewType.SCHEMA)) == 1
        assert len(report.by_type(SkewType.FEATURE)) == 1

    def test_summary_contains_status(self) -> None:
        report = TrainServeSkewReport(zero_skew=True)
        assert "ZERO SKEW" in report.summary()

    def test_summary_contains_failed_status(self) -> None:
        e = SkewEvidence("x", SkewType.FEATURE, 0.0, 1.0, 1.0, "HIGH")
        report = TrainServeSkewReport(evidences=[e], zero_skew=False)
        assert "SKEW DETECTED" in report.summary()


# ── TrainServeSkewChecker — Schema Skew ───────────────────────────────────────

class TestSchemaSkew:
    def test_no_skew_identical_schemas(self) -> None:
        checker = TrainServeSkewChecker()
        df = _make_df()
        evidences = checker.check_schema_skew(df, df)
        assert evidences == []

    def test_detects_missing_column_at_serving(self) -> None:
        checker = TrainServeSkewChecker()
        train_df = _make_df(cols=["pay_ratio", "util_rate"])
        serve_df = _make_df(cols=["pay_ratio"])  # util_rate missing
        evidences = checker.check_schema_skew(train_df, serve_df)
        names = [e.feature_name for e in evidences]
        assert "util_rate" in names
        assert any(e.severity == "HIGH" for e in evidences)

    def test_extra_column_at_serving_is_low(self) -> None:
        checker = TrainServeSkewChecker()
        train_df = _make_df(cols=["pay_ratio"])
        serve_df = _make_df(cols=["pay_ratio", "extra_col"])
        evidences = checker.check_schema_skew(train_df, serve_df)
        assert any(e.feature_name == "extra_col" and e.severity == "LOW" for e in evidences)


# ── TrainServeSkewChecker — Feature Skew ──────────────────────────────────────

class TestFeatureSkew:
    def test_no_skew_identical_data(self) -> None:
        checker = TrainServeSkewChecker()
        df = _make_df(500)
        evidences = checker.check_feature_skew(df, df)
        assert all(e.severity == "NONE" or e.severity == "LOW" for e in evidences)
        assert not any(e.severity == "HIGH" for e in evidences)

    def test_high_skew_detected_on_shifted_data(self) -> None:
        checker = TrainServeSkewChecker()
        train_df = _make_df(500, shift=0.0)
        serve_df = _make_df(500, shift=0.5)  # large shift
        evidences = checker.check_feature_skew(train_df, serve_df, ["pay_ratio"])
        assert any(e.severity == "HIGH" for e in evidences)

    def test_skips_non_numeric_columns(self) -> None:
        checker = TrainServeSkewChecker()
        train_df = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1.0, 2.0, 3.0]})
        serve_df = pd.DataFrame({"cat": ["x", "y", "z"], "num": [1.1, 2.1, 3.1]})
        evidences = checker.check_feature_skew(train_df, serve_df)
        names = [e.feature_name for e in evidences]
        assert "cat" not in names

    def test_psi_zero_for_identical(self) -> None:
        checker = TrainServeSkewChecker()
        df = _make_df(500)
        evidences = checker.check_feature_skew(df, df, ["pay_ratio"])
        assert evidences == []  # PSI=0 → NONE severity → no evidence emitted

    def test_custom_feature_names(self) -> None:
        checker = TrainServeSkewChecker()
        df = _make_df(cols=["pay_ratio", "util_rate"])
        evidences = checker.check_feature_skew(df, df, ["pay_ratio"])
        # only pay_ratio checked
        assert all(e.feature_name == "pay_ratio" for e in evidences)


# ── TrainServeSkewChecker — Prediction Skew ───────────────────────────────────

class TestPredictionSkew:
    def test_no_skew_identical_scores(self) -> None:
        checker = TrainServeSkewChecker(pred_delta_threshold=1e-4)
        scores = [0.1, 0.5, 0.9, 0.3]
        evidences = checker.check_prediction_skew(scores, scores)
        assert evidences == []

    def test_detects_large_score_delta(self) -> None:
        checker = TrainServeSkewChecker(pred_delta_threshold=1e-4)
        train_scores = [0.5, 0.6, 0.7]
        serve_scores = [0.5, 0.6, 0.9]  # last differs by 0.2
        evidences = checker.check_prediction_skew(train_scores, serve_scores)
        assert len(evidences) == 1
        assert evidences[0].severity in ("LOW", "HIGH")

    def test_length_mismatch_returns_high(self) -> None:
        checker = TrainServeSkewChecker()
        evidences = checker.check_prediction_skew([0.5, 0.6], [0.5])
        assert evidences[0].severity == "HIGH"

    def test_empty_scores_returns_empty(self) -> None:
        checker = TrainServeSkewChecker()
        assert checker.check_prediction_skew([], []) == []


# ── TrainServeSkewChecker — Full Run ──────────────────────────────────────────

class TestSkewCheckerRun:
    def test_zero_skew_on_same_data(self) -> None:
        checker = TrainServeSkewChecker()
        df = _make_df(500)
        report = checker.run(df, df)
        assert report.zero_skew

    def test_not_zero_skew_on_missing_column(self) -> None:
        checker = TrainServeSkewChecker()
        train_df = _make_df(cols=["pay_ratio", "util_rate"])
        serve_df = _make_df(cols=["pay_ratio"])
        report = checker.run(train_df, serve_df)
        assert not report.zero_skew

    def test_not_zero_skew_on_large_feature_shift(self) -> None:
        checker = TrainServeSkewChecker()
        train_df = _make_df(500, shift=0.0)
        serve_df = _make_df(500, shift=0.6)
        report = checker.run(train_df, serve_df, feature_names=["pay_ratio"])
        assert not report.zero_skew

    def test_prediction_skew_included_in_report(self) -> None:
        checker = TrainServeSkewChecker(pred_delta_threshold=1e-4)
        df = _make_df(100)
        train_scores = [0.5] * 10
        serve_scores = [0.8] * 10  # large delta
        report = checker.run(df, df, train_scores=train_scores, serve_scores=serve_scores)
        assert any(e.skew_type == SkewType.PREDICTION for e in report.evidences)

    def test_evidence_list_empty_for_zero_skew(self) -> None:
        checker = TrainServeSkewChecker()
        df = _make_df(500)
        report = checker.run(df, df)
        assert report.high_severity_count() == 0
