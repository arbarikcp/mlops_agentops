"""Tests for training/slice_eval.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.slice_eval import (
    evaluate_slices,
    fit_ood_detector,
    ood_report,
    slice_gap_report,
    worst_slices,
)


def _make_data(n: int = 1_000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build a synthetic dataset with slice columns and some class imbalance."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "EDUCATION": rng.choice([1, 2, 3, 4], size=n),
        "SEX": rng.choice([1, 2], size=n),
        "MARRIAGE": rng.choice([1, 2, 3], size=n),
        "numeric_feat": rng.standard_normal(n),
    })
    y_prob = np.clip(rng.beta(2, 5, n), 0.01, 0.99)
    y_true = (y_prob + rng.normal(0, 0.15, n) > 0.4).astype(int)
    return X, y_true, y_prob


class TestEvaluateSlices:
    def test_returns_dataframe(self):
        X, y_true, y_prob = _make_data()
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["SEX"])
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns_present(self):
        X, y_true, y_prob = _make_data()
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["SEX"])
        required = {"slice_col", "slice_val", "n", "roc_auc", "average_precision", "calibration_error"}
        assert required.issubset(df.columns)

    def test_metrics_in_valid_range(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["SEX", "EDUCATION"])
        assert (df["roc_auc"].between(0.0, 1.0)).all()
        assert (df["average_precision"].between(0.0, 1.0)).all()
        assert (df["calibration_error"] >= 0.0).all()

    def test_positive_rate_in_range(self):
        X, y_true, y_prob = _make_data()
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["SEX"])
        assert (df["positive_rate"].between(0.0, 1.0)).all()

    def test_skips_missing_column(self):
        X, y_true, y_prob = _make_data()
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["NONEXISTENT"])
        assert len(df) == 0

    def test_skips_tiny_slices(self):
        X, y_true, y_prob = _make_data(n=100)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["EDUCATION"], min_size=200)
        assert len(df) == 0

    def test_default_columns_used_if_none(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob)
        assert len(df) > 0  # EDUCATION, SEX, MARRIAGE all present in X

    def test_n_matches_actual_slice_size(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["SEX"])
        for _, row in df.iterrows():
            col, val = row["slice_col"], row["slice_val"]
            actual_n = (X[col].astype(str) == val).sum()
            assert int(row["n"]) == int(actual_n)


class TestWorstSlices:
    def test_returns_at_most_n_rows(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["EDUCATION", "SEX"])
        worst = worst_slices(df, n=3)
        assert len(worst) <= 3

    def test_worst_has_lowest_metric(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["EDUCATION", "SEX"])
        worst = worst_slices(df, metric="roc_auc", n=3)
        if len(df) > 3:
            assert float(worst["roc_auc"].max()) <= float(df["roc_auc"].median())

    def test_returns_subset_of_columns(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["SEX"])
        worst = worst_slices(df, n=2)
        assert "slice_col" in worst.columns
        assert "slice_val" in worst.columns


class TestSliceGapReport:
    def test_flag_large_gap(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["EDUCATION", "SEX"])
        # Setting overall = 1.0 (perfect) guarantees every real slice has gap > 0.05
        overall = {"roc_auc": 1.0}
        gaps = slice_gap_report(df, overall, warn_threshold=0.05)
        assert gaps["flag"].any()

    def test_no_flag_small_gap(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["SEX"])
        overall = {"roc_auc": 0.0}  # artificially low — all slices will be above
        gaps = slice_gap_report(df, overall, warn_threshold=0.05)
        assert not gaps["flag"].any()

    def test_sorted_by_gap_descending(self):
        X, y_true, y_prob = _make_data(n=2000)
        df = evaluate_slices(X, y_true, y_prob, slice_cols=["EDUCATION"])
        overall = {"roc_auc": 0.80}
        gaps = slice_gap_report(df, overall)
        if len(gaps) > 1:
            assert gaps["gap"].iloc[0] >= gaps["gap"].iloc[-1]


class TestOodDetector:
    def test_fit_and_score_shape(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((500, 5))
        X_test = rng.standard_normal((100, 5))
        detector = fit_ood_detector(X_train)
        scores = detector.decision_function(X_test)
        assert scores.shape == (100,)

    def test_ood_report_keys(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((500, 5))
        X_test = rng.standard_normal((100, 5))
        detector = fit_ood_detector(X_train)
        report = ood_report(detector, X_test)
        assert set(report.keys()) == {"ood_fraction", "mean_score", "p5_score", "n_samples"}

    def test_ood_fraction_in_range(self):
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((500, 5))
        X_test = rng.standard_normal((100, 5))
        detector = fit_ood_detector(X_train)
        report = ood_report(detector, X_test)
        assert 0.0 <= report["ood_fraction"] <= 1.0

    def test_ood_samples_score_lower_than_in_dist(self):
        """OOD samples (completely different distribution) must score lower."""
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((500, 5))
        X_in = rng.standard_normal((200, 5))
        X_ood = rng.standard_normal((200, 5)) * 5 + 20  # very different
        detector = fit_ood_detector(X_train)
        score_in = float(detector.decision_function(X_in).mean())
        score_ood = float(detector.decision_function(X_ood).mean())
        assert score_ood < score_in

    def test_n_samples_correct(self):
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((300, 4))
        X_test = rng.standard_normal((77, 4))
        detector = fit_ood_detector(X_train)
        report = ood_report(detector, X_test)
        assert report["n_samples"] == 77
