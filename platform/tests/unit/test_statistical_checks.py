"""Tests for data/contracts/statistical_checks.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.contracts.statistical_checks import (
    DatasetStats,
    check_class_balance,
    check_mean_drift,
    check_null_drift,
    compute_dataset_stats,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 500, seed: int = 0, null_rate: float = 0.0, positive_rate: float = 0.22) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "feature_a": rng.standard_normal(n),
        "feature_b": rng.uniform(0, 100, n),
        "category": rng.choice(["x", "y", "z"], n),
        "DEFAULT_PAYMENT_NEXT_MONTH": (rng.random(n) < positive_rate).astype(int),
    })
    if null_rate > 0:
        null_mask = rng.random(n) < null_rate
        df.loc[null_mask, "feature_a"] = np.nan
    return df


# ── compute_dataset_stats ─────────────────────────────────────────────────────

class TestComputeDatasetStats:
    def test_returns_dataset_stats(self) -> None:
        df = _make_df()
        stats = compute_dataset_stats(df)
        assert isinstance(stats, DatasetStats)

    def test_n_rows_correct(self) -> None:
        df = _make_df(n=300)
        stats = compute_dataset_stats(df, dataset_name="test")
        assert stats.n_rows == 300

    def test_all_columns_in_stats(self) -> None:
        df = _make_df()
        stats = compute_dataset_stats(df)
        for col in df.columns:
            assert col in stats.columns

    def test_positive_rate_computed(self) -> None:
        df = _make_df(positive_rate=0.25)
        stats = compute_dataset_stats(df)
        assert stats.positive_rate is not None
        assert abs(stats.positive_rate - 0.25) < 0.05  # within 5% of target

    def test_null_rate_for_null_column(self) -> None:
        df = _make_df(n=1000, null_rate=0.10)
        stats = compute_dataset_stats(df)
        nr = stats.columns["feature_a"].null_rate
        assert abs(nr - 0.10) < 0.03

    def test_numeric_columns_have_mean(self) -> None:
        df = _make_df()
        stats = compute_dataset_stats(df)
        assert stats.columns["feature_a"].mean is not None
        assert stats.columns["feature_b"].mean is not None

    def test_categorical_column_no_mean(self) -> None:
        df = _make_df()
        stats = compute_dataset_stats(df)
        assert stats.columns["category"].mean is None

    def test_serialise_roundtrip(self) -> None:
        df = _make_df()
        stats = compute_dataset_stats(df, dataset_name="roundtrip")
        d = stats.to_dict()
        restored = DatasetStats.from_dict(d)
        assert restored.n_rows == stats.n_rows
        assert restored.dataset_name == stats.dataset_name
        assert abs(restored.columns["feature_a"].mean - stats.columns["feature_a"].mean) < 1e-10

    def test_positive_rate_none_when_no_target(self) -> None:
        df = _make_df().drop(columns=["DEFAULT_PAYMENT_NEXT_MONTH"])
        stats = compute_dataset_stats(df, target_col="DEFAULT_PAYMENT_NEXT_MONTH")
        assert stats.positive_rate is None


# ── check_null_drift ──────────────────────────────────────────────────────────

class TestCheckNullDrift:
    def test_no_drift_no_flags(self) -> None:
        df = _make_df(n=500)
        ref_stats = compute_dataset_stats(df, dataset_name="ref")
        cur_stats = compute_dataset_stats(df, dataset_name="cur")
        result = check_null_drift(cur_stats, ref_stats, threshold=0.05)
        assert isinstance(result, pd.DataFrame)
        assert not result["flag"].any()

    def test_introduced_null_drift_flagged(self) -> None:
        ref_df = _make_df(n=1000, null_rate=0.0)
        cur_df = _make_df(n=1000, null_rate=0.20)
        ref_stats = compute_dataset_stats(ref_df, dataset_name="ref")
        cur_stats = compute_dataset_stats(cur_df, dataset_name="cur")
        result = check_null_drift(cur_stats, ref_stats, threshold=0.05)
        # feature_a null_rate went from 0 to ~0.20 — must be flagged
        flagged = result[result["flag"]]
        assert "feature_a" in flagged["column"].values

    def test_result_has_expected_columns(self) -> None:
        df = _make_df()
        stats = compute_dataset_stats(df)
        result = check_null_drift(stats, stats)
        required = {"column", "current_null_rate", "reference_null_rate", "drift", "flag"}
        assert required.issubset(set(result.columns))

    def test_sorted_by_drift_descending(self) -> None:
        df = _make_df(n=500, null_rate=0.1)
        stats = compute_dataset_stats(df)
        result = check_null_drift(stats, stats)
        if len(result) > 1:
            assert result["drift"].iloc[0] >= result["drift"].iloc[-1]


# ── check_mean_drift ──────────────────────────────────────────────────────────

class TestCheckMeanDrift:
    def test_no_drift_no_flags(self) -> None:
        df = _make_df(n=500)
        stats = compute_dataset_stats(df)
        result = check_mean_drift(stats, stats, z_threshold=3.0)
        assert isinstance(result, pd.DataFrame)
        assert not result["flag"].any()

    def test_large_mean_shift_flagged(self) -> None:
        rng = np.random.default_rng(5)
        ref_df = pd.DataFrame({"x": rng.standard_normal(500), "DEFAULT_PAYMENT_NEXT_MONTH": np.zeros(500, int)})
        cur_df = pd.DataFrame({"x": rng.standard_normal(500) + 20.0, "DEFAULT_PAYMENT_NEXT_MONTH": np.zeros(500, int)})
        ref_stats = compute_dataset_stats(ref_df, dataset_name="ref")
        cur_stats = compute_dataset_stats(cur_df, dataset_name="cur")
        result = check_mean_drift(cur_stats, ref_stats, z_threshold=3.0)
        flagged = result[result["flag"]]
        assert "x" in flagged["column"].values

    def test_result_has_expected_columns(self) -> None:
        df = _make_df()
        stats = compute_dataset_stats(df)
        result = check_mean_drift(stats, stats)
        required = {"column", "current_mean", "reference_mean", "z_score", "severity", "flag"}
        assert required.issubset(set(result.columns))

    def test_severity_ok_for_no_drift(self) -> None:
        df = _make_df(n=500)
        stats = compute_dataset_stats(df)
        result = check_mean_drift(stats, stats)
        # Self-comparison: z=0, all "ok"
        assert (result["severity"] == "ok").all()

    def test_only_numeric_columns_included(self) -> None:
        df = _make_df()
        stats = compute_dataset_stats(df)
        result = check_mean_drift(stats, stats)
        # "category" is a string column — must not appear
        assert "category" not in result["column"].values


# ── check_class_balance ───────────────────────────────────────────────────────

class TestCheckClassBalance:
    def test_normal_balance_in_range(self) -> None:
        df = _make_df(n=1000, positive_rate=0.22)
        stats = compute_dataset_stats(df)
        result = check_class_balance(stats, min_rate=0.10, max_rate=0.40)
        assert result["in_range"] is True

    def test_zero_positives_out_of_range(self) -> None:
        df = _make_df(n=500, positive_rate=0.0)
        stats = compute_dataset_stats(df)
        result = check_class_balance(stats, min_rate=0.10, max_rate=0.40)
        assert result["in_range"] is False

    def test_all_positives_out_of_range(self) -> None:
        df = _make_df(n=500, positive_rate=1.0)
        stats = compute_dataset_stats(df)
        result = check_class_balance(stats, min_rate=0.10, max_rate=0.40)
        assert result["in_range"] is False

    def test_missing_target_returns_none(self) -> None:
        df = _make_df().drop(columns=["DEFAULT_PAYMENT_NEXT_MONTH"])
        stats = compute_dataset_stats(df, target_col=None)
        result = check_class_balance(stats)
        assert result["in_range"] is None
        assert result["positive_rate"] is None

    def test_result_has_expected_keys(self) -> None:
        df = _make_df(n=500, positive_rate=0.22)
        stats = compute_dataset_stats(df)
        result = check_class_balance(stats)
        assert set(result.keys()) == {"positive_rate", "in_range", "min_rate", "max_rate"}
