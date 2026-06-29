"""Tests for data/contracts/feature_schema.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandera as pa
import pytest

from data.contracts.feature_schema import (
    check_no_infinite_values,
    validate_features,
)

# ── Fixture: minimal valid engineered feature dataframe ──────────────────────

N = 100


@pytest.fixture
def valid_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ID": range(1, N + 1),
        "LIMIT_BAL": rng.choice([50_000.0, 100_000.0, 200_000.0], N),
        "SEX": rng.choice([1, 2], N),
        "EDUCATION": rng.choice([1, 2, 3, 4], N),
        "MARRIAGE": rng.choice([1, 2, 3], N),
        "AGE": rng.integers(20, 70, N).astype(int),
        "PAY_0": rng.integers(-2, 2, N),
        "PAY_2": rng.integers(-2, 2, N),
        "PAY_3": rng.integers(-2, 2, N),
        "PAY_4": rng.integers(-2, 2, N),
        "PAY_5": rng.integers(-2, 2, N),
        "PAY_6": rng.integers(-2, 2, N),
        **{f"BILL_AMT{i}": rng.uniform(-5_000, 50_000, N) for i in range(1, 7)},
        **{f"PAY_AMT{i}": rng.uniform(0, 10_000, N) for i in range(1, 7)},
        "DEFAULT_PAYMENT_NEXT_MONTH": rng.choice([0, 1], N),
        # Derived features within valid bounds
        "utilization_rate": rng.uniform(0.0, 0.9, N),
        "payment_ratio": rng.uniform(0.0, 2.0, N),
        "max_delay": rng.uniform(-2.0, 3.0, N),
        "avg_delay": rng.uniform(-2.0, 2.0, N),
        "consecutive_delays": rng.integers(0, 4, N).astype(float),
        "bill_trend": rng.uniform(-5.0, 5.0, N),
        "total_payment_ratio": rng.uniform(0.0, 2.0, N),
    })


# ── Happy-path tests ──────────────────────────────────────────────────────────

class TestValidData:
    def test_valid_df_passes(self, valid_df: pd.DataFrame) -> None:
        result = validate_features(valid_df)
        assert len(result) == N

    def test_returns_dataframe(self, valid_df: pd.DataFrame) -> None:
        result = validate_features(valid_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_original_columns_preserved(self, valid_df: pd.DataFrame) -> None:
        result = validate_features(valid_df)
        assert set(valid_df.columns).issubset(set(result.columns))

    def test_extra_columns_allowed(self, valid_df: pd.DataFrame) -> None:
        df = valid_df.copy()
        df["extra_col"] = 1.0
        result = validate_features(df)
        assert "extra_col" in result.columns


# ── Derived feature semantic checks ──────────────────────────────────────────

class TestDerivedFeatureChecks:
    def test_nan_utilization_rate_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "utilization_rate"] = float("nan")
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)

    def test_negative_consecutive_delays_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "consecutive_delays"] = -1.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)

    def test_consecutive_delays_above_6_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "consecutive_delays"] = 7.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)

    def test_max_delay_above_9_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "max_delay"] = 10.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)

    def test_negative_payment_ratio_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "payment_ratio"] = -1.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)


# ── Base column checks ────────────────────────────────────────────────────────

class TestBaseColumnChecks:
    def test_cleaned_education_only_1234(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "EDUCATION"] = 5  # raw uncleaned value
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)

    def test_cleaned_marriage_no_zero(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "MARRIAGE"] = 0  # raw uncleaned value
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)

    def test_negative_pay_amt_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "PAY_AMT1"] = -100.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)

    def test_duplicate_id_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[1, "ID"] = bad.loc[0, "ID"]
        with pytest.raises(pa.errors.SchemaErrors):
            validate_features(bad)


# ── Infinite value detection ──────────────────────────────────────────────────

class TestInfiniteValueDetection:
    def test_no_inf_in_clean_data(self, valid_df: pd.DataFrame) -> None:
        result = check_no_infinite_values(valid_df)
        assert result == []

    def test_detects_positive_inf(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "utilization_rate"] = float("inf")
        flagged = check_no_infinite_values(bad)
        assert "utilization_rate" in flagged

    def test_detects_negative_inf(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "bill_trend"] = float("-inf")
        flagged = check_no_infinite_values(bad)
        assert "bill_trend" in flagged

    def test_returns_list(self, valid_df: pd.DataFrame) -> None:
        result = check_no_infinite_values(valid_df)
        assert isinstance(result, list)
