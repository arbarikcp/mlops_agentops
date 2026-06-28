"""Unit tests for data/contracts/raw_schema.py.

Tests verify the Pandera schema rejects bad data and accepts clean data.
Each test exercises a specific rule in raw_schema.

Run:
    cd platform && uv run pytest tests/data/test_raw_schema.py -v
"""
import numpy as np
import pandas as pd
import pandera as pa
import pytest

from data.contracts.raw_schema import check_class_balance, validate_raw


# ── Fixture: minimal valid dataframe ─────────────────────────────────────────

N = 60

@pytest.fixture
def valid_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ID": range(1, N + 1),
        "LIMIT_BAL": rng.choice([50_000.0, 100_000.0, 200_000.0], N),
        "SEX": rng.choice([1, 2], N),
        "EDUCATION": rng.choice([1, 2, 3, 4], N),       # already clean
        "MARRIAGE": rng.choice([1, 2, 3], N),            # already clean
        "AGE": rng.integers(20, 70, N).astype(int),
        "PAY_0": rng.integers(-2, 2, N),
        "PAY_2": rng.integers(-2, 2, N),
        "PAY_3": rng.integers(-2, 2, N),
        "PAY_4": rng.integers(-2, 2, N),
        "PAY_5": rng.integers(-2, 2, N),
        "PAY_6": rng.integers(-2, 2, N),
        **{f"BILL_AMT{i}": rng.uniform(0, 50_000, N) for i in range(1, 7)},
        **{f"PAY_AMT{i}": rng.uniform(0, 10_000, N) for i in range(1, 7)},
        "DEFAULT_PAYMENT_NEXT_MONTH": rng.choice([0, 1], N),
    })


# ── Happy-path tests ──────────────────────────────────────────────────────────

class TestValidData:
    def test_valid_df_passes(self, valid_df: pd.DataFrame) -> None:
        result = validate_raw(valid_df)
        assert len(result) == N

    def test_returns_dataframe(self, valid_df: pd.DataFrame) -> None:
        result = validate_raw(valid_df)
        assert isinstance(result, pd.DataFrame)


# ── Failure-path tests ────────────────────────────────────────────────────────

class TestSchemaFailures:
    def test_bad_education_value_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "EDUCATION"] = 5  # undocumented — should be cleaned before validation
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_bad_marriage_value_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "MARRIAGE"] = 0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_negative_pay_amt_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "PAY_AMT1"] = -500.0
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_invalid_sex_value_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "SEX"] = 3  # only 1 or 2 allowed
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_invalid_target_value_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "DEFAULT_PAYMENT_NEXT_MONTH"] = 2
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_duplicate_id_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[1, "ID"] = bad.loc[0, "ID"]
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_age_below_18_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "AGE"] = 15
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_limit_bal_too_low_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "LIMIT_BAL"] = 100.0  # below 10,000 minimum
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)

    def test_null_target_fails(self, valid_df: pd.DataFrame) -> None:
        bad = valid_df.copy()
        bad.loc[0, "DEFAULT_PAYMENT_NEXT_MONTH"] = None
        with pytest.raises(pa.errors.SchemaErrors):
            validate_raw(bad)


# ── Class balance check ───────────────────────────────────────────────────────

class TestClassBalance:
    def test_no_warning_for_normal_balance(
        self, valid_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Ensure ~22% positive rate
        df = valid_df.copy()
        n_pos = int(N * 0.22)
        df["DEFAULT_PAYMENT_NEXT_MONTH"] = [1] * n_pos + [0] * (N - n_pos)
        import logging
        with caplog.at_level(logging.WARNING, logger="data.contracts.raw_schema"):
            check_class_balance(df)
        assert "outside expected" not in caplog.text

    def test_warning_for_extreme_imbalance(
        self, valid_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        df = valid_df.copy()
        df["DEFAULT_PAYMENT_NEXT_MONTH"] = 0  # 0% positive
        import logging
        with caplog.at_level(logging.WARNING, logger="data.contracts.raw_schema"):
            check_class_balance(df)
        assert "outside expected" in caplog.text
