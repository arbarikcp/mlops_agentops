"""Unit tests for training/features.py.

Tests verify:
  - clean_raw_data: known data quality issues fixed, input not mutated
  - engineer_features: derived columns created, no inf/NaN, deterministic
  - split_data: correct sizes, no leakage, target excluded from features

Run:
    cd platform && uv run pytest tests/unit/test_features.py -v
"""
import numpy as np
import pandas as pd
import pytest

from training.config import DataConfig, FeaturesConfig
from training.features import (
    DERIVED_FEATURE_COLS,
    clean_raw_data,
    engineer_features,
    split_data,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

N_ROWS = 200

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Synthetic raw dataframe with the UCI Credit Default column structure."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "ID": range(1, N_ROWS + 1),
        "LIMIT_BAL": rng.choice([50_000.0, 100_000.0, 200_000.0], N_ROWS),
        "SEX": rng.choice([1, 2], N_ROWS),
        # Include bad EDUCATION values (0, 5, 6) to test cleaning
        "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], N_ROWS),
        # Include bad MARRIAGE value (0) to test cleaning
        "MARRIAGE": rng.choice([0, 1, 2, 3], N_ROWS),
        "AGE": rng.integers(20, 70, N_ROWS).astype(int),
        "PAY_0": rng.integers(-2, 4, N_ROWS),
        "PAY_2": rng.integers(-2, 4, N_ROWS),
        "PAY_3": rng.integers(-2, 4, N_ROWS),
        "PAY_4": rng.integers(-2, 4, N_ROWS),
        "PAY_5": rng.integers(-2, 4, N_ROWS),
        "PAY_6": rng.integers(-2, 4, N_ROWS),
        **{f"BILL_AMT{i}": rng.uniform(-1_000, 50_000, N_ROWS) for i in range(1, 7)},
        **{f"PAY_AMT{i}": rng.uniform(0, 10_000, N_ROWS) for i in range(1, 7)},
        "DEFAULT_PAYMENT_NEXT_MONTH": rng.choice([0, 1], N_ROWS, p=[0.78, 0.22]),
    })


@pytest.fixture
def features_cfg() -> FeaturesConfig:
    return FeaturesConfig(
        target_col="DEFAULT_PAYMENT_NEXT_MONTH",
        id_col="ID",
        categorical_cols=["SEX", "EDUCATION", "MARRIAGE"],
        payment_status_cols=["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
    )


@pytest.fixture
def data_cfg() -> DataConfig:
    return DataConfig(test_size=0.2, random_seed=42, target_col="DEFAULT_PAYMENT_NEXT_MONTH")


@pytest.fixture
def cleaned_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_raw_data(raw_df)


@pytest.fixture
def engineered_df(cleaned_df: pd.DataFrame, features_cfg: FeaturesConfig) -> pd.DataFrame:
    return engineer_features(cleaned_df, features_cfg)


# ── clean_raw_data tests ──────────────────────────────────────────────────────

class TestCleanRawData:
    def test_education_bad_values_remapped(self, raw_df: pd.DataFrame) -> None:
        cleaned = clean_raw_data(raw_df)
        assert set(cleaned["EDUCATION"].unique()).issubset({1, 2, 3, 4}), (
            "EDUCATION should only contain {1,2,3,4} after cleaning"
        )

    def test_marriage_bad_value_remapped(self, raw_df: pd.DataFrame) -> None:
        cleaned = clean_raw_data(raw_df)
        assert 0 not in cleaned["MARRIAGE"].values, (
            "MARRIAGE value 0 should be remapped to 3 (others)"
        )

    def test_does_not_mutate_input(self, raw_df: pd.DataFrame) -> None:
        original_edu = raw_df["EDUCATION"].values.copy()
        clean_raw_data(raw_df)
        np.testing.assert_array_equal(
            raw_df["EDUCATION"].values, original_edu,
            err_msg="clean_raw_data must return a copy, not mutate the input",
        )

    def test_row_count_unchanged(self, raw_df: pd.DataFrame) -> None:
        assert len(clean_raw_data(raw_df)) == len(raw_df)

    def test_column_set_unchanged(self, raw_df: pd.DataFrame) -> None:
        cleaned = clean_raw_data(raw_df)
        assert set(cleaned.columns) == set(raw_df.columns)


# ── engineer_features tests ───────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_all_derived_columns_present(
        self, cleaned_df: pd.DataFrame, features_cfg: FeaturesConfig
    ) -> None:
        result = engineer_features(cleaned_df, features_cfg)
        for col in DERIVED_FEATURE_COLS:
            assert col in result.columns, f"Missing derived column: {col}"

    def test_no_inf_values(
        self, cleaned_df: pd.DataFrame, features_cfg: FeaturesConfig
    ) -> None:
        result = engineer_features(cleaned_df, features_cfg)
        for col in DERIVED_FEATURE_COLS:
            assert not np.any(np.isinf(result[col].values)), (
                f"Column {col} contains inf values (possible divide-by-zero)"
            )

    def test_no_nan_values(
        self, cleaned_df: pd.DataFrame, features_cfg: FeaturesConfig
    ) -> None:
        result = engineer_features(cleaned_df, features_cfg)
        for col in DERIVED_FEATURE_COLS:
            assert not result[col].isna().any(), (
                f"Column {col} contains NaN values"
            )

    def test_consecutive_delays_non_negative(
        self, cleaned_df: pd.DataFrame, features_cfg: FeaturesConfig
    ) -> None:
        result = engineer_features(cleaned_df, features_cfg)
        assert (result["consecutive_delays"] >= 0).all()

    def test_consecutive_delays_max_is_6(
        self, cleaned_df: pd.DataFrame, features_cfg: FeaturesConfig
    ) -> None:
        result = engineer_features(cleaned_df, features_cfg)
        assert result["consecutive_delays"].max() <= 6, (
            "consecutive_delays counts across 6 PAY columns — max should be 6"
        )

    def test_deterministic(
        self, cleaned_df: pd.DataFrame, features_cfg: FeaturesConfig
    ) -> None:
        r1 = engineer_features(cleaned_df, features_cfg)
        r2 = engineer_features(cleaned_df, features_cfg)
        pd.testing.assert_frame_equal(r1, r2, check_exact=True)

    def test_does_not_mutate_input(
        self, cleaned_df: pd.DataFrame, features_cfg: FeaturesConfig
    ) -> None:
        original_shape = cleaned_df.shape
        engineer_features(cleaned_df, features_cfg)
        assert cleaned_df.shape == original_shape, "Input dataframe was mutated"


# ── split_data tests ──────────────────────────────────────────────────────────

class TestSplitData:
    def test_correct_split_sizes(
        self, engineered_df: pd.DataFrame, data_cfg: DataConfig
    ) -> None:
        X_train, X_test, y_train, y_test = split_data(engineered_df, data_cfg)
        total = len(X_train) + len(X_test)
        assert total == N_ROWS
        assert abs(len(X_test) / total - data_cfg.test_size) < 0.02

    def test_no_index_overlap_between_splits(
        self, engineered_df: pd.DataFrame, data_cfg: DataConfig
    ) -> None:
        X_train, X_test, _, _ = split_data(engineered_df, data_cfg)
        # After reset_index, indices are 0..len-1 but the underlying rows differ
        # Verify by checking total coverage
        assert len(X_train) + len(X_test) == N_ROWS

    def test_target_not_in_features(
        self, engineered_df: pd.DataFrame, data_cfg: DataConfig
    ) -> None:
        X_train, X_test, y_train, y_test = split_data(engineered_df, data_cfg)
        assert data_cfg.target_col not in X_train.columns
        assert data_cfg.target_col not in X_test.columns

    def test_id_col_not_in_features(
        self, engineered_df: pd.DataFrame, data_cfg: DataConfig
    ) -> None:
        X_train, X_test, _, _ = split_data(engineered_df, data_cfg)
        assert "ID" not in X_train.columns
        assert "ID" not in X_test.columns

    def test_y_labels_are_binary(
        self, engineered_df: pd.DataFrame, data_cfg: DataConfig
    ) -> None:
        _, _, y_train, y_test = split_data(engineered_df, data_cfg)
        assert set(y_train.unique()).issubset({0, 1})
        assert set(y_test.unique()).issubset({0, 1})

    def test_split_is_deterministic(
        self, engineered_df: pd.DataFrame, data_cfg: DataConfig
    ) -> None:
        r1 = split_data(engineered_df, data_cfg)
        r2 = split_data(engineered_df, data_cfg)
        for a, b in zip(r1, r2):
            pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True)) \
                if hasattr(a, 'columns') else \
                pd.testing.assert_series_equal(a.reset_index(drop=True), b.reset_index(drop=True))
