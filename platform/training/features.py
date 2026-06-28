"""Feature engineering for the credit-risk model.

All functions are pure (no side effects, no randomness) — guaranteed deterministic.
The DVC featurize stage calls this module's __main__ block.

Data flow:
    raw CSV → clean_raw_data() → engineer_features() → split_data() → X_train, X_test

Why pure functions:
  - Easy to unit-test without mocking
  - Determinism is provable (same input → same output, always)
  - Can be called from both DVC pipeline and MLflow training

Usage (DVC stage):
    python -m training.features \\
        --input data/raw/credit_card_default.csv \\
        --output data/processed/features.parquet \\
        --params params.yaml

Usage (from code):
    from training.features import clean_raw_data, engineer_features, split_data
    from training.config import TrainingParams
    cfg = TrainingParams.from_yaml("params.yaml")
    df_clean = clean_raw_data(df_raw)
    df_feat  = engineer_features(df_clean, cfg.features)
    X_train, X_test, y_train, y_test = split_data(df_feat, cfg)

Debugging:
    - Print df.describe() after each step to catch NaN/inf propagation early.
    - Verify with: df_feat.isnull().sum()  →  should be all zeros.
    - Check engineer_features output shape: should be raw_cols + 7 derived features.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from training.config import DataConfig, FeaturesConfig, TrainingParams

log = logging.getLogger(__name__)

# Derived feature names — defined here so tests can import them
DERIVED_FEATURE_COLS = [
    "utilization_rate",
    "payment_ratio",
    "max_delay",
    "avg_delay",
    "consecutive_delays",
    "bill_trend",
    "total_payment_ratio",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data quality issues in UCI Credit Default raw data.

    Issues found in EDA (see docs/phase0/day06_dataset_eda.md):
    - EDUCATION values 0, 5, 6 are undocumented → remap to 4 (others)
    - MARRIAGE value 0 is undocumented → remap to 3 (others)

    Does NOT modify the input dataframe (returns a copy).
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    log.debug("Cleaned EDUCATION and MARRIAGE undocumented values")
    return df


def engineer_features(df: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Compute 7 derived features from the raw credit columns.

    All derived features are ratio/aggregate transforms — no fitted parameters,
    no state, no randomness. Safe to call multiple times on the same data.

    Derived features:
        utilization_rate    = BILL_AMT1 / (LIMIT_BAL + 1)
            How much of the credit limit is currently used.

        payment_ratio       = PAY_AMT1 / (|BILL_AMT1| + 1)
            Did the applicant pay their most recent bill? >1 = overpaid.

        max_delay           = max of PAY_* columns
            Worst single-month delay. Strong individual predictor.

        avg_delay           = mean of PAY_* columns
            Average delay across the 6-month window.

        consecutive_delays  = count of PAY_* > 0
            How many months had a positive delay (sustained delinquency).

        bill_trend          = (BILL_AMT1 - BILL_AMT6) / (|BILL_AMT6| + 1)
            Is debt growing (+) or shrinking (-)?

        total_payment_ratio = sum(PAY_AMT1..6) / (sum(|BILL_AMT1..6|) + 1)
            Aggregate payment effort over 6 months.
    """
    df = df.copy()

    bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
    pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
    pay_status_cols = cfg.payment_status_cols

    # Guard: avoid divide-by-zero via +1 denominator (avoids inf, not NaN)
    df["utilization_rate"] = df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1.0)
    df["payment_ratio"] = df["PAY_AMT1"] / (df["BILL_AMT1"].abs() + 1.0)

    pay_status = df[pay_status_cols]
    df["max_delay"] = pay_status.max(axis=1)
    df["avg_delay"] = pay_status.mean(axis=1)
    df["consecutive_delays"] = (pay_status > 0).sum(axis=1).astype(float)

    df["bill_trend"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / (df["BILL_AMT6"].abs() + 1.0)

    total_paid = df[pay_amt_cols].sum(axis=1)
    total_billed = df[bill_cols].abs().sum(axis=1)
    df["total_payment_ratio"] = total_paid / (total_billed + 1.0)

    log.debug("Engineered %d derived features", len(DERIVED_FEATURE_COLS))
    return df


def split_data(
    df: pd.DataFrame,
    cfg: DataConfig | None = None,
    *,
    target_col: str = "DEFAULT_PAYMENT_NEXT_MONTH",
    id_col: str = "ID",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Time-based train/test split to prevent data leakage.

    Why time-based, not random:
        Credit data has temporal structure — using random shuffle would allow
        future information (newer applications) to appear in training data,
        making evaluation metrics artificially optimistic.
        The UCI dataset is ordered by ID (proxy for time).

    Returns:
        X_train, X_test, y_train, y_test
    """
    if cfg is not None:
        target_col = cfg.target_col
        test_size = cfg.test_size
        random_seed = cfg.random_seed

    # Sort by ID to ensure consistent temporal ordering across runs
    df_sorted = df.sort_values(id_col).reset_index(drop=True)

    n = len(df_sorted)
    split_idx = int(n * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    exclude = {target_col, id_col}
    feature_cols = [c for c in df_sorted.columns if c not in exclude]

    X_train = train_df[feature_cols].reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_train = train_df[target_col].reset_index(drop=True)
    y_test = test_df[target_col].reset_index(drop=True)

    log.info(
        "Split: train=%d (%.0f%%) test=%d (%.0f%%) | features=%d",
        len(X_train), (1 - test_size) * 100,
        len(X_test), test_size * 100,
        len(feature_cols),
    )
    return X_train, X_test, y_train, y_test


# ── DVC featurize stage entry point ─────────────────────────────────────────

def _featurize(input_path: Path, output_path: Path, params_path: Path) -> None:
    """Read raw CSV → clean → engineer → write parquet. Called by DVC stage."""
    from training.config import TrainingParams
    cfg = TrainingParams.from_yaml(params_path)

    df = pd.read_csv(input_path)
    df.columns = [str(c).strip().upper() for c in df.columns]
    log.info("Loaded %d rows from %s", len(df), input_path)

    df = clean_raw_data(df)
    df = engineer_features(df, cfg.features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("Saved %d rows × %d cols to %s", len(df), len(df.columns), output_path)


if __name__ == "__main__":
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Featurize raw credit data")
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output", required=True, help="Path for output parquet")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    _featurize(Path(args.input), Path(args.output), Path(args.params))
