"""Pandera schema for the engineered feature dataset (post-featurization).

Validates the output of training.features.engineer_features():
  - 32 base columns from raw_schema (cleaned)
  - 7 derived columns added by engineer_features()

Distinct from raw_schema in two ways:
  1. EDUCATION / MARRIAGE have already been cleaned (no 0/5/6 values)
  2. 7 derived feature columns are present and checked for semantic invariants

Critical: run clean_raw_data() + engineer_features() before this schema.
Running this on raw (un-cleaned) data will raise SchemaErrors for EDUCATION.

Usage:
    from data.contracts.feature_schema import validate_features, feature_schema
    validated_df = validate_features(df)

Run standalone:
    uv run python -m data.contracts.feature_schema data/processed/features.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

log = logging.getLogger(__name__)

# Valid categorical domains — must have been cleaned before this schema runs
_VALID_SEX = [1, 2]
_VALID_EDUCATION = [1, 2, 3, 4]      # 0/5/6 remapped to 4 by clean_raw_data()
_VALID_MARRIAGE = [1, 2, 3]          # 0 remapped to 3 by clean_raw_data()
_VALID_PAY_STATUS = list(range(-2, 10))

# Derived feature bounds — see training/features.py for derivation
# These are generous bounds; they catch NaN/inf and extreme coding errors,
# not natural outliers (e.g. a customer with a very large credit line).
_UTIL_RATE_BOUNDS = (-1.0, 20.0)     # BILL_AMT1 / (LIMIT_BAL+1)
_PAYMENT_RATIO_BOUNDS = (0.0, 500.0) # PAY_AMT1 / (|BILL_AMT1|+1); 0-floor because PAY_AMT >= 0
_DELAY_BOUNDS = (-2.0, 9.0)          # same domain as PAY_* status values
_CONSEC_DELAY_BOUNDS = (0.0, 6.0)    # count of PAY_* > 0; 6 months max
_BILL_TREND_BOUNDS = (-500.0, 500.0) # (BILL_AMT1 - BILL_AMT6) / (|BILL_AMT6|+1)
_TOTAL_PAY_RATIO_BOUNDS = (0.0, 500.0)

feature_schema = DataFrameSchema(
    columns={
        # ── Identity ─────────────────────────────────────────────────────────
        "ID": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
            unique=True,
            description="Applicant ID — must be unique",
        ),

        # ── Base numeric features ─────────────────────────────────────────────
        "LIMIT_BAL": Column(
            float,
            checks=Check.between(10_000, 1_000_000),
            nullable=False,
            description="Credit limit (NTD)",
        ),
        "AGE": Column(
            int,
            checks=Check.between(18, 100),
            nullable=False,
        ),

        # ── Cleaned categoricals ──────────────────────────────────────────────
        "SEX": Column(
            int,
            checks=Check.isin(_VALID_SEX),
            nullable=False,
            description="1=male, 2=female",
        ),
        "EDUCATION": Column(
            int,
            checks=Check.isin(_VALID_EDUCATION),
            nullable=False,
            description="1=grad, 2=univ, 3=high-school, 4=other. Must be pre-cleaned.",
        ),
        "MARRIAGE": Column(
            int,
            checks=Check.isin(_VALID_MARRIAGE),
            nullable=False,
            description="1=married, 2=single, 3=other. Must be pre-cleaned.",
        ),

        # ── Payment status columns ────────────────────────────────────────────
        **{
            f"PAY_{col}": Column(
                int,
                checks=Check.isin(_VALID_PAY_STATUS),
                nullable=False,
            )
            for col in [0, 2, 3, 4, 5, 6]
        },

        # ── Bill statement amounts (negative = credit balance) ────────────────
        **{
            f"BILL_AMT{i}": Column(float, nullable=False)
            for i in range(1, 7)
        },

        # ── Payment amounts (always ≥ 0) ──────────────────────────────────────
        **{
            f"PAY_AMT{i}": Column(
                float,
                checks=Check.greater_than_or_equal_to(0),
                nullable=False,
            )
            for i in range(1, 7)
        },

        # ── Target ───────────────────────────────────────────────────────────
        "DEFAULT_PAYMENT_NEXT_MONTH": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="1=defaulted next month, 0=did not",
        ),

        # ── Derived features (added by engineer_features()) ──────────────────
        "utilization_rate": Column(
            float,
            checks=[
                Check.between(*_UTIL_RATE_BOUNDS),
                Check(lambda s: s.notna().all(), error="utilization_rate has NaN"),
            ],
            nullable=False,
            description="BILL_AMT1 / (LIMIT_BAL+1): credit utilisation fraction",
        ),
        "payment_ratio": Column(
            float,
            checks=[
                Check.between(*_PAYMENT_RATIO_BOUNDS),
                Check(lambda s: s.notna().all(), error="payment_ratio has NaN"),
            ],
            nullable=False,
            description="PAY_AMT1 / (|BILL_AMT1|+1): how much of the last bill was paid",
        ),
        "max_delay": Column(
            float,
            checks=Check.between(*_DELAY_BOUNDS),
            nullable=False,
            description="max(PAY_*): worst single-month repayment delay",
        ),
        "avg_delay": Column(
            float,
            checks=Check.between(*_DELAY_BOUNDS),
            nullable=False,
            description="mean(PAY_*): average delay across the 6-month window",
        ),
        "consecutive_delays": Column(
            float,
            checks=Check.between(*_CONSEC_DELAY_BOUNDS),
            nullable=False,
            description="count(PAY_* > 0): months with active delinquency",
        ),
        "bill_trend": Column(
            float,
            checks=[
                Check.between(*_BILL_TREND_BOUNDS),
                Check(lambda s: s.notna().all(), error="bill_trend has NaN"),
            ],
            nullable=False,
            description="(BILL_AMT1 - BILL_AMT6)/(|BILL_AMT6|+1): debt trajectory",
        ),
        "total_payment_ratio": Column(
            float,
            checks=[
                Check.between(*_TOTAL_PAY_RATIO_BOUNDS),
                Check(lambda s: s.notna().all(), error="total_payment_ratio has NaN"),
            ],
            nullable=False,
            description="sum(PAY_AMT) / (sum(|BILL_AMT|)+1): 6-month payment effort",
        ),
    },
    coerce=True,
    strict=False,  # extra columns (e.g. intermediate engineered cols) are allowed
    name="credit_feature_v1",
)


def validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validate engineered features DataFrame. Raises pa.errors.SchemaErrors on failure.

    Call after clean_raw_data() + engineer_features() — not on raw data.
    Uses lazy=True to collect ALL violations before raising.
    """
    return feature_schema.validate(df, lazy=True)


def check_no_infinite_values(df: pd.DataFrame) -> list[str]:
    """Return list of column names that contain any inf/-inf values.

    Pandera does not check for inf by default (they pass type checks but
    break downstream model training). Run this as a supplementary check.
    """
    numeric = df.select_dtypes(include="number")
    return [col for col in numeric.columns if numeric[col].isin([float("inf"), float("-inf")]).any()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validate feature parquet against feature schema")
    parser.add_argument("parquet_path", help="Path to features.parquet")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_path)

    inf_cols = check_no_infinite_values(df)
    if inf_cols:
        log.error("❌ Infinite values found in: %s", inf_cols)
        sys.exit(1)

    try:
        validate_features(df)
        log.info("✅ Feature schema validation PASSED for %s (%d rows)", args.parquet_path, len(df))
    except pa.errors.SchemaErrors as err:
        log.error("❌ Feature schema validation FAILED:")
        if args.verbose:
            print(err.failure_cases.to_string())
        else:
            print(err.failure_cases.head(10).to_string())
            print(f"... ({len(err.failure_cases)} total failures; use --verbose for all)")
        sys.exit(1)
