"""Pandera schema for the raw UCI Credit Card Default dataset.

This is the v0 contract (drafted Day 6, formalised Phase 3).
Running this module validates a CSV against the schema.

Usage:
    python -m data.contracts.raw_schema data/raw/credit_card_default.csv

Debugging schema failures:
    - SchemaError output shows the failing column and the rule.
    - Run with --verbose to see a row-level failure table.
    - EDUCATION / MARRIAGE bad values are expected in raw data → clean first.

See: docs/phase0/day06_dataset_eda.md for data quality notes.
See: docs/phase1/day08_dvc_minio.md for threat model (data poisoning).
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

# Valid categorical domains (after cleaning)
_VALID_SEX = [1, 2]
_VALID_EDUCATION = [1, 2, 3, 4]       # 0/5/6 are "others" — cleaned before validation
_VALID_MARRIAGE = [1, 2, 3]           # 0 is "unknown" — cleaned before validation
_VALID_PAY_STATUS = list(range(-2, 10))  # -2=no use, -1=paid, 0=revolving, 1..9=months late

raw_schema = DataFrameSchema(
    columns={
        "ID": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
            unique=True,
            description="Applicant ID (1-indexed, unique per row)",
        ),
        "LIMIT_BAL": Column(
            float,
            checks=Check.between(10_000, 1_000_000),
            nullable=False,
            description="Credit limit (NTD)",
        ),
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
            description="1=grad, 2=university, 3=high school, 4=others (0/5/6 cleaned → 4)",
        ),
        "MARRIAGE": Column(
            int,
            checks=Check.isin(_VALID_MARRIAGE),
            nullable=False,
            description="1=married, 2=single, 3=others (0 cleaned → 3)",
        ),
        "AGE": Column(
            int,
            checks=Check.between(18, 100),
            nullable=False,
        ),
        # Payment status columns — ordinal scale
        **{
            f"PAY_{col}": Column(
                int,
                checks=Check.isin(_VALID_PAY_STATUS),
                nullable=False,
                description=f"Repayment status {col} months ago (-2=no use, -1=paid duly, N=N months late)",
            )
            for col in [0, 2, 3, 4, 5, 6]
        },
        # Bill statement amounts (can be negative = credit balance)
        **{
            f"BILL_AMT{i}": Column(float, nullable=False)
            for i in range(1, 7)
        },
        # Payment amounts (always ≥ 0)
        **{
            f"PAY_AMT{i}": Column(
                float,
                checks=Check.greater_than_or_equal_to(0),
                nullable=False,
            )
            for i in range(1, 7)
        },
        "DEFAULT_PAYMENT_NEXT_MONTH": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Target: 1=defaulted next month, 0=did not",
        ),
    },
    coerce=True,    # cast dtypes where safe
    strict=False,   # allow extra columns (we add engineered features later)
    name="raw_credit_default",
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dataframe against the raw schema. Raises pa.errors.SchemaError on failure."""
    return raw_schema.validate(df, lazy=True)


def check_class_balance(df: pd.DataFrame, col: str = "DEFAULT_PAYMENT_NEXT_MONTH") -> None:
    """Warn if class balance is outside expected 15–30% positive rate."""
    rate = df[col].mean()
    if not (0.15 <= rate <= 0.30):
        log.warning("Class balance %.1f%% is outside expected 15%%–30%% range", rate * 100)
    else:
        log.info("Class balance: %.1f%% positive (expected 15%%–30%%)", rate * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Validate raw CSV against Pandera schema")
    parser.add_argument("csv_path", help="Path to raw CSV")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df.columns = [str(c).strip().upper() for c in df.columns]

    try:
        validate_raw(df)
        check_class_balance(df)
        log.info("✅ Schema validation PASSED for %s", args.csv_path)
    except pa.errors.SchemaErrors as err:
        log.error("❌ Schema validation FAILED:")
        if args.verbose:
            print(err.failure_cases.to_string())
        else:
            print(err.failure_cases.head(10).to_string())
            print(f"... ({len(err.failure_cases)} total failures; use --verbose for all)")
        sys.exit(1)
