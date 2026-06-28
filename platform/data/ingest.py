"""Download and save the UCI Credit Card Default dataset.

The UCI dataset arrives as an .xls file with a title row above the header.
We normalise column names to UPPER_CASE and rename the confusing PAY_1 → PAY_0.

Usage (DVC stage):
    python -m data.ingest --output data/raw/credit_card_default.csv

Debugging tips:
    - If the download fails: check network, try setting HTTP_PROXY env var.
    - If column count is wrong: the XLS format may have changed; update header=1.
    - Run with DEBUG logging for row-by-row info: LOG_LEVEL=DEBUG python -m data.ingest
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# UCI dataset #350 — Credit Card Default
UCI_XLS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)
EXPECTED_ROW_COUNT = 30_000
EXPECTED_COL_COUNT = 25


def download_raw(output_path: Path) -> pd.DataFrame:
    """Download UCI Credit Default, normalise columns, save CSV. Returns DataFrame."""
    log.info("Downloading %s", UCI_XLS_URL)

    # header=1 skips the title row that UCI adds above the actual header
    df = pd.read_excel(UCI_XLS_URL, header=1, engine="xlrd")

    df = _normalise_columns(df)
    _validate_shape(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Saved %d rows × %d cols to %s", len(df), len(df.columns), output_path)
    return df


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, upper-case, rename PAY_1 → PAY_0 (UCI naming quirk)."""
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    # UCI dataset sometimes exports as PAY_1 for the most-recent month
    if "PAY_1" in df.columns and "PAY_0" not in df.columns:
        df = df.rename(columns={"PAY_1": "PAY_0"})
    return df


def _validate_shape(df: pd.DataFrame) -> None:
    if len(df) < EXPECTED_ROW_COUNT * 0.95:
        raise ValueError(
            f"Expected ≥{int(EXPECTED_ROW_COUNT*0.95)} rows, got {len(df)}. "
            "The download may be incomplete."
        )
    required = {
        "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "DEFAULT_PAYMENT_NEXT_MONTH",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalisation: {missing}")
    log.info("Shape validation passed: %d rows, %d columns", len(df), len(df.columns))


def load_from_disk(csv_path: Path) -> pd.DataFrame:
    """Load already-downloaded data. Used by training when not re-ingesting."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. Run: python -m data.ingest"
        )
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().upper() for c in df.columns]
    log.info("Loaded %d rows from %s", len(df), csv_path)
    return df


if __name__ == "__main__":
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest UCI Credit Default dataset")
    parser.add_argument(
        "--output", default="data/raw/credit_card_default.csv",
        help="Destination CSV path (default: data/raw/credit_card_default.csv)",
    )
    args = parser.parse_args()
    download_raw(Path(args.output))
