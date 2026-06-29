"""Dataset-level statistical checks — the third layer of the data contract.

Pandera handles row-level schema enforcement (Layer 1 + Layer 2).
This module handles dataset-level distribution checks (Layer 3):

  - Null-rate drift: has the fraction of nulls in any column changed?
  - Mean drift: has the mean of any numeric column shifted more than Z sigma?
  - Class balance: is the positive rate still within the expected range?

These checks require a *reference snapshot* (typically computed from training data).
Without a reference, there is no baseline to compare against.

Key dataclasses:
    DatasetStats    — per-column statistics snapshot (serialisable to dict/JSON)
    DriftResult     — per-column drift result with severity flag

Key functions:
    compute_dataset_stats()  — build a DatasetStats from a DataFrame
    check_null_drift()       — compare current vs reference null rates
    check_mean_drift()       — z-score comparison of column means
    check_class_balance()    — verify positive rate in expected range

See: docs/phase3/day19_data_contracts.md for theory.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_NULL_DRIFT_THRESHOLD = 0.05   # 5 percentage-point shift in null rate
_DEFAULT_MEAN_DRIFT_Z = 3.0            # 3-sigma shift triggers alert
_CLASS_BALANCE_MIN = 0.10              # below 10% is concerning
_CLASS_BALANCE_MAX = 0.40              # above 40% is concerning


@dataclass
class ColumnStats:
    """Per-column statistics snapshot for numeric columns."""

    name: str
    null_rate: float
    mean: float | None
    std: float | None
    p5: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    n_unique: int
    n_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ColumnStats":
        return cls(**d)


@dataclass
class DatasetStats:
    """Collection of per-column statistics for one dataset snapshot.

    Attributes:
        columns:       Mapping of column_name → ColumnStats.
        n_rows:        Total number of rows in the snapshot.
        positive_rate: Fraction of target==1 (None if no target column).
        created_at:    ISO-8601 timestamp string when the snapshot was created.
        dataset_name:  Human-readable identifier.
    """

    columns: dict[str, ColumnStats]
    n_rows: int
    positive_rate: float | None
    created_at: str
    dataset_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "n_rows": self.n_rows,
            "positive_rate": self.positive_rate,
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetStats":
        cols = {k: ColumnStats.from_dict(v) for k, v in d["columns"].items()}
        return cls(
            columns=cols,
            n_rows=d["n_rows"],
            positive_rate=d.get("positive_rate"),
            created_at=d["created_at"],
            dataset_name=d["dataset_name"],
        )


@dataclass
class DriftResult:
    """Per-column drift assessment."""

    column: str
    current_value: float
    reference_value: float
    drift: float        # absolute difference
    severity: str       # "ok" | "warn" | "critical"
    flag: bool          # True if drift exceeds threshold


def compute_dataset_stats(
    df: pd.DataFrame,
    *,
    dataset_name: str = "unnamed",
    target_col: str | None = "DEFAULT_PAYMENT_NEXT_MONTH",
) -> DatasetStats:
    """Compute per-column statistics for a DataFrame.

    Only numeric columns are profiled (null rates are computed for all).
    Categorical columns contribute null_rate and n_unique but no mean/std/percentiles.

    Args:
        df:           DataFrame to profile.
        dataset_name: Human-readable label for this snapshot.
        target_col:   If present, compute positive_rate from this binary column.

    Returns:
        DatasetStats snapshot.
    """
    from datetime import datetime, timezone
    created_at = datetime.now(timezone.utc).isoformat()

    col_stats: dict[str, ColumnStats] = {}
    n = len(df)

    for col in df.columns:
        series = df[col]
        null_rate = float(series.isna().mean())
        n_unique = int(series.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                mean = float(non_null.mean())
                std = float(non_null.std())
                p5 = float(np.percentile(non_null, 5))
                p25 = float(np.percentile(non_null, 25))
                p50 = float(np.percentile(non_null, 50))
                p75 = float(np.percentile(non_null, 75))
                p95 = float(np.percentile(non_null, 95))
            else:
                mean = std = p5 = p25 = p50 = p75 = p95 = None
        else:
            mean = std = p5 = p25 = p50 = p75 = p95 = None

        col_stats[col] = ColumnStats(
            name=col,
            null_rate=null_rate,
            mean=mean,
            std=std,
            p5=p5,
            p25=p25,
            p50=p50,
            p75=p75,
            p95=p95,
            n_unique=n_unique,
            n_rows=n,
        )

    positive_rate: float | None = None
    if target_col and target_col in df.columns:
        positive_rate = float(df[target_col].mean())

    log.info(
        "Computed stats for %d columns, %d rows (positive_rate=%.3f)",
        len(col_stats),
        n,
        positive_rate if positive_rate is not None else -1.0,
    )
    return DatasetStats(
        columns=col_stats,
        n_rows=n,
        positive_rate=positive_rate,
        created_at=created_at,
        dataset_name=dataset_name,
    )


def check_null_drift(
    current: DatasetStats,
    reference: DatasetStats,
    threshold: float = _DEFAULT_NULL_DRIFT_THRESHOLD,
) -> pd.DataFrame:
    """Compare null rates between current and reference snapshots.

    Args:
        current:    Stats for the dataset being validated.
        reference:  Stats for the known-good baseline (e.g. training data).
        threshold:  Absolute null-rate difference that triggers a flag.

    Returns:
        DataFrame with columns: column, current_null_rate, reference_null_rate, drift, flag.
        Sorted by drift descending.
    """
    rows = []
    common_cols = set(current.columns) & set(reference.columns)

    for col in sorted(common_cols):
        cur_null = current.columns[col].null_rate
        ref_null = reference.columns[col].null_rate
        drift = abs(cur_null - ref_null)
        flag = drift > threshold
        if flag:
            log.warning(
                "Null drift on %r: reference=%.4f current=%.4f drift=%.4f (threshold=%.4f)",
                col, ref_null, cur_null, drift, threshold,
            )
        rows.append({
            "column": col,
            "current_null_rate": cur_null,
            "reference_null_rate": ref_null,
            "drift": drift,
            "flag": flag,
        })

    df = pd.DataFrame(rows).sort_values("drift", ascending=False).reset_index(drop=True)
    n_flagged = int(df["flag"].sum())
    log.info("Null drift check: %d/%d columns flagged (threshold=%.4f)", n_flagged, len(df), threshold)
    return df


def check_mean_drift(
    current: DatasetStats,
    reference: DatasetStats,
    z_threshold: float = _DEFAULT_MEAN_DRIFT_Z,
) -> pd.DataFrame:
    """Compare column means using z-score normalised by reference std.

    z = |mean_current - mean_reference| / (std_reference + ε)

    A z-score > 3.0 means the current mean is more than 3 standard deviations
    away from the reference mean — unlikely under the null hypothesis of no shift.

    Args:
        current:     Current dataset stats.
        reference:   Reference (training-time) dataset stats.
        z_threshold: Z-score above which we flag (default 3.0).

    Returns:
        DataFrame with columns: column, current_mean, reference_mean, z_score, flag.
        Only numeric columns with non-None means are included.
        Sorted by z_score descending.
    """
    rows = []
    common_cols = set(current.columns) & set(reference.columns)

    for col in sorted(common_cols):
        cur_stats = current.columns[col]
        ref_stats = reference.columns[col]

        if cur_stats.mean is None or ref_stats.mean is None:
            continue  # categorical / all-null column

        ref_std = ref_stats.std or 0.0
        eps = 1e-8
        z = abs(cur_stats.mean - ref_stats.mean) / (ref_std + eps)

        # Map z-score to severity tier
        if z <= 2.0:
            severity = "ok"
        elif z <= z_threshold:
            severity = "warn"
        else:
            severity = "critical"

        flag = z > z_threshold
        if flag:
            log.warning(
                "Mean drift on %r: ref_mean=%.4f cur_mean=%.4f z=%.2f (threshold=%.1f)",
                col, ref_stats.mean, cur_stats.mean, z, z_threshold,
            )
        rows.append({
            "column": col,
            "current_mean": cur_stats.mean,
            "reference_mean": ref_stats.mean,
            "z_score": z,
            "severity": severity,
            "flag": flag,
        })

    df = pd.DataFrame(rows).sort_values("z_score", ascending=False).reset_index(drop=True)
    n_flagged = int(df["flag"].sum())
    log.info("Mean drift check: %d/%d numeric columns flagged (z_threshold=%.1f)", n_flagged, len(df), z_threshold)
    return df


def check_class_balance(
    stats: DatasetStats,
    *,
    min_rate: float = _CLASS_BALANCE_MIN,
    max_rate: float = _CLASS_BALANCE_MAX,
) -> dict[str, Any]:
    """Check whether the positive rate is within the expected range.

    Args:
        stats:    DatasetStats with a computed positive_rate.
        min_rate: Minimum acceptable positive rate (default 0.10).
        max_rate: Maximum acceptable positive rate (default 0.40).

    Returns:
        Dict with keys: positive_rate, in_range, min_rate, max_rate.
    """
    rate = stats.positive_rate
    if rate is None:
        log.warning("Cannot check class balance: positive_rate not computed (target_col missing?)")
        return {"positive_rate": None, "in_range": None, "min_rate": min_rate, "max_rate": max_rate}

    in_range = min_rate <= rate <= max_rate
    if not in_range:
        log.warning(
            "Class balance %.1f%% is outside expected %.0f%%–%.0f%% range",
            rate * 100, min_rate * 100, max_rate * 100,
        )
    else:
        log.info("Class balance: %.1f%% positive (OK: %.0f%%–%.0f%%)", rate * 100, min_rate * 100, max_rate * 100)

    return {
        "positive_rate": rate,
        "in_range": in_range,
        "min_rate": min_rate,
        "max_rate": max_rate,
    }
