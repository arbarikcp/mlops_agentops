"""Ground-truth pipeline utilities — join, backfill, and correction detection.

In production, ground-truth labels are assembled from two sources:
  1. Prediction log: applicant_id, prediction_date, risk_score, model_version
  2. Outcome store: applicant_id, outcome_date, outcome_type (default/paid)

This module provides:
  - join_predictions_with_outcomes(): inner join filtered by outcome delay
  - detect_label_corrections(): find applicant_ids with multiple label rows
  - backfill_labels(): deduplicate to most recent label per applicant
  - LabelArrivalCurve: compute T+1/7/30/90/180 confirmation fractions

See: docs/phase3/day20_label_contracts.md for theory.

Usage:
    from data.contracts.ground_truth import (
        join_predictions_with_outcomes,
        backfill_labels,
        LabelArrivalCurve,
    )

    joined = join_predictions_with_outcomes(predictions, outcomes, outcome_delay_days=90)
    labels = backfill_labels(joined)
    curve = LabelArrivalCurve.compute(labels)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)


def join_predictions_with_outcomes(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    outcome_delay_days: int = 90,
    id_col: str = "applicant_id",
    pred_date_col: str = "prediction_date",
    outcome_date_col: str = "outcome_date",
) -> pd.DataFrame:
    """Inner join predictions with outcomes, keeping only confirmed labels.

    A label is "confirmed" when:
        outcome_date >= prediction_date + outcome_delay_days

    This prevents training on provisional outcomes that may change.

    Args:
        predictions:        DataFrame with at minimum [id_col, pred_date_col].
        outcomes:           DataFrame with at minimum [id_col, outcome_date_col, "label"].
        outcome_delay_days: Minimum days between prediction and confirmed outcome.
        id_col:             Column used as the join key.
        pred_date_col:      Column in predictions recording when prediction was made.
        outcome_date_col:   Column in outcomes recording when outcome was confirmed.

    Returns:
        Joined DataFrame containing only rows where the outcome delay is satisfied.
        Rows from predictions with no matching confirmed outcome are dropped (inner join).
    """
    pred = predictions.copy()
    out = outcomes.copy()

    pred[pred_date_col] = pd.to_datetime(pred[pred_date_col])
    out[outcome_date_col] = pd.to_datetime(out[outcome_date_col])

    merged = pred.merge(out, on=id_col, how="inner")

    min_outcome = merged[pred_date_col] + pd.Timedelta(days=outcome_delay_days)
    confirmed_mask = merged[outcome_date_col] >= min_outcome
    result = merged[confirmed_mask].reset_index(drop=True)

    n_dropped = len(merged) - len(result)
    log.info(
        "Ground-truth join: %d predictions + %d outcomes → %d confirmed rows (%d provisional dropped)",
        len(predictions), len(outcomes), len(result), n_dropped,
    )
    return result


def detect_label_corrections(
    label_df: pd.DataFrame,
    *,
    id_col: str = "applicant_id",
    timestamp_col: str = "label_timestamp",
) -> pd.DataFrame:
    """Identify rows that are corrections of previously issued labels.

    A correction exists when an applicant_id appears more than once.
    The most recent label_timestamp is the current label; older rows are superseded.

    Args:
        label_df:      Label batch DataFrame.
        id_col:        Column holding the applicant identifier.
        timestamp_col: Column holding when the label was generated (ISO-8601 str or datetime).

    Returns:
        label_df with two new columns:
            is_superseded: True for rows that have been replaced by a newer label.
            is_current:    True for the most recent label per applicant_id.
    """
    df = label_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Rank rows per applicant_id by timestamp, descending
    df["_rank"] = df.groupby(id_col)[timestamp_col].rank(method="first", ascending=False)
    df["is_current"] = df["_rank"] == 1
    df["is_superseded"] = df["_rank"] > 1
    df = df.drop(columns=["_rank"])

    n_corrections = int(df["is_superseded"].sum())
    if n_corrections > 0:
        log.warning("Detected %d superseded labels (label corrections)", n_corrections)
    else:
        log.info("No label corrections detected")

    return df


def backfill_labels(
    label_df: pd.DataFrame,
    *,
    id_col: str = "applicant_id",
    timestamp_col: str = "label_timestamp",
) -> pd.DataFrame:
    """Deduplicate label batch, keeping the most recent label per applicant.

    Equivalent to applying all corrections: for each applicant, keep only the
    current (most recent) label, discarding superseded rows.

    Args:
        label_df:      Label batch DataFrame (may contain correction rows).
        id_col:        Applicant identifier column.
        timestamp_col: Label generation timestamp column.

    Returns:
        Deduplicated DataFrame — one row per applicant_id (most recent label).
        Original row order is not preserved.
    """
    df = label_df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    df_sorted = df.sort_values(timestamp_col, ascending=False)
    deduped = df_sorted.drop_duplicates(subset=[id_col], keep="first").reset_index(drop=True)

    n_dropped = len(df) - len(deduped)
    log.info(
        "Backfill: %d rows → %d rows (%d superseded labels removed)",
        len(df), len(deduped), n_dropped,
    )
    return deduped


@dataclass
class LabelArrivalCurve:
    """Fraction of labels confirmed at T+N days after the observation date.

    Attributes:
        t1, t7, t30, t90, t180: Confirmation fraction at each horizon.
        n_total: Total rows in the batch.
    """

    t1: float
    t7: float
    t30: float
    t90: float
    t180: float
    n_total: int

    @classmethod
    def compute(
        cls,
        label_df: pd.DataFrame,
        *,
        observation_col: str = "observation_date",
        outcome_col: str = "outcome_date",
    ) -> "LabelArrivalCurve":
        """Compute the arrival curve from a label batch.

        Args:
            label_df:        Label DataFrame with observation_date and outcome_date.
            observation_col: Column holding the observation date.
            outcome_col:     Column holding the outcome confirmation date.

        Returns:
            LabelArrivalCurve with fractions at standard horizons.
        """
        df = label_df.copy()
        df["_obs"] = pd.to_datetime(df[observation_col])
        df["_out"] = pd.to_datetime(df[outcome_col])
        df["_lag_days"] = (df["_out"] - df["_obs"]).dt.days

        n = len(df)

        def _frac(days: int) -> float:
            return float((df["_lag_days"] >= days).sum() / n) if n > 0 else 0.0

        curve = cls(
            t1=_frac(1),
            t7=_frac(7),
            t30=_frac(30),
            t90=_frac(90),
            t180=_frac(180),
            n_total=n,
        )

        log.info(
            "Label arrival curve (n=%d): T+1=%.0f%% T+7=%.0f%% T+30=%.0f%% T+90=%.0f%% T+180=%.0f%%",
            n, curve.t1 * 100, curve.t7 * 100, curve.t30 * 100, curve.t90 * 100, curve.t180 * 100,
        )
        return curve

    def to_dict(self) -> dict[str, float | int]:
        return {
            "t1": self.t1, "t7": self.t7, "t30": self.t30,
            "t90": self.t90, "t180": self.t180, "n_total": self.n_total,
        }

    def is_ready_for_training(self, min_t90_fraction: float = 0.90) -> bool:
        """Return True if enough labels have confirmed at the T+90 horizon."""
        return self.t90 >= min_t90_fraction
