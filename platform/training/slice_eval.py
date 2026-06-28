"""Slice-level evaluation and OOD detection for the credit-risk classifier.

Problem: aggregate AUC=0.78 can hide poor performance on important subgroups.
A model trained on imbalanced demographic data might work well overall but
fail on protected groups — which is both a fairness and a business risk.

Slice evaluation:
    evaluate_slices() computes per-slice AUC, AP, ECE for each category.
    worst_slices() surfaces the N weakest segments.
    slice_gap_report() flags segments with > threshold gap from aggregate.

OOD detection:
    IsolationForest on training features. Production batches that look
    unlike training data are flagged before they produce silent failures.

Usage:
    slices = evaluate_slices(X_test, y_test, y_prob, ["EDUCATION", "SEX"])
    worst = worst_slices(slices, metric="roc_auc", n=5)
    gaps = slice_gap_report(slices, overall_metrics, warn_threshold=0.05)

    detector = fit_ood_detector(X_train.to_numpy())
    report = ood_report(detector, X_prod.to_numpy())
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from training.evaluate import calibration_error

log = logging.getLogger(__name__)

SLICE_COLS: list[str] = ["EDUCATION", "MARRIAGE", "SEX"]
MIN_SLICE_SIZE: int = 50  # fewer samples → unreliable AUC


def evaluate_slices(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    slice_cols: Sequence[str] | None = None,
    min_size: int = MIN_SLICE_SIZE,
) -> pd.DataFrame:
    """Compute per-slice metrics for each unique value of each slice column.

    Args:
        X:          Feature DataFrame (must include slice columns).
        y_true:     Ground truth binary labels.
        y_prob:     Predicted probabilities (calibrated).
        slice_cols: Columns to slice on. Defaults to EDUCATION, MARRIAGE, SEX.
        min_size:   Skip slices with fewer than this many samples.

    Returns:
        DataFrame with columns: slice_col, slice_val, n, roc_auc,
        average_precision, calibration_error, positive_rate.
    """
    if slice_cols is None:
        slice_cols = [c for c in SLICE_COLS if c in X.columns]

    rows = []
    for col in slice_cols:
        if col not in X.columns:
            log.warning("Slice column '%s' not in DataFrame — skipped", col)
            continue
        for val in sorted(X[col].unique()):
            mask = (X[col].to_numpy() == val)
            n = int(mask.sum())
            if n < min_size:
                log.debug("Skipping %s=%s (n=%d < min=%d)", col, val, n, min_size)
                continue
            y_t = y_true[mask]
            y_p = y_prob[mask]
            if len(np.unique(y_t)) < 2:
                log.debug("Skipping %s=%s — single class present", col, val)
                continue
            rows.append(
                {
                    "slice_col": col,
                    "slice_val": str(val),
                    "n": n,
                    "roc_auc": float(roc_auc_score(y_t, y_p)),
                    "average_precision": float(average_precision_score(y_t, y_p)),
                    "calibration_error": float(calibration_error(y_t, y_p)),
                    "positive_rate": float(y_t.mean()),
                }
            )

    return pd.DataFrame(rows)


def worst_slices(
    slice_df: pd.DataFrame,
    metric: str = "roc_auc",
    n: int = 5,
) -> pd.DataFrame:
    """Return the N slices with the lowest value of metric."""
    return (
        slice_df
        .nsmallest(n, metric)[["slice_col", "slice_val", "n", metric, "positive_rate"]]
        .reset_index(drop=True)
    )


def slice_gap_report(
    slice_df: pd.DataFrame,
    overall_metrics: dict[str, float],
    metric: str = "roc_auc",
    warn_threshold: float = 0.05,
) -> pd.DataFrame:
    """Flag slices where metric is more than warn_threshold below aggregate.

    Args:
        overall_metrics: dict from compute_metrics() — the aggregate values.
        warn_threshold:  Alert if slice metric < aggregate - warn_threshold.

    Returns:
        DataFrame with added columns: overall, gap, flag.
        Sorted by gap descending (worst slices first).
    """
    overall = overall_metrics.get(metric, float("nan"))
    df = slice_df.copy()
    df["overall"] = overall
    df["gap"] = overall - df[metric]
    df["flag"] = df["gap"] > warn_threshold
    return df.sort_values("gap", ascending=False).reset_index(drop=True)


def fit_ood_detector(
    X_train: np.ndarray,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """Fit an Isolation Forest OOD detector on training data.

    The fitted detector scores new samples by how easy they are to isolate
    via random partitioning. Scores from decision_function():
        positive → in-distribution
        negative → OOD / anomalous

    Args:
        contamination: expected fraction of anomalies in training data.
                       Affects the decision boundary (score=0 threshold).
    """
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(X_train)
    log.info(
        "OOD detector fitted on %d training samples (contamination=%.2f)",
        len(X_train),
        contamination,
    )
    return clf


def ood_report(
    detector: IsolationForest,
    X: np.ndarray,
    label: str = "dataset",
) -> dict[str, float]:
    """Compute OOD statistics for a dataset.

    Returns:
        ood_fraction:  fraction of samples with score < 0 (flagged as OOD).
        mean_score:    mean anomaly score (higher = more in-distribution).
        p5_score:      5th percentile score (tail of distribution).
        n_samples:     number of samples scored.
    """
    scores = detector.decision_function(X)
    report = {
        "ood_fraction": float((scores < 0).mean()),
        "mean_score": float(scores.mean()),
        "p5_score": float(np.percentile(scores, 5)),
        "n_samples": int(len(X)),
    }
    log.info(
        "[%s] OOD fraction: %.3f | mean score: %.4f | p5 score: %.4f",
        label,
        report["ood_fraction"],
        report["mean_score"],
        report["p5_score"],
    )
    return report
