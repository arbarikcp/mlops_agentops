"""Cost-sensitive threshold selection for the credit-risk classifier.

Problem: default threshold 0.5 minimises neither cost nor business risk.
This domain (Day 4 system design):
    FP cost (decline good customer)  = $2,000 lost LTV
    FN cost (approve bad customer)   = $8,000 average default loss

With FN 4x more expensive than FP, the optimal threshold shifts downward
(be more conservative — decline borderline cases rather than approve them).
Closed-form estimate: t* ≈ C_FP / (C_FP + C_FN) = 2000 / 10000 = 0.20.

Usage:
    result = find_cost_optimal_threshold(y_true, y_prob)
    result.log_summary()

    sweep_df = threshold_sweep(y_true, y_prob)
    # sweep_df.to_csv("metrics/threshold_sweep.csv") then log to MLflow
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

log = logging.getLogger(__name__)

DEFAULT_FP_COST: float = 2_000.0
DEFAULT_FN_COST: float = 8_000.0


@dataclass
class ThresholdResult:
    """Optimal threshold and associated confusion-matrix statistics."""

    threshold: float
    total_cost: float
    expected_cost_per_sample: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float

    def log_summary(self) -> None:
        log.info(
            "Optimal threshold: %.3f | Cost/sample: $%.2f | "
            "Precision: %.3f | Recall: %.3f | TP=%d FP=%d FN=%d TN=%d",
            self.threshold,
            self.expected_cost_per_sample,
            self.precision,
            self.recall,
            self.true_positives,
            self.false_positives,
            self.false_negatives,
            self.true_negatives,
        )


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    n_points: int = 200,
) -> ThresholdResult:
    """Find threshold that minimises total expected cost.

    Sweeps [0.01, 0.99] with n_points steps, evaluates total cost at each,
    returns the threshold with the minimum.

    Args:
        y_true:    Ground truth labels (binary 0/1).
        y_prob:    Calibrated predicted probabilities.
        fp_cost:   Cost per false positive (declined good customer).
        fn_cost:   Cost per false negative (approved bad customer).
        n_points:  Resolution of the threshold sweep.

    Returns:
        ThresholdResult with the optimal threshold and confusion-matrix stats.
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    best_cost = float("inf")
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = float(fp) * fp_cost + float(fn) * fn_cost
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)

    y_pred_best = (y_prob >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_best).ravel()
    n = len(y_true)

    return ThresholdResult(
        threshold=best_t,
        total_cost=best_cost,
        expected_cost_per_sample=best_cost / n,
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
        precision=float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        recall=float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
    )


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    n_points: int = 200,
) -> pd.DataFrame:
    """Return the full cost curve as a DataFrame.

    Columns: threshold, total_cost, fp, fn, precision, recall.
    Log as a CSV artifact to MLflow to plot the cost curve.
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": float(t),
                "total_cost": float(int(fp) * fp_cost + int(fn) * fn_cost),
                "fp": int(fp),
                "fn": int(fn),
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)
