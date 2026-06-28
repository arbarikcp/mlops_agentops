"""Model evaluation metrics for the credit-risk classifier.

Why these metrics (not just accuracy):
  - roc_auc:            Threshold-independent; measures ranking ability.
  - average_precision:  Better than AUC for imbalanced classes (22% positive).
  - brier_score:        Proper scoring rule; rewards well-calibrated probabilities.
  - calibration_error:  Expected Calibration Error (ECE); are probabilities trustworthy?

The ECE matters here because:
  1. Our serving code will threshold on the probability (Day 16: cost-sensitive threshold).
  2. A poorly calibrated model (e.g. always outputs 0.9 when truth is 0.6) will produce
     wrong approve/review/decline decisions regardless of AUC.

Usage:
    from training.evaluate import compute_metrics
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    # → {"roc_auc": 0.78, "average_precision": 0.55, ...}

Debugging calibration issues:
    - Plot reliability diagram: from sklearn.calibration import CalibrationDisplay
    - ECE > 0.1 usually means the model needs Platt scaling or isotonic regression.
    - See Day 15 (Phase 2) for calibration techniques.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Partitions predictions into n_bins by predicted probability.
    In each bin, computes |mean_predicted - fraction_positive|.
    ECE = mean over bins. Perfect calibration → ECE = 0.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return float(np.mean(np.abs(fraction_pos - mean_pred)))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute the full evaluation metric suite.

    Args:
        y_true:    Ground truth binary labels (0/1).
        y_prob:    Predicted probabilities for the positive class.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dict with keys: roc_auc, average_precision, brier_score,
        calibration_error, threshold, n_samples, positive_rate.
    """
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "calibration_error": calibration_error(y_true, y_prob),
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }

    log.info(
        "Metrics — AUC: %.4f | AP: %.4f | Brier: %.4f | ECE: %.4f",
        metrics["roc_auc"],
        metrics["average_precision"],
        metrics["brier_score"],
        metrics["calibration_error"],
    )
    return metrics


def compute_confusion_details(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fp_cost: float = 2_000.0,
    fn_cost: float = 8_000.0,
) -> dict[str, float | int]:
    """Compute confusion matrix details and expected cost at a given threshold.

    Cost model from Day 4 system design:
        FP cost (decline good customer) = $2,000 lost LTV
        FN cost (approve bad customer)  = $8,000 average default loss
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "expected_cost": float(fp * fp_cost + fn * fn_cost),
    }
