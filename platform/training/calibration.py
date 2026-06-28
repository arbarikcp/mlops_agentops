"""Probability calibration for the credit-risk classifier.

Problem: LightGBM raw probabilities are not true probabilities.
A score of 0.80 should mean the customer defaults ~80% of the time.
Without calibration it may mean only 60% — wrong threshold decisions follow.

Two methods:
  sigmoid (Platt scaling) — logistic regression on raw scores.
      Works well with < 1000 calibration samples.
  isotonic — piecewise monotone function.
      More flexible; needs >= 1000 calibration samples (default).

Critical: always calibrate on a held-out calibration set, not training data.
The model has already overfit training scores; they are artificially extreme.

Note on sklearn compatibility: sklearn >= 1.4 removed cv="prefit" from
CalibratedClassifierCV. We use IsotonicRegression and LogisticRegression
directly so the calibration logic is version-independent.

Usage:
    calibrated = fit_calibrator(model, X_cal, y_cal, method="isotonic")
    report = calibration_report(model, calibrated, X_test, y_test)
    report.log_summary()
    df = reliability_data(y_test, calibrated.predict_proba(X_test)[:, 1])
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from training.evaluate import calibration_error

log = logging.getLogger(__name__)


class _CalibratedWrapper:
    """Thin wrapper combining a fitted base model with a post-hoc calibrator.

    Exposes predict_proba so it is a drop-in replacement for the base model.
    """

    def __init__(self, base_model: Any, calibrator: Any, method: str) -> None:
        self._base = base_model
        self._calibrator = calibrator
        self.method = method

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._base.predict_proba(X)[:, 1]
        cal = self._calibrator.predict(raw.reshape(-1, 1)).ravel()
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1 - cal, cal])


@dataclass
class CalibrationReport:
    """ECE and Brier score before and after calibration."""

    method: str
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float

    @property
    def improved(self) -> bool:
        return self.ece_after < self.ece_before

    def log_summary(self) -> None:
        direction = "improved" if self.improved else "worsened"
        log.info(
            "Calibration [%s] %s — ECE: %.4f → %.4f | Brier: %.4f → %.4f",
            self.method,
            direction,
            self.ece_before,
            self.ece_after,
            self.brier_before,
            self.brier_after,
        )


def fit_calibrator(
    base_model: Any,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = "isotonic",
) -> _CalibratedWrapper:
    """Fit a calibrator on a held-out calibration set.

    Args:
        base_model: Fitted estimator with predict_proba (e.g. LGBMClassifier).
        X_cal:      Calibration features. Must NOT overlap with training data.
        y_cal:      Calibration labels (binary 0/1).
        method:     "isotonic" (default, needs >= 1000 samples) or
                    "sigmoid" (Platt scaling, works with < 1000 samples).

    Returns:
        _CalibratedWrapper — has predict_proba, acts as a drop-in replacement.
    """
    raw_probs = base_model.predict_proba(X_cal)[:, 1].reshape(-1, 1)

    if method == "isotonic":
        cal: Any = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw_probs, y_cal)
    elif method == "sigmoid":
        cal = LogisticRegression(C=1.0, max_iter=200)
        cal.fit(raw_probs, y_cal)
    else:
        raise ValueError(f"Unknown calibration method: {method!r}. Use 'isotonic' or 'sigmoid'.")

    log.info("Calibrator fitted [%s] on %d samples", method, len(y_cal))
    return _CalibratedWrapper(base_model=base_model, calibrator=cal, method=method)


def calibration_report(
    base_model: Any,
    calibrated_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str = "isotonic",
) -> CalibrationReport:
    """Compare ECE and Brier score before and after calibration on test set."""
    probs_raw = base_model.predict_proba(X_test)[:, 1]
    probs_cal = calibrated_model.predict_proba(X_test)[:, 1]

    return CalibrationReport(
        method=method,
        ece_before=calibration_error(y_test, probs_raw),
        ece_after=calibration_error(y_test, probs_cal),
        brier_before=float(brier_score_loss(y_test, probs_raw)),
        brier_after=float(brier_score_loss(y_test, probs_cal)),
    )


def reliability_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return reliability diagram data as a DataFrame.

    Each row = one probability bin.
    Columns: mean_predicted, fraction_positive, gap.
    gap > 0 → underconfident; gap < 0 → overconfident.

    Suitable for logging as a CSV artifact to MLflow.
    """
    fraction_pos, mean_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return pd.DataFrame(
        {
            "mean_predicted": mean_pred,
            "fraction_positive": fraction_pos,
            "gap": fraction_pos - mean_pred,
        }
    )
