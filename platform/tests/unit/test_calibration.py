"""Tests for training/calibration.py."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from training.calibration import (
    CalibrationReport,
    calibration_report,
    fit_calibrator,
    reliability_data,
)


def _fitted_logistic(n: int = 600, seed: int = 42):
    """Return a fitted LogisticRegression and matching arrays for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    model = LogisticRegression(random_state=42, max_iter=300)
    model.fit(X, y)
    return model, X, y


class TestFitCalibrator:
    def test_returns_predict_proba(self):
        model, X, y = _fitted_logistic()
        cal = fit_calibrator(model, X, y, method="sigmoid")
        assert hasattr(cal, "predict_proba")

    def test_isotonic_shape(self):
        model, X, y = _fitted_logistic()
        cal = fit_calibrator(model, X, y, method="isotonic")
        probs = cal.predict_proba(X)[:, 1]
        assert probs.shape == (len(X),)

    def test_sigmoid_probs_in_unit_interval(self):
        model, X, y = _fitted_logistic()
        cal = fit_calibrator(model, X, y, method="sigmoid")
        probs = cal.predict_proba(X)[:, 1]
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0

    def test_isotonic_probs_in_unit_interval(self):
        model, X, y = _fitted_logistic()
        cal = fit_calibrator(model, X, y, method="isotonic")
        probs = cal.predict_proba(X)[:, 1]
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0


class TestCalibrationReport:
    def test_report_type(self):
        model, X, y = _fitted_logistic()
        cal = fit_calibrator(model, X, y, method="sigmoid")
        report = calibration_report(model, cal, X, y, method="sigmoid")
        assert isinstance(report, CalibrationReport)

    def test_ece_values_in_range(self):
        model, X, y = _fitted_logistic()
        cal = fit_calibrator(model, X, y, method="sigmoid")
        report = calibration_report(model, cal, X, y)
        assert 0.0 <= report.ece_before <= 1.0
        assert 0.0 <= report.ece_after <= 1.0

    def test_brier_values_non_negative(self):
        model, X, y = _fitted_logistic()
        cal = fit_calibrator(model, X, y, method="sigmoid")
        report = calibration_report(model, cal, X, y)
        assert report.brier_before >= 0.0
        assert report.brier_after >= 0.0

    def test_improved_flag_is_bool(self):
        model, X, y = _fitted_logistic()
        cal = fit_calibrator(model, X, y, method="sigmoid")
        report = calibration_report(model, cal, X, y)
        assert isinstance(report.improved, bool)

    def test_logistic_regression_already_calibrated(self):
        """LogisticRegression is inherently calibrated — ECE should be low."""
        model, X, y = _fitted_logistic(n=2000)
        cal = fit_calibrator(model, X, y, method="sigmoid")
        report = calibration_report(model, cal, X, y)
        assert report.ece_before < 0.15


class TestReliabilityData:
    def test_expected_columns(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 1000)
        y_prob = np.clip(rng.uniform(0, 1, 1000), 0.01, 0.99)
        df = reliability_data(y_true, y_prob, n_bins=5)
        assert set(df.columns) == {"mean_predicted", "fraction_positive", "gap"}

    def test_gap_definition(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 1000)
        y_prob = np.clip(rng.uniform(0, 1, 1000), 0.01, 0.99)
        df = reliability_data(y_true, y_prob, n_bins=5)
        expected_gap = df["fraction_positive"] - df["mean_predicted"]
        np.testing.assert_allclose(df["gap"].to_numpy(), expected_gap.to_numpy(), atol=1e-10)

    def test_well_calibrated_small_gap(self):
        """Randomly sampled probabilities give small average gap."""
        rng = np.random.default_rng(2)
        probs = rng.uniform(0.0, 1.0, 20_000)
        y_true = (rng.uniform(0, 1, 20_000) < probs).astype(int)
        df = reliability_data(y_true, probs, n_bins=10)
        assert float(df["gap"].abs().mean()) < 0.07
