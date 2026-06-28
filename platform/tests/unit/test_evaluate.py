"""Unit tests for training/evaluate.py.

Tests verify:
  - compute_metrics: correct keys, AUC=1 for perfect model, ~0.5 for random
  - calibration_error: returns float in [0,1]
  - compute_confusion_details: correct TN/FP/FN/TP, cost calculation

Run:
    cd platform && uv run pytest tests/unit/test_evaluate.py -v
"""
import numpy as np
import pytest

from training.evaluate import calibration_error, compute_confusion_details, compute_metrics

EXPECTED_METRIC_KEYS = {
    "roc_auc", "average_precision", "brier_score", "calibration_error",
    "threshold", "n_samples", "positive_rate",
}


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_classifier_auc_is_1(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.05, 0.1, 0.15, 0.85, 0.9, 0.95])
        m = compute_metrics(y_true, y_prob)
        assert m["roc_auc"] == pytest.approx(1.0, abs=1e-6)

    def test_worst_classifier_auc_is_0(self) -> None:
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.95, 0.9, 0.85, 0.05, 0.1, 0.15])  # inverted
        m = compute_metrics(y_true, y_prob)
        assert m["roc_auc"] == pytest.approx(0.0, abs=1e-6)

    def test_random_classifier_auc_near_half(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.choice([0, 1], 2000, p=[0.78, 0.22])
        y_prob = rng.uniform(0, 1, 2000)
        m = compute_metrics(y_true, y_prob)
        assert 0.40 < m["roc_auc"] < 0.60

    def test_all_expected_keys_present(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
        m = compute_metrics(y_true, y_prob)
        assert EXPECTED_METRIC_KEYS.issubset(set(m.keys()))

    def test_numeric_values_are_floats(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        m = compute_metrics(y_true, y_prob)
        float_keys = EXPECTED_METRIC_KEYS - {"n_samples"}
        for k in float_keys:
            assert isinstance(m[k], float), f"{k} should be float, got {type(m[k])}"

    def test_n_samples_correct(self) -> None:
        n = 100
        rng = np.random.default_rng(0)
        y_true = rng.choice([0, 1], n)
        y_prob = rng.uniform(0, 1, n)
        m = compute_metrics(y_true, y_prob)
        assert m["n_samples"] == n

    def test_positive_rate_correct(self) -> None:
        y_true = np.array([0, 0, 0, 1])  # 25% positive
        y_prob = np.array([0.1, 0.2, 0.3, 0.9])
        m = compute_metrics(y_true, y_prob)
        assert m["positive_rate"] == pytest.approx(0.25, abs=1e-6)

    def test_brier_score_perfect_model(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        m = compute_metrics(y_true, y_prob)
        assert m["brier_score"] == pytest.approx(0.0, abs=1e-6)

    def test_threshold_stored_in_output(self) -> None:
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        m = compute_metrics(y_true, y_prob, threshold=0.42)
        assert m["threshold"] == pytest.approx(0.42)

    def test_metrics_are_bounded(self) -> None:
        rng = np.random.default_rng(99)
        y_true = rng.choice([0, 1], 500, p=[0.78, 0.22])
        y_prob = rng.uniform(0, 1, 500)
        m = compute_metrics(y_true, y_prob)
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["average_precision"] <= 1.0
        assert 0.0 <= m["brier_score"] <= 1.0
        assert 0.0 <= m["calibration_error"] <= 1.0


# ── calibration_error ─────────────────────────────────────────────────────────

class TestCalibrationError:
    def test_returns_float(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7])
        ece = calibration_error(y_true, y_prob)
        assert isinstance(ece, float)

    def test_bounded_between_0_and_1(self) -> None:
        rng = np.random.default_rng(7)
        y_true = rng.choice([0, 1], 500)
        y_prob = rng.uniform(0, 1, 500)
        ece = calibration_error(y_true, y_prob)
        assert 0.0 <= ece <= 1.0

    def test_perfect_calibration_near_zero(self) -> None:
        # Perfectly calibrated: each row where y_prob=p has ~p fraction of y_true=1
        rng = np.random.default_rng(42)
        n = 5000
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.uniform(0, 1, n) < y_prob).astype(int)
        ece = calibration_error(y_true, y_prob, n_bins=10)
        assert ece < 0.1, f"Expected low ECE for well-calibrated model, got {ece:.3f}"


# ── compute_confusion_details ─────────────────────────────────────────────────

class TestComputeConfusionDetails:
    def test_counts_sum_to_n(self) -> None:
        n = 100
        rng = np.random.default_rng(0)
        y_true = rng.choice([0, 1], n, p=[0.78, 0.22])
        y_prob = rng.uniform(0, 1, n)
        d = compute_confusion_details(y_true, y_prob, threshold=0.5)
        total = d["true_negatives"] + d["false_positives"] + d["false_negatives"] + d["true_positives"]
        assert total == n

    def test_perfect_classifier_zero_fp_fn(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        d = compute_confusion_details(y_true, y_prob, threshold=0.5)
        assert d["false_positives"] == 0
        assert d["false_negatives"] == 0

    def test_expected_cost_calculation(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.1, 0.1, 0.9])  # 1 FP, 1 FN
        d = compute_confusion_details(y_true, y_prob, threshold=0.5, fp_cost=2000.0, fn_cost=8000.0)
        assert d["expected_cost"] == pytest.approx(2000.0 + 8000.0)
