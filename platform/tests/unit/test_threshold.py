"""Tests for training/threshold.py."""
from __future__ import annotations

import numpy as np
import pytest

from training.threshold import (
    ThresholdResult,
    find_cost_optimal_threshold,
    threshold_sweep,
)


def _separable_case(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Positives score 0.6–1.0, negatives score 0.0–0.4 — clear separation."""
    rng = np.random.default_rng(seed)
    y_true = np.array([1] * 100 + [0] * 400)
    y_prob = np.concatenate([
        np.clip(rng.normal(0.80, 0.06, 100), 0.01, 0.99),
        np.clip(rng.normal(0.20, 0.06, 400), 0.01, 0.99),
    ])
    return y_true, y_prob


def _noisy_case(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping distributions — harder to separate."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, 500)
    y_prob = np.clip(rng.beta(2, 5, 500), 0.01, 0.99)
    return y_true, y_prob


class TestFindCostOptimalThreshold:
    def test_returns_threshold_result(self):
        y_true, y_prob = _separable_case()
        result = find_cost_optimal_threshold(y_true, y_prob)
        assert isinstance(result, ThresholdResult)

    def test_threshold_in_valid_range(self):
        y_true, y_prob = _separable_case()
        result = find_cost_optimal_threshold(y_true, y_prob)
        assert 0.0 < result.threshold < 1.0

    def test_confusion_matrix_sums_to_n(self):
        y_true, y_prob = _separable_case()
        result = find_cost_optimal_threshold(y_true, y_prob)
        total = (
            result.true_positives
            + result.false_positives
            + result.false_negatives
            + result.true_negatives
        )
        assert total == len(y_true)

    def test_precision_recall_in_range(self):
        y_true, y_prob = _separable_case()
        result = find_cost_optimal_threshold(y_true, y_prob)
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0

    def test_high_fn_cost_lowers_threshold(self):
        """FN 10x more expensive → model should be more conservative (lower threshold)."""
        y_true, y_prob = _noisy_case()
        result_symmetric = find_cost_optimal_threshold(y_true, y_prob, fp_cost=5000, fn_cost=5000)
        result_fn_heavy = find_cost_optimal_threshold(y_true, y_prob, fp_cost=1000, fn_cost=10000)
        assert result_fn_heavy.threshold <= result_symmetric.threshold + 0.15

    def test_total_cost_matches_manual(self):
        y_true, y_prob = _separable_case()
        result = find_cost_optimal_threshold(y_true, y_prob, fp_cost=2000, fn_cost=8000)
        manual_cost = result.false_positives * 2000 + result.false_negatives * 8000
        assert abs(result.total_cost - manual_cost) < 0.01

    def test_expected_cost_per_sample(self):
        y_true, y_prob = _separable_case()
        result = find_cost_optimal_threshold(y_true, y_prob)
        expected = result.total_cost / len(y_true)
        assert abs(result.expected_cost_per_sample - expected) < 0.01


class TestThresholdSweep:
    def test_returns_correct_length(self):
        y_true, y_prob = _separable_case()
        df = threshold_sweep(y_true, y_prob, n_points=50)
        assert len(df) == 50

    def test_expected_columns(self):
        y_true, y_prob = _separable_case()
        df = threshold_sweep(y_true, y_prob, n_points=10)
        assert set(df.columns) >= {"threshold", "total_cost", "fp", "fn", "precision", "recall"}

    def test_total_cost_non_negative(self):
        y_true, y_prob = _separable_case()
        df = threshold_sweep(y_true, y_prob, n_points=50)
        assert (df["total_cost"] >= 0).all()

    def test_minimum_in_sweep_matches_optimal(self):
        y_true, y_prob = _separable_case()
        df = threshold_sweep(y_true, y_prob, n_points=200)
        result = find_cost_optimal_threshold(y_true, y_prob, n_points=200)
        sweep_min = df["total_cost"].min()
        assert abs(result.total_cost - sweep_min) < 1.0

    def test_thresholds_monotonic(self):
        y_true, y_prob = _separable_case()
        df = threshold_sweep(y_true, y_prob, n_points=20)
        assert df["threshold"].is_monotonic_increasing
