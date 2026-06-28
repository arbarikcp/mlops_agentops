"""Tests for training/decision.py."""
from __future__ import annotations

import numpy as np
import pytest

from training.decision import (
    DECISION_APPROVE,
    DECISION_DECLINE,
    DECISION_REVIEW,
    ThresholdBand,
    find_review_band,
)


class TestThresholdBand:
    def test_decide_approve(self):
        band = ThresholdBand(approve=0.30, decline=0.70)
        assert band.decide(0.00) == DECISION_APPROVE
        assert band.decide(0.10) == DECISION_APPROVE
        assert band.decide(0.29) == DECISION_APPROVE

    def test_decide_review(self):
        band = ThresholdBand(approve=0.30, decline=0.70)
        assert band.decide(0.30) == DECISION_REVIEW
        assert band.decide(0.50) == DECISION_REVIEW
        assert band.decide(0.699) == DECISION_REVIEW

    def test_decide_decline(self):
        band = ThresholdBand(approve=0.30, decline=0.70)
        assert band.decide(0.70) == DECISION_DECLINE
        assert band.decide(0.90) == DECISION_DECLINE
        assert band.decide(1.00) == DECISION_DECLINE

    def test_invalid_band_approve_gt_decline_raises(self):
        with pytest.raises(ValueError, match="must be strictly less than"):
            ThresholdBand(approve=0.80, decline=0.30)

    def test_invalid_band_equal_raises(self):
        with pytest.raises(ValueError):
            ThresholdBand(approve=0.50, decline=0.50)

    def test_route_batch_all_three_classes(self):
        band = ThresholdBand(approve=0.30, decline=0.70)
        scores = np.array([0.10, 0.50, 0.90])
        decisions = band.route_batch(scores)
        assert decisions[0] == DECISION_APPROVE
        assert decisions[1] == DECISION_REVIEW
        assert decisions[2] == DECISION_DECLINE

    def test_route_batch_shape_preserved(self):
        band = ThresholdBand(approve=0.20, decline=0.80)
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 500)
        decisions = band.route_batch(scores)
        assert decisions.shape == (500,)

    def test_routing_stats_sum_to_one(self):
        band = ThresholdBand(approve=0.30, decline=0.70)
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 10_000)
        stats = band.routing_stats(scores)
        total = stats["pct_approve"] + stats["pct_review"] + stats["pct_decline"]
        assert abs(total - 1.0) < 1e-9

    def test_routing_stats_band_width(self):
        band = ThresholdBand(approve=0.20, decline=0.80)
        scores = np.array([0.5])
        stats = band.routing_stats(scores)
        assert stats["review_band_width"] == pytest.approx(0.60, abs=1e-9)

    def test_narrow_band_few_reviews(self):
        """Very narrow band should route almost nothing to review."""
        band = ThresholdBand(approve=0.49, decline=0.51)
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 10_000)
        stats = band.routing_stats(scores)
        assert stats["pct_review"] < 0.05

    def test_wide_band_many_reviews(self):
        """Very wide band should route most predictions to review."""
        band = ThresholdBand(approve=0.01, decline=0.99)
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 10_000)
        stats = band.routing_stats(scores)
        assert stats["pct_review"] > 0.90

    def test_frozen_immutable(self):
        band = ThresholdBand(approve=0.3, decline=0.7)
        with pytest.raises((AttributeError, TypeError)):
            band.approve = 0.4  # type: ignore[misc]


class TestFindReviewBand:
    def _data(self, n: int = 1000, seed: int = 42):
        rng = np.random.default_rng(seed)
        y_prob = rng.uniform(0.01, 0.99, n)
        y_true = (y_prob + rng.normal(0, 0.1, n) > 0.5).astype(int)
        return y_true, y_prob

    def test_returns_three_classes(self):
        y_true, y_prob = self._data()
        df = find_review_band(y_true, y_prob, 0.30, 0.70)
        assert set(df["decision"]) == {DECISION_APPROVE, DECISION_REVIEW, DECISION_DECLINE}

    def test_pct_sums_to_one(self):
        y_true, y_prob = self._data()
        df = find_review_band(y_true, y_prob, 0.30, 0.70)
        assert abs(df["pct"].sum() - 1.0) < 1e-6

    def test_positive_rate_in_range(self):
        y_true, y_prob = self._data()
        df = find_review_band(y_true, y_prob, 0.30, 0.70)
        assert (df["positive_rate"].between(0.0, 1.0)).all()

    def test_approve_lower_default_rate_than_decline(self):
        """Approve bucket should have a lower actual default rate than decline bucket."""
        y_true, y_prob = self._data(n=5000)
        df = find_review_band(y_true, y_prob, 0.25, 0.75)
        approve_rate = float(df.loc[df["decision"] == DECISION_APPROVE, "positive_rate"].iloc[0])
        decline_rate = float(df.loc[df["decision"] == DECISION_DECLINE, "positive_rate"].iloc[0])
        assert approve_rate < decline_rate
