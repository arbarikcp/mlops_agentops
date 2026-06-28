"""Three-class decision routing for the credit-risk classifier.

Binary decisions (approve/decline) force the model to commit on every case.
Borderline cases are genuinely uncertain — routing them to a human reviewer
at ~$100/review is cheap insurance against an $8,000 default or $2,000 missed LTV.

Decision routing:
    score < approve_threshold            → APPROVE  (very likely good customer)
    approve_threshold ≤ score < decline  → REVIEW   (uncertain → human queue)
    score ≥ decline_threshold            → DECLINE  (very likely bad customer)

ThresholdBand is immutable (frozen dataclass). Safe to store in registry, pass
around, or use as a dict key. The band itself is a versioned serving artefact
— log it to MLflow alongside the model.

Usage:
    band = ThresholdBand(approve=0.15, decline=0.45)
    decision = band.decide(score=0.34)      # → "review"
    decisions = band.route_batch(y_prob)    # vectorised
    stats = band.routing_stats(y_prob)      # pct per class
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DECISION_APPROVE: str = "approve"
DECISION_REVIEW: str = "review"
DECISION_DECLINE: str = "decline"


@dataclass(frozen=True)
class ThresholdBand:
    """Immutable 3-class routing boundary.

    Invariant: approve < decline (a zero-width band has no review region).
    """

    approve: float  # score strictly below this → approve
    decline: float  # score at or above this → decline

    def __post_init__(self) -> None:
        if self.approve >= self.decline:
            raise ValueError(
                f"approve threshold ({self.approve}) must be strictly less than "
                f"decline threshold ({self.decline})"
            )

    def decide(self, score: float) -> str:
        """Route a single calibrated probability score."""
        if score < self.approve:
            return DECISION_APPROVE
        if score >= self.decline:
            return DECISION_DECLINE
        return DECISION_REVIEW

    def route_batch(self, scores: np.ndarray) -> np.ndarray:
        """Route a batch of scores. Returns array of decision strings (vectorised)."""
        return np.where(
            scores < self.approve,
            DECISION_APPROVE,
            np.where(scores >= self.decline, DECISION_DECLINE, DECISION_REVIEW),
        )

    def routing_stats(self, scores: np.ndarray) -> dict[str, float]:
        """Return the fraction of predictions routed to each decision class."""
        decisions = self.route_batch(scores)
        n = len(scores)
        return {
            "pct_approve": float((decisions == DECISION_APPROVE).sum() / n),
            "pct_review": float((decisions == DECISION_REVIEW).sum() / n),
            "pct_decline": float((decisions == DECISION_DECLINE).sum() / n),
            "review_band_width": float(self.decline - self.approve),
        }


def find_review_band(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    approve_threshold: float,
    decline_threshold: float,
) -> pd.DataFrame:
    """Compute actual positive rate per decision class at given thresholds.

    Returns a DataFrame with columns:
        decision, n, pct, positive_rate

    positive_rate in the approve bucket = fraction of approved customers who
    actually default. This is the key business metric: should be very low.
    """
    band = ThresholdBand(approve=approve_threshold, decline=decline_threshold)
    decisions = band.route_batch(y_prob)

    rows = []
    for label in (DECISION_APPROVE, DECISION_REVIEW, DECISION_DECLINE):
        mask = decisions == label
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "decision": label,
                "n": n,
                "pct": float(n / len(y_true)),
                "positive_rate": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def calibrate_band_for_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fp_cost: float = 2_000.0,
    fn_cost: float = 8_000.0,
    target_review_pct: float = 0.15,
) -> ThresholdBand:
    """Derive approve/decline thresholds centred on the cost-optimal threshold.

    Centres the review band on the threshold from find_cost_optimal_threshold,
    then widens it symmetrically until ~target_review_pct of predictions fall inside.

    Args:
        target_review_pct: target fraction to route to human review (default 15%).

    Returns:
        ThresholdBand with approve and decline thresholds.
    """
    from training.threshold import find_cost_optimal_threshold

    result = find_cost_optimal_threshold(y_true, y_prob, fp_cost, fn_cost)
    centre = result.threshold
    half = target_review_pct / 2.0

    approve_t = float(np.clip(centre - half, 0.05, 0.45))
    decline_t = float(np.clip(centre + half, approve_t + 0.05, 0.95))

    band = ThresholdBand(approve=round(approve_t, 3), decline=round(decline_t, 3))
    stats = band.routing_stats(y_prob)
    log.info(
        "Band [approve<%.3f, decline>=%.3f] → %.1f%% approve, %.1f%% review, %.1f%% decline",
        band.approve,
        band.decline,
        stats["pct_approve"] * 100,
        stats["pct_review"] * 100,
        stats["pct_decline"] * 100,
    )
    return band
