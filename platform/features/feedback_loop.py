"""Label Feedback Loop: join delayed ground truth, recompute metrics, trigger retraining.

Implements the 8-step closed feedback loop for credit risk:
  PREDICT → DECIDE → LOG → AWAIT_OUTCOME → JOIN_LABEL → RECOMPUTE → TRIGGER → APPROVE

Classes:
  LoopPhase          — 8 phases of the feedback loop
  PredictionRecord   — one model prediction with features and decision
  OutcomeRecord      — ground truth label with arrival delay
  LabeledExample     — joined prediction + outcome ready for retraining
  RetrainTrigger     — decision on whether to retrain the model
  LoopResult         — full result of one feedback loop tick
  GroundTruthJoiner  — joins predictions with outcomes using delay filter
  MetricRecomputer   — computes AUC, approval_rate on labeled examples
  RetrainDecider     — decides whether to trigger a retrain
  LabelFeedbackLoop  — orchestrates all 8 steps in `.tick()`

See: docs/phase6/day44_label_feedback.md
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Loop Phase ────────────────────────────────────────────────────────────────

class LoopPhase(str, Enum):
    PREDICT = "predict"
    DECIDE = "decide"
    LOG = "log"
    AWAIT_OUTCOME = "await_outcome"
    JOIN_LABEL = "join_label"
    RECOMPUTE = "recompute"
    TRIGGER = "trigger"
    APPROVE = "approve"


# ── Records ───────────────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    """One model prediction with its decision context.

    Attributes:
        prediction_id: Unique identifier (links to OutcomeRecord).
        entity_key:    Entity that was scored (e.g. "customer_id:C1").
        score:         Model probability output [0, 1].
        decision:      "approve" / "review" / "decline".
        prediction_ts: When the prediction was made (UTC).
        features:      Feature values used at prediction time (for audit).
    """

    prediction_id: str
    entity_key: str
    score: float
    decision: str
    prediction_ts: datetime
    features: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prediction_id:
            raise ValueError("PredictionRecord.prediction_id must not be empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"PredictionRecord.score must be in [0,1], got {self.score}")
        if self.decision not in ("approve", "review", "decline"):
            raise ValueError(f"PredictionRecord.decision must be approve/review/decline, got '{self.decision}'")


@dataclass
class OutcomeRecord:
    """Ground truth label for a past prediction.

    Attributes:
        prediction_id:   Links to PredictionRecord.
        actual_outcome:  1 = default, 0 = paid (or None if still awaited).
        outcome_ts:      When the outcome was confirmed (UTC).
        delay_days:      Days between prediction and outcome confirmation.
    """

    prediction_id: str
    actual_outcome: int | None
    outcome_ts: datetime
    delay_days: float = 0.0

    def __post_init__(self) -> None:
        if not self.prediction_id:
            raise ValueError("OutcomeRecord.prediction_id must not be empty")
        if self.actual_outcome is not None and self.actual_outcome not in (0, 1):
            raise ValueError(f"OutcomeRecord.actual_outcome must be 0, 1, or None. Got {self.actual_outcome}")
        if self.delay_days < 0:
            raise ValueError(f"OutcomeRecord.delay_days must be >= 0, got {self.delay_days}")


@dataclass
class LabeledExample:
    """A fully joined prediction + outcome, ready for retraining.

    Attributes:
        prediction_id: Shared key with PredictionRecord and OutcomeRecord.
        entity_key:    Entity identifier.
        score:         Original model score.
        label:         Confirmed ground truth (0 or 1).
        delay_days:    How long it took for the label to arrive.
        decision:      Original routing decision.
        features:      Feature snapshot at prediction time.
    """

    prediction_id: str
    entity_key: str
    score: float
    label: int
    delay_days: float
    decision: str
    features: dict[str, Any] = field(default_factory=dict)


# ── Retrain Trigger ───────────────────────────────────────────────────────────

@dataclass
class RetrainTrigger:
    """Decision on whether to trigger model retraining.

    Attributes:
        triggered:    True if retraining should be initiated.
        reason:       Human-readable explanation.
        metric_delta: AUC change that triggered (or failed to trigger) this.
        n_new_labels: Number of new labeled examples in this batch.
    """

    triggered: bool
    reason: str
    metric_delta: float
    n_new_labels: int


# ── Loop Result ───────────────────────────────────────────────────────────────

@dataclass
class LoopResult:
    """Result of one complete feedback loop tick.

    Attributes:
        phase_reached:    The last phase completed in this tick.
        labeled_examples: Joined examples with confirmed labels.
        current_metrics:  Recomputed metrics on the new labeled set.
        trigger:          Whether a retrain was triggered.
        n_provisional:    Examples not yet confirmed (delay not met).
    """

    phase_reached: LoopPhase
    labeled_examples: list[LabeledExample]
    current_metrics: dict[str, float]
    trigger: RetrainTrigger
    n_provisional: int = 0


# ── Ground Truth Joiner ───────────────────────────────────────────────────────

class GroundTruthJoiner:
    """Joins prediction records with outcome records using a minimum delay filter.

    Args:
        min_delay_days: Only include outcomes where delay_days >= this value.
                        Ensures labels are stable (past the arrival curve inflection).
    """

    def __init__(self, min_delay_days: float = 30.0) -> None:
        if min_delay_days < 0:
            raise ValueError("min_delay_days must be >= 0")
        self.min_delay_days = min_delay_days

    def join(
        self,
        predictions: list[PredictionRecord],
        outcomes: list[OutcomeRecord],
    ) -> tuple[list[LabeledExample], int]:
        """Join predictions with outcomes, filtering out provisional labels.

        Args:
            predictions: All predictions to join.
            outcomes:    All available outcomes.

        Returns:
            (confirmed_examples, n_provisional) tuple.
            n_provisional = outcomes excluded due to insufficient delay.
        """
        outcome_map = {o.prediction_id: o for o in outcomes}
        confirmed: list[LabeledExample] = []
        n_provisional = 0

        for pred in predictions:
            outcome = outcome_map.get(pred.prediction_id)
            if outcome is None:
                n_provisional += 1  # no outcome yet
                continue
            if outcome.actual_outcome is None:
                n_provisional += 1  # outcome not confirmed
                continue
            if outcome.delay_days < self.min_delay_days:
                n_provisional += 1  # too recent — label may still change
                continue

            confirmed.append(LabeledExample(
                prediction_id=pred.prediction_id,
                entity_key=pred.entity_key,
                score=pred.score,
                label=outcome.actual_outcome,
                delay_days=outcome.delay_days,
                decision=pred.decision,
                features=pred.features,
            ))

        return confirmed, n_provisional


# ── Metric Recomputer ─────────────────────────────────────────────────────────

class MetricRecomputer:
    """Recomputes model quality metrics on a labeled example set.

    Computes AUC, approval rate, default rate among approved, and label balance.
    Uses a simple trapezoidal AUC implementation (no sklearn required).
    """

    def recompute(self, examples: list[LabeledExample]) -> dict[str, float]:
        """Compute metrics from a list of LabeledExamples.

        Args:
            examples: Joined labeled examples.

        Returns:
            Dict of metric_name → float value.
            Returns NaN values if examples list is too small for meaningful metrics.
        """
        if len(examples) < 10:
            return {
                "auc": float("nan"),
                "approval_rate": float("nan"),
                "default_rate_approved": float("nan"),
                "label_positive_rate": float("nan"),
                "n_labeled": float(len(examples)),
            }

        labels = [e.label for e in examples]
        scores = [e.score for e in examples]

        auc = self._compute_auc(labels, scores)
        approved = [e for e in examples if e.decision == "approve"]
        approval_rate = len(approved) / len(examples)
        default_rate = (
            sum(1 for e in approved if e.label == 1) / len(approved)
            if approved
            else float("nan")
        )
        pos_rate = sum(labels) / len(labels)

        return {
            "auc": auc,
            "approval_rate": approval_rate,
            "default_rate_approved": default_rate,
            "label_positive_rate": pos_rate,
            "n_labeled": float(len(examples)),
        }

    @staticmethod
    def _compute_auc(labels: list[int], scores: list[float]) -> float:
        """Compute ROC-AUC using the Mann-Whitney U statistic."""
        if len(set(labels)) < 2:
            return float("nan")

        pos = [s for s, l in zip(scores, labels) if l == 1]
        neg = [s for s, l in zip(scores, labels) if l == 0]
        if not pos or not neg:
            return float("nan")

        # U statistic: count pairs where positive score > negative score
        n_correct = sum(1 for p in pos for n in neg if p > n)
        n_tie = sum(1 for p in pos for n in neg if p == n)
        return (n_correct + 0.5 * n_tie) / (len(pos) * len(neg))


# ── Retrain Decider ───────────────────────────────────────────────────────────

class RetrainDecider:
    """Decides whether to trigger retraining based on metric change and batch size.

    Two conditions must both be true to trigger:
    1. n_new_labels >= min_batch_size (enough data for reliable signal)
    2. |auc_delta| >= auc_drift_threshold (meaningful quality change)

    Args:
        min_batch_size:      Minimum new labels required to trigger.
        auc_drift_threshold: Minimum |AUC change| to trigger.
    """

    def __init__(
        self,
        min_batch_size: int = 100,
        auc_drift_threshold: float = 0.02,
    ) -> None:
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        if auc_drift_threshold < 0:
            raise ValueError("auc_drift_threshold must be >= 0")
        self.min_batch_size = min_batch_size
        self.auc_drift_threshold = auc_drift_threshold

    def decide(
        self,
        current_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
        n_new_labels: int,
    ) -> RetrainTrigger:
        """Evaluate whether to trigger retraining.

        Args:
            current_metrics:  Metrics computed on the new labeled batch.
            baseline_metrics: Metrics from the most recent training run.
            n_new_labels:     Number of new confirmed labeled examples.

        Returns:
            RetrainTrigger with triggered=True if both conditions are met.
        """
        current_auc = current_metrics.get("auc", float("nan"))
        baseline_auc = baseline_metrics.get("auc", float("nan"))

        if math.isnan(current_auc) or math.isnan(baseline_auc):
            return RetrainTrigger(
                triggered=False,
                reason="AUC not available (insufficient data)",
                metric_delta=float("nan"),
                n_new_labels=n_new_labels,
            )

        delta = current_auc - baseline_auc

        if n_new_labels < self.min_batch_size:
            return RetrainTrigger(
                triggered=False,
                reason=f"Batch too small ({n_new_labels} < {self.min_batch_size})",
                metric_delta=delta,
                n_new_labels=n_new_labels,
            )

        if abs(delta) < self.auc_drift_threshold:
            return RetrainTrigger(
                triggered=False,
                reason=f"AUC delta {delta:+.4f} below threshold ±{self.auc_drift_threshold}",
                metric_delta=delta,
                n_new_labels=n_new_labels,
            )

        direction = "degraded" if delta < 0 else "improved"
        return RetrainTrigger(
            triggered=True,
            reason=f"AUC {direction} by {delta:+.4f} with {n_new_labels} new labels",
            metric_delta=delta,
            n_new_labels=n_new_labels,
        )


# ── Label Feedback Loop ───────────────────────────────────────────────────────

class LabelFeedbackLoop:
    """Orchestrates the 8-step closed feedback loop.

    Args:
        joiner:           GroundTruthJoiner with delay filter.
        recomputer:       MetricRecomputer for quality metrics.
        decider:          RetrainDecider with trigger thresholds.
        baseline_metrics: Metrics from the most recent deployed model.
    """

    def __init__(
        self,
        joiner: GroundTruthJoiner | None = None,
        recomputer: MetricRecomputer | None = None,
        decider: RetrainDecider | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ) -> None:
        self.joiner = joiner or GroundTruthJoiner()
        self.recomputer = recomputer or MetricRecomputer()
        self.decider = decider or RetrainDecider()
        self.baseline_metrics: dict[str, float] = baseline_metrics or {"auc": 0.70}

    def tick(
        self,
        predictions: list[PredictionRecord],
        outcomes: list[OutcomeRecord],
    ) -> LoopResult:
        """Execute one full pass of the feedback loop.

        Phases executed:
          AWAIT_OUTCOME → JOIN_LABEL → RECOMPUTE → TRIGGER

        Args:
            predictions: All available prediction records.
            outcomes:    All available outcome records.

        Returns:
            LoopResult with labeled examples, metrics, and trigger decision.
        """
        # Phase 5: JOIN_LABEL
        confirmed, n_provisional = self.joiner.join(predictions, outcomes)

        # Phase 6: RECOMPUTE
        current_metrics = self.recomputer.recompute(confirmed)

        # Phase 7: TRIGGER
        trigger = self.decider.decide(
            current_metrics, self.baseline_metrics, len(confirmed)
        )

        return LoopResult(
            phase_reached=LoopPhase.TRIGGER,
            labeled_examples=confirmed,
            current_metrics=current_metrics,
            trigger=trigger,
            n_provisional=n_provisional,
        )

    def update_baseline(self, metrics: dict[str, float]) -> None:
        """Update the baseline after a successful retrain and deploy."""
        self.baseline_metrics = metrics
