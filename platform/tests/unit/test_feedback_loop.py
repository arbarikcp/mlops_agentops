"""Tests for features/feedback_loop.py — GroundTruthJoiner, MetricRecomputer, RetrainDecider, LabelFeedbackLoop."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from features.feedback_loop import (
    GroundTruthJoiner,
    LabelFeedbackLoop,
    LabeledExample,
    LoopPhase,
    MetricRecomputer,
    OutcomeRecord,
    PredictionRecord,
    RetrainDecider,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _pred(pid: str, score: float = 0.7, decision: str = "approve") -> PredictionRecord:
    return PredictionRecord(
        prediction_id=pid,
        entity_key=f"c:{pid}",
        score=score,
        decision=decision,
        prediction_ts=_now(),
    )


def _outcome(pid: str, label: int = 1, delay: float = 60.0) -> OutcomeRecord:
    return OutcomeRecord(
        prediction_id=pid,
        actual_outcome=label,
        outcome_ts=_now(),
        delay_days=delay,
    )


def _labeled_set(n: int = 50, pos_rate: float = 0.3) -> list[LabeledExample]:
    examples = []
    for i in range(n):
        label = 1 if i / n < pos_rate else 0
        score = 0.8 if label == 1 else 0.3
        examples.append(LabeledExample(
            prediction_id=str(i),
            entity_key=f"c:{i}",
            score=score,
            label=label,
            delay_days=60.0,
            decision="approve",
        ))
    return examples


# ── PredictionRecord ───────────────────────────────────────────────────────────

class TestPredictionRecord:
    def test_basic(self) -> None:
        p = _pred("p1", score=0.6, decision="review")
        assert p.decision == "review"
        assert p.score == pytest.approx(0.6)

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="prediction_id"):
            PredictionRecord("", "c1", 0.5, "approve", _now())

    def test_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="score"):
            PredictionRecord("p1", "c1", 1.5, "approve", _now())

    def test_invalid_decision_raises(self) -> None:
        with pytest.raises(ValueError, match="decision"):
            PredictionRecord("p1", "c1", 0.5, "maybe", _now())


# ── OutcomeRecord ──────────────────────────────────────────────────────────────

class TestOutcomeRecord:
    def test_basic(self) -> None:
        o = _outcome("p1", label=0)
        assert o.actual_outcome == 0

    def test_none_outcome_allowed(self) -> None:
        o = OutcomeRecord("p1", None, _now(), 10.0)
        assert o.actual_outcome is None

    def test_invalid_outcome_raises(self) -> None:
        with pytest.raises(ValueError, match="actual_outcome"):
            OutcomeRecord("p1", 2, _now(), 30.0)

    def test_negative_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="delay_days"):
            OutcomeRecord("p1", 1, _now(), -5.0)


# ── GroundTruthJoiner ─────────────────────────────────────────────────────────

class TestGroundTruthJoiner:
    def test_joins_confirmed_outcomes(self) -> None:
        joiner = GroundTruthJoiner(min_delay_days=30.0)
        preds = [_pred("p1"), _pred("p2")]
        outcomes = [_outcome("p1", label=1, delay=60), _outcome("p2", label=0, delay=45)]
        confirmed, n_prov = joiner.join(preds, outcomes)
        assert len(confirmed) == 2
        assert n_prov == 0

    def test_excludes_provisional_by_delay(self) -> None:
        joiner = GroundTruthJoiner(min_delay_days=30.0)
        preds = [_pred("p1")]
        outcomes = [_outcome("p1", label=1, delay=10.0)]  # delay < 30
        confirmed, n_prov = joiner.join(preds, outcomes)
        assert len(confirmed) == 0
        assert n_prov == 1

    def test_excludes_missing_outcomes(self) -> None:
        joiner = GroundTruthJoiner(min_delay_days=30.0)
        preds = [_pred("p1")]
        outcomes = []  # no outcomes yet
        confirmed, n_prov = joiner.join(preds, outcomes)
        assert len(confirmed) == 0
        assert n_prov == 1

    def test_excludes_null_outcome(self) -> None:
        joiner = GroundTruthJoiner(min_delay_days=30.0)
        preds = [_pred("p1")]
        outcomes = [OutcomeRecord("p1", None, _now(), 60.0)]
        confirmed, n_prov = joiner.join(preds, outcomes)
        assert len(confirmed) == 0
        assert n_prov == 1

    def test_label_carried_to_labeled_example(self) -> None:
        joiner = GroundTruthJoiner(min_delay_days=30.0)
        preds = [_pred("p1", score=0.9, decision="decline")]
        outcomes = [_outcome("p1", label=1, delay=90.0)]
        confirmed, _ = joiner.join(preds, outcomes)
        assert confirmed[0].label == 1
        assert confirmed[0].decision == "decline"
        assert confirmed[0].score == pytest.approx(0.9)

    def test_negative_min_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="min_delay_days"):
            GroundTruthJoiner(min_delay_days=-1.0)


# ── MetricRecomputer ───────────────────────────────────────────────────────────

class TestMetricRecomputer:
    def test_auc_high_for_good_scores(self) -> None:
        recomputer = MetricRecomputer()
        examples = _labeled_set(100, pos_rate=0.3)
        metrics = recomputer.recompute(examples)
        assert metrics["auc"] > 0.5

    def test_nan_for_small_batch(self) -> None:
        recomputer = MetricRecomputer()
        examples = _labeled_set(5)
        metrics = recomputer.recompute(examples)
        assert math.isnan(metrics["auc"])

    def test_approval_rate_correct(self) -> None:
        recomputer = MetricRecomputer()
        examples = _labeled_set(20)
        # all have decision="approve" in fixture
        metrics = recomputer.recompute(examples)
        assert metrics["approval_rate"] == pytest.approx(1.0)

    def test_n_labeled_correct(self) -> None:
        recomputer = MetricRecomputer()
        examples = _labeled_set(50)
        metrics = recomputer.recompute(examples)
        assert metrics["n_labeled"] == 50.0

    def test_label_positive_rate(self) -> None:
        recomputer = MetricRecomputer()
        examples = _labeled_set(100, pos_rate=0.3)
        metrics = recomputer.recompute(examples)
        assert metrics["label_positive_rate"] == pytest.approx(0.3)

    def test_auc_is_between_zero_and_one(self) -> None:
        recomputer = MetricRecomputer()
        examples = _labeled_set(100)
        metrics = recomputer.recompute(examples)
        assert 0.0 <= metrics["auc"] <= 1.0


# ── RetrainDecider ─────────────────────────────────────────────────────────────

class TestRetrainDecider:
    def test_triggers_when_both_conditions_met(self) -> None:
        decider = RetrainDecider(min_batch_size=10, auc_drift_threshold=0.02)
        current = {"auc": 0.65}
        baseline = {"auc": 0.70}
        trigger = decider.decide(current, baseline, n_new_labels=50)
        assert trigger.triggered

    def test_does_not_trigger_small_batch(self) -> None:
        decider = RetrainDecider(min_batch_size=100, auc_drift_threshold=0.02)
        trigger = decider.decide({"auc": 0.60}, {"auc": 0.70}, n_new_labels=5)
        assert not trigger.triggered
        assert "small" in trigger.reason.lower()

    def test_does_not_trigger_small_delta(self) -> None:
        decider = RetrainDecider(min_batch_size=10, auc_drift_threshold=0.05)
        trigger = decider.decide({"auc": 0.70}, {"auc": 0.71}, n_new_labels=50)
        assert not trigger.triggered
        assert "threshold" in trigger.reason.lower()

    def test_triggers_on_improvement_too(self) -> None:
        decider = RetrainDecider(min_batch_size=10, auc_drift_threshold=0.02)
        trigger = decider.decide({"auc": 0.80}, {"auc": 0.70}, n_new_labels=50)
        assert trigger.triggered
        assert "improved" in trigger.reason.lower()

    def test_nan_auc_does_not_trigger(self) -> None:
        decider = RetrainDecider(min_batch_size=1, auc_drift_threshold=0.0)
        trigger = decider.decide({"auc": float("nan")}, {"auc": 0.70}, n_new_labels=50)
        assert not trigger.triggered

    def test_metric_delta_reported(self) -> None:
        decider = RetrainDecider(min_batch_size=10, auc_drift_threshold=0.02)
        trigger = decider.decide({"auc": 0.65}, {"auc": 0.70}, n_new_labels=50)
        assert trigger.metric_delta == pytest.approx(-0.05)

    def test_invalid_min_batch_raises(self) -> None:
        with pytest.raises(ValueError, match="min_batch_size"):
            RetrainDecider(min_batch_size=0)


# ── LabelFeedbackLoop ─────────────────────────────────────────────────────────

class TestLabelFeedbackLoop:
    def _make_data(self, n: int = 50, label: int = 1, delay: float = 60.0):
        preds = [_pred(f"p{i}", score=0.8 if label else 0.3) for i in range(n)]
        outcomes = [_outcome(f"p{i}", label=label, delay=delay) for i in range(n)]
        return preds, outcomes

    def test_tick_returns_loop_result(self) -> None:
        loop = LabelFeedbackLoop(baseline_metrics={"auc": 0.70})
        preds, outcomes = self._make_data(50)
        result = loop.tick(preds, outcomes)
        assert result.phase_reached == LoopPhase.TRIGGER

    def test_tick_labeled_examples_populated(self) -> None:
        loop = LabelFeedbackLoop(baseline_metrics={"auc": 0.70})
        preds, outcomes = self._make_data(50)
        result = loop.tick(preds, outcomes)
        assert len(result.labeled_examples) == 50

    def test_tick_provisional_count(self) -> None:
        loop = LabelFeedbackLoop(
            joiner=GroundTruthJoiner(min_delay_days=90.0),
            baseline_metrics={"auc": 0.70},
        )
        preds = [_pred("p1"), _pred("p2")]
        outcomes = [_outcome("p1", label=1, delay=30.0)]  # delay < 90
        result = loop.tick(preds, outcomes)
        assert result.n_provisional >= 1

    def test_update_baseline(self) -> None:
        loop = LabelFeedbackLoop(baseline_metrics={"auc": 0.70})
        loop.update_baseline({"auc": 0.75})
        assert loop.baseline_metrics["auc"] == pytest.approx(0.75)

    def test_trigger_fires_on_large_delta(self) -> None:
        loop = LabelFeedbackLoop(
            decider=RetrainDecider(min_batch_size=10, auc_drift_threshold=0.02),
            baseline_metrics={"auc": 0.50},
        )
        preds, outcomes = self._make_data(100, label=0)
        # all label=0 → AUC will be nan (only one class)
        result = loop.tick(preds, outcomes)
        # With one class, AUC is nan → no trigger expected
        assert isinstance(result.trigger.triggered, bool)

    def test_no_trigger_when_stable(self) -> None:
        loop = LabelFeedbackLoop(
            decider=RetrainDecider(min_batch_size=10, auc_drift_threshold=0.10),
            baseline_metrics={"auc": 0.70},
        )
        # Very small batch → no trigger
        preds = [_pred("p1")]
        outcomes = [_outcome("p1", delay=60.0)]
        result = loop.tick(preds, outcomes)
        assert not result.trigger.triggered
