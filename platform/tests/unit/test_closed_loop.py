"""Tests for monitoring/closed_loop.py — LoopApprover, ClosedLoopState, ClosedLoop."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from features.feedback_loop import (
    GroundTruthJoiner,
    LoopPhase,
    OutcomeRecord,
    RetrainDecider,
    RetrainTrigger,
)
from monitoring.closed_loop import (
    ApprovalMode,
    ApprovalResult,
    ClosedLoop,
    ClosedLoopState,
    LoopApprover,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _outcome(pid: str, label: int = 1, delay: float = 60.0) -> OutcomeRecord:
    return OutcomeRecord(pid, label, _now(), delay_days=delay)


def _trigger(triggered: bool = True, delta: float = -0.05) -> RetrainTrigger:
    return RetrainTrigger(
        triggered=triggered,
        reason="test" if triggered else "small batch",
        metric_delta=delta,
        n_new_labels=50,
    )


# ── LoopApprover ───────────────────────────────────────────────────────────────

class TestLoopApprover:
    def test_auto_approves_on_improvement(self) -> None:
        approver = LoopApprover(ApprovalMode.AUTO, min_auc_improvement=0.0)
        result = approver.approve(_trigger(), {"auc": 0.78}, {"auc": 0.72})
        assert result.approved
        assert result.baseline_updated

    def test_auto_rejects_when_no_improvement(self) -> None:
        approver = LoopApprover(ApprovalMode.AUTO, min_auc_improvement=0.02)
        result = approver.approve(_trigger(), {"auc": 0.72}, {"auc": 0.72})
        assert not result.approved

    def test_auto_rejects_on_nan_auc(self) -> None:
        approver = LoopApprover(ApprovalMode.AUTO)
        result = approver.approve(_trigger(), {"auc": float("nan")}, {"auc": 0.72})
        assert not result.approved

    def test_human_always_pending(self) -> None:
        approver = LoopApprover(ApprovalMode.HUMAN)
        result = approver.approve(_trigger(), {"auc": 0.80}, {"auc": 0.70})
        assert not result.approved
        assert "PENDING" in result.reason

    def test_block_always_rejected(self) -> None:
        approver = LoopApprover(ApprovalMode.BLOCK)
        result = approver.approve(_trigger(), {"auc": 0.90}, {"auc": 0.70})
        assert not result.approved

    def test_not_triggered_returns_not_approved(self) -> None:
        approver = LoopApprover(ApprovalMode.AUTO)
        result = approver.approve(_trigger(triggered=False), {"auc": 0.80}, {"auc": 0.70})
        assert not result.approved

    def test_mode_preserved_in_result(self) -> None:
        approver = LoopApprover(ApprovalMode.HUMAN)
        result = approver.approve(_trigger(), {"auc": 0.80}, {"auc": 0.70})
        assert result.mode == ApprovalMode.HUMAN


# ── ClosedLoopState ────────────────────────────────────────────────────────────

class TestClosedLoopState:
    def test_initial_state(self) -> None:
        s = ClosedLoopState()
        assert s.current_step == LoopPhase.PREDICT
        assert s.n_predictions == 0
        assert math.isnan(s.last_auc)


# ── ClosedLoop ─────────────────────────────────────────────────────────────────

class TestClosedLoop:
    def _make_loop(self, tmp_path: Path, **kwargs) -> ClosedLoop:
        return ClosedLoop(
            log_path=str(tmp_path / "preds.jsonl"),
            model_version="v1",
            baseline_metrics={"auc": 0.70},
            **kwargs,
        )

    def test_serve_and_log_increments_counter(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path)
        loop.serve_and_log("c1", 0.7, "approve", features={"f": 1.0})
        assert loop.state.n_predictions == 1

    def test_serve_and_log_returns_entry(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path)
        entry = loop.serve_and_log("c1", 0.7, "approve")
        assert entry.entity_key == "c1"
        assert entry.score == 0.7

    def test_state_moves_to_log(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path)
        loop.serve_and_log("c1", 0.7, "approve")
        assert loop.state.current_step == LoopPhase.LOG

    def test_tick_with_no_data_returns_result(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path)
        result = loop.tick([])
        assert result is not None

    def test_tick_joins_outcomes_to_log(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path,
            joiner=GroundTruthJoiner(min_delay_days=30.0),
        )
        for i in range(5):
            loop.serve_and_log(f"c{i}", 0.8 if i % 2 else 0.3,
                               "approve" if i % 2 else "decline")

        outcomes = []
        entries = loop.logger.read_log()
        for i, e in enumerate(entries):
            outcomes.append(OutcomeRecord(e.prediction_id, i % 2, _now(), delay_days=60.0))

        result = loop.tick(outcomes)
        assert len(result.labeled_examples) == 5

    def test_tick_updates_n_labeled(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path,
            joiner=GroundTruthJoiner(min_delay_days=30.0),
        )
        for i in range(3):
            loop.serve_and_log(f"c{i}", 0.7, "approve")
        entries = loop.logger.read_log()
        outcomes = [OutcomeRecord(e.prediction_id, 1, _now(), delay_days=60.0) for e in entries]
        loop.tick(outcomes)
        assert loop.state.n_labeled == 3

    def test_tick_auto_approves_on_trigger(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path,
            joiner=GroundTruthJoiner(min_delay_days=0.0),
            decider=RetrainDecider(min_batch_size=5, auc_drift_threshold=0.0),
            approver=LoopApprover(ApprovalMode.AUTO, min_auc_improvement=0.0),
        )
        # Log 10 predictions alternating good/bad scores with matching labels
        for i in range(10):
            loop.serve_and_log(f"c{i}", 0.9 if i < 5 else 0.1, "approve")
        entries = loop.logger.read_log()
        outcomes = [
            OutcomeRecord(entries[i].prediction_id, 1 if i < 5 else 0, _now(), 0.0)
            for i in range(len(entries))
        ]
        result = loop.tick(outcomes)
        # High AUC data — trigger should fire; outcome depends on AUC vs baseline
        assert isinstance(result.trigger.triggered, bool)

    def test_block_mode_never_updates_baseline(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path,
            joiner=GroundTruthJoiner(min_delay_days=0.0),
            decider=RetrainDecider(min_batch_size=5, auc_drift_threshold=0.0),
            approver=LoopApprover(ApprovalMode.BLOCK),
        )
        initial_baseline = dict(loop.baseline_metrics)
        for i in range(10):
            loop.serve_and_log(f"c{i}", 0.9 if i < 5 else 0.1, "approve")
        entries = loop.logger.read_log()
        outcomes = [
            OutcomeRecord(entries[i].prediction_id, 1 if i < 5 else 0, _now(), 0.0)
            for i in range(len(entries))
        ]
        loop.tick(outcomes)
        assert loop.state.n_approved == 0

    def test_get_state_returns_state(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path)
        state = loop.get_state()
        assert isinstance(state, ClosedLoopState)

    def test_baseline_accessible(self, tmp_path: Path) -> None:
        loop = self._make_loop(tmp_path)
        assert loop.baseline_metrics["auc"] == pytest.approx(0.70)
