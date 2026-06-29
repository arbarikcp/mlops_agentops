"""Closed-loop learning system: orchestrates all 8 steps from PREDICT to APPROVE.

Day 52 — wraps Phase 6 feedback components (GroundTruthJoiner, MetricRecomputer,
RetrainDecider) with a PredictionLogger and LoopApprover into a single ClosedLoop
that tracks state across steps.

Classes:
  ApprovalMode   — AUTO / HUMAN / BLOCK
  ApprovalResult — outcome of the APPROVE step
  LoopApprover   — human-in-the-loop or auto-approval gate (Step 8)
  ClosedLoopState — current position in the 8-step state machine
  ClosedLoop     — orchestrates all steps; produces LoopResult per tick

See: docs/phase7/day52_closed_loop.md
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from features.feedback_loop import (
    GroundTruthJoiner,
    LabelFeedbackLoop,
    LoopPhase,
    LoopResult,
    MetricRecomputer,
    OutcomeRecord,
    PredictionRecord,
    RetrainDecider,
    RetrainTrigger,
)
from monitoring.prediction_logger import PredictionLogEntry, PredictionLogger


# ── ApprovalMode ──────────────────────────────────────────────────────────────

class ApprovalMode(str, Enum):
    AUTO  = "auto"   # Approve automatically if AUC improved
    HUMAN = "human"  # Return PENDING — waits for external signal
    BLOCK = "block"  # Always reject (CI dry-run)


# ── ApprovalResult ─────────────────────────────────────────────────────────────

@dataclass
class ApprovalResult:
    """Outcome of the APPROVE step (Step 8).

    Attributes:
        approved:         True if the new model is cleared for deployment.
        mode:             Which approval mode was used.
        reason:           Human-readable explanation.
        baseline_updated: True if the baseline metrics were updated after approval.
    """

    approved: bool
    mode: ApprovalMode
    reason: str = ""
    baseline_updated: bool = False


# ── LoopApprover ──────────────────────────────────────────────────────────────

class LoopApprover:
    """Human-in-the-loop or automated gate for Step 8 (APPROVE).

    Args:
        mode:                AUTO / HUMAN / BLOCK.
        min_auc_improvement: Minimum AUC delta for AUTO approval (default: 0.0 — any improvement).
    """

    def __init__(
        self,
        mode: ApprovalMode = ApprovalMode.AUTO,
        min_auc_improvement: float = 0.0,
    ) -> None:
        self.mode = mode
        self.min_auc_improvement = min_auc_improvement

    def approve(
        self,
        trigger: RetrainTrigger,
        current_metrics: dict[str, float],
        baseline_metrics: dict[str, float],
    ) -> ApprovalResult:
        """Run the approval gate.

        Args:
            trigger:          The RetrainTrigger from Step 7.
            current_metrics:  Metrics on new labeled data.
            baseline_metrics: Previously accepted baseline metrics.

        Returns:
            ApprovalResult with approved=True if the gate passes.
        """
        if not trigger.triggered:
            return ApprovalResult(approved=False, mode=self.mode, reason="trigger not fired")

        if self.mode == ApprovalMode.BLOCK:
            return ApprovalResult(approved=False, mode=self.mode, reason="approval blocked (CI mode)")

        if self.mode == ApprovalMode.HUMAN:
            return ApprovalResult(
                approved=False, mode=self.mode,
                reason="awaiting human approval — PENDING",
            )

        # AUTO — approve if AUC improved by at least min_auc_improvement
        current_auc = current_metrics.get("auc", float("nan"))
        baseline_auc = baseline_metrics.get("auc", float("nan"))

        if math.isnan(current_auc) or math.isnan(baseline_auc):
            return ApprovalResult(
                approved=False, mode=self.mode,
                reason="AUC is NaN — cannot auto-approve",
            )

        delta = current_auc - baseline_auc
        if delta >= self.min_auc_improvement:
            return ApprovalResult(
                approved=True, mode=self.mode,
                reason=f"AUTO approved — AUC delta={delta:+.4f} >= min={self.min_auc_improvement}",
                baseline_updated=True,
            )

        return ApprovalResult(
            approved=False, mode=self.mode,
            reason=f"AUTO rejected — AUC delta={delta:+.4f} < min={self.min_auc_improvement}",
        )


# ── ClosedLoopState ────────────────────────────────────────────────────────────

@dataclass
class ClosedLoopState:
    """Current position and counters in the 8-step closed-loop state machine.

    Attributes:
        current_step:    Current LoopPhase (one of the 8 steps).
        n_predictions:   Total predictions logged in this loop session.
        n_labeled:       Total confirmed labeled examples accumulated.
        last_auc:        Most-recent recomputed AUC (NaN if never computed).
        last_trigger_ts: UTC timestamp of last retrain trigger (None if never).
        n_approved:      Number of times approval gate passed.
    """

    current_step: LoopPhase = LoopPhase.PREDICT
    n_predictions: int = 0
    n_labeled: int = 0
    last_auc: float = float("nan")
    last_trigger_ts: datetime | None = None
    n_approved: int = 0


# ── ClosedLoop ─────────────────────────────────────────────────────────────────

class ClosedLoop:
    """Orchestrates all 8 closed-loop steps for a credit-risk model.

    Usage::

        loop = ClosedLoop(
            log_path="logs/predictions.jsonl",
            baseline_metrics={"auc": 0.76},
        )
        # Step 1–3: serve + log
        loop.serve_and_log("c123", score=0.72, decision="approve", features={...})

        # Steps 5–8: tick when outcomes arrive
        result = loop.tick(outcomes)

    Args:
        log_path:         JSONL prediction log path.
        model_version:    Model version embedded in log entries.
        baseline_metrics: Starting AUC / approval rate for trigger comparison.
        joiner:           Overridable GroundTruthJoiner (default: 30-day delay).
        recomputer:       Overridable MetricRecomputer.
        decider:          Overridable RetrainDecider.
        approver:         Overridable LoopApprover (default: AUTO).
    """

    def __init__(
        self,
        log_path: str = "logs/predictions.jsonl",
        model_version: str = "v1",
        baseline_metrics: dict[str, float] | None = None,
        joiner: GroundTruthJoiner | None = None,
        recomputer: MetricRecomputer | None = None,
        decider: RetrainDecider | None = None,
        approver: LoopApprover | None = None,
    ) -> None:
        self.logger = PredictionLogger(log_path, model_version=model_version)
        self._feedback = LabelFeedbackLoop(
            joiner=joiner,
            recomputer=recomputer,
            decider=decider,
            baseline_metrics=baseline_metrics or {"auc": 0.70},
        )
        self.approver = approver or LoopApprover(mode=ApprovalMode.AUTO)
        self.state = ClosedLoopState()

    # ── Steps 1–3: PREDICT → DECIDE → LOG ────────────────────────────────────

    def serve_and_log(
        self,
        entity_key: str,
        score: float,
        decision: str,
        features: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
        correlation_id: str | None = None,
    ) -> PredictionLogEntry:
        """Execute Steps 1–3: score received → decision → prediction logged.

        Args:
            entity_key:     Customer identifier.
            score:          Model output probability.
            decision:       approve / review / decline.
            features:       Feature snapshot for replay.
            latency_ms:     Inference latency.
            correlation_id: Optional request-scoped trace ID.

        Returns:
            The written PredictionLogEntry.
        """
        self.state.current_step = LoopPhase.LOG
        entry = self.logger.log(
            entity_key=entity_key,
            score=score,
            decision=decision,
            features=features or {},
            latency_ms=latency_ms,
            correlation_id=correlation_id,
        )
        self.state.n_predictions += 1
        return entry

    # ── Steps 5–8: JOIN → RECOMPUTE → TRIGGER → APPROVE ─────────────────────

    def tick(self, outcomes: list[OutcomeRecord]) -> LoopResult:
        """Execute Steps 5–8 using logged predictions and incoming outcomes.

        Reads the prediction log, joins with outcomes (Step 5), recomputes
        metrics (Step 6), checks the trigger (Step 7), and runs the approval
        gate (Step 8) if triggered.

        Args:
            outcomes: List of OutcomeRecord objects (delayed ground truth).

        Returns:
            LoopResult from the feedback loop (phase_reached, trigger, etc.).
        """
        self.state.current_step = LoopPhase.AWAIT_OUTCOME

        # Convert log entries to PredictionRecord objects for the feedback loop
        log_entries = self.logger.read_log()
        predictions = [
            PredictionRecord(
                prediction_id=e.prediction_id,
                entity_key=e.entity_key,
                score=e.score,
                decision=e.decision,
                prediction_ts=e.prediction_ts,
                features=e.features,
            )
            for e in log_entries
        ]

        # Steps 5–7 via LabelFeedbackLoop
        result = self._feedback.tick(predictions, outcomes)
        self.state.n_labeled = len(result.labeled_examples)

        auc = result.current_metrics.get("auc", float("nan"))
        if not math.isnan(auc):
            self.state.last_auc = auc

        # Step 8 — APPROVE
        if result.trigger.triggered:
            self.state.last_trigger_ts = datetime.now(timezone.utc)
            approval = self.approver.approve(
                result.trigger,
                result.current_metrics,
                self._feedback.baseline_metrics,
            )
            if approval.approved and approval.baseline_updated:
                self._feedback.update_baseline(result.current_metrics)
                self.state.n_approved += 1
            self.state.current_step = LoopPhase.APPROVE
        else:
            self.state.current_step = LoopPhase.TRIGGER

        return result

    def get_state(self) -> ClosedLoopState:
        return self.state

    @property
    def baseline_metrics(self) -> dict[str, float]:
        return self._feedback.baseline_metrics
