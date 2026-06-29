"""Model validation gate — thresholds, champion/challenger, auto-promote.

Production-grade model gate for the credit-risk training pipeline:
  - ModelMetrics:     typed metrics snapshot for a trained model
  - GateThresholds:   configurable thresholds for all gate checks
  - ChampionRegistry: in-process registry tracking current champion + history
  - ModelGateReport:  typed gate evaluation result
  - ModelGate:        runs hard gates + champion comparison + auto-promote

See: docs/phase5/day35_model_gate.md
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── Model Metrics ─────────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    """Metrics snapshot for a trained model.

    Attributes:
        auc:               ROC-AUC on the held-out test set.
        ece:               Expected Calibration Error (lower is better).
        brier:             Brier score (lower is better).
        slice_auc_gap:     Max AUC gap across demographic slices.
        cost_at_threshold: Business cost (FP × cost_fp + FN × cost_fn).
        n_test:            Number of test rows used for evaluation.
        model_version:     Human-readable version string (e.g. "v-abc12345").
        status:            "candidate" | "champion" | "rejected" | "previous_stable".
        extra:             Additional metadata (e.g. training config, feature names).
    """

    auc: float
    ece: float = 0.0
    brier: float = 0.0
    slice_auc_gap: float = 0.0
    cost_at_threshold: float = 0.0
    n_test: int = 0
    model_version: str = "unknown"
    status: str = "candidate"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.auc <= 1.0:
            raise ValueError(f"auc must be in [0, 1], got {self.auc}")
        if self.ece < 0:
            raise ValueError(f"ece must be >= 0, got {self.ece}")
        if self.slice_auc_gap < 0:
            raise ValueError(f"slice_auc_gap must be >= 0, got {self.slice_auc_gap}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "auc": self.auc,
            "ece": self.ece,
            "brier": self.brier,
            "slice_auc_gap": self.slice_auc_gap,
            "cost_at_threshold": self.cost_at_threshold,
            "n_test": self.n_test,
            "model_version": self.model_version,
            "status": self.status,
            **self.extra,
        }


# ── Gate Thresholds ───────────────────────────────────────────────────────────

@dataclass
class GateThresholds:
    """All configurable thresholds for the model gate.

    Attributes:
        min_auc:         Minimum AUC (hard gate — absolute floor).
        max_ece:         Maximum Expected Calibration Error.
        max_slice_gap:   Maximum AUC gap across demographic slices.
        champion_delta:  Challenger must exceed champion AUC by this much.
        max_cost:        Maximum acceptable business cost (None = not checked).
    """

    min_auc: float = 0.75
    max_ece: float = 0.05
    max_slice_gap: float = 0.10
    champion_delta: float = 0.005
    max_cost: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.min_auc <= 1:
            raise ValueError(f"min_auc must be in (0, 1], got {self.min_auc}")
        if self.max_ece < 0:
            raise ValueError(f"max_ece must be >= 0, got {self.max_ece}")
        if self.max_slice_gap < 0:
            raise ValueError(f"max_slice_gap must be >= 0, got {self.max_slice_gap}")
        if self.champion_delta < 0:
            raise ValueError(f"champion_delta must be >= 0, got {self.champion_delta}")

    @classmethod
    def from_env(cls) -> "GateThresholds":
        """Load thresholds from environment variables with defaults."""
        return cls(
            min_auc=float(os.environ.get("GATE_MIN_AUC", "0.75")),
            max_ece=float(os.environ.get("GATE_MAX_ECE", "0.05")),
            max_slice_gap=float(os.environ.get("GATE_MAX_SLICE_GAP", "0.10")),
            champion_delta=float(os.environ.get("GATE_CHAMPION_DELTA", "0.005")),
        )


# ── Champion Registry ─────────────────────────────────────────────────────────

class ChampionRegistry:
    """In-process registry tracking champion model and history.

    In production this would be backed by MLflow Model Registry or a
    database. Here we use an in-memory list for testability.

    Attributes:
        current_champion: The current champion ModelMetrics, or None.
        history:          Ordered list of all ModelMetrics ever registered.
    """

    def __init__(self) -> None:
        self.current_champion: ModelMetrics | None = None
        self.history: list[ModelMetrics] = []

    def get_champion(self) -> ModelMetrics | None:
        """Return current champion metrics, or None if no champion yet."""
        return self.current_champion

    def promote(self, metrics: ModelMetrics) -> None:
        """Set `metrics` as the new champion.

        The previous champion is archived as 'previous_stable'.
        """
        if self.current_champion is not None:
            self.current_champion.status = "previous_stable"

        metrics.status = "champion"
        self.current_champion = metrics
        self.history.append(metrics)
        log.info(
            "Champion promoted: version=%s AUC=%.4f",
            metrics.model_version, metrics.auc,
        )

    def reject(self, metrics: ModelMetrics, reason: str = "") -> None:
        """Record a rejected challenger in history (for audit trail)."""
        metrics.status = "rejected"
        metrics.extra["rejection_reason"] = reason
        self.history.append(metrics)
        log.info(
            "Challenger rejected: version=%s AUC=%.4f reason=%r",
            metrics.model_version, metrics.auc, reason,
        )

    def rollback(self) -> ModelMetrics | None:
        """Roll back to the most recent 'previous_stable' model.

        Returns the restored champion metrics, or None if no previous stable.
        """
        for m in reversed(self.history):
            if m.status in ("previous_stable", "champion") and m is not self.current_champion:
                if self.current_champion:
                    self.current_champion.status = "rolled_back"
                m.status = "champion"
                self.current_champion = m
                log.info("Rollback to version=%s", m.model_version)
                return m
        return None

    def previous_stable(self) -> ModelMetrics | None:
        """Return the most recent previous_stable model."""
        for m in reversed(self.history):
            if m.status == "previous_stable":
                return m
        return None


# ── Model Gate Report ─────────────────────────────────────────────────────────

@dataclass
class ModelGateReport:
    """Result of a ModelGate evaluation.

    Attributes:
        passed:              True if all gates passed (note: passed ≠ promoted).
        promoted:            True if challenger was promoted to champion.
        challenger_metrics:  Metrics for the evaluated challenger.
        champion_metrics:    Metrics for the current champion (if any).
        gate_failures:       List of gate failure messages (empty if passed).
        promotion_reason:    Human-readable reason for promotion.
        rejection_reason:    Human-readable reason for rejection.
        duration_s:          Wall-clock seconds for gate evaluation.
    """

    passed: bool
    promoted: bool
    challenger_metrics: ModelMetrics
    champion_metrics: ModelMetrics | None = None
    gate_failures: list[str] = field(default_factory=list)
    promotion_reason: str | None = None
    rejection_reason: str | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "promoted": self.promoted,
            "challenger": self.challenger_metrics.to_dict(),
            "champion": self.champion_metrics.to_dict() if self.champion_metrics else None,
            "gate_failures": self.gate_failures,
            "promotion_reason": self.promotion_reason,
            "rejection_reason": self.rejection_reason,
            "duration_s": self.duration_s,
        }


# ── Model Gate ────────────────────────────────────────────────────────────────

class ModelGate:
    """Evaluates a trained model against hard gates and champion comparison.

    Evaluation sequence:
        1. Hard gates: min_auc, max_ece, max_slice_gap, max_cost
        2. If any hard gate fails → reject immediately (do not compare to champion)
        3. Champion comparison: challenger.auc - champion.auc >= champion_delta
        4. If no champion → promote automatically (first model)
        5. If challenger beats champion → promote
        6. Otherwise → reject

    Args:
        thresholds: GateThresholds with all configurable limits.
        registry:   ChampionRegistry for champion lookup and promotion.
    """

    def __init__(
        self,
        thresholds: GateThresholds | None = None,
        registry: ChampionRegistry | None = None,
    ) -> None:
        self.thresholds = thresholds or GateThresholds()
        self.registry = registry or ChampionRegistry()

    def evaluate(self, metrics: ModelMetrics) -> ModelGateReport:
        """Evaluate a challenger model.

        Args:
            metrics: ModelMetrics computed on the held-out test set.

        Returns:
            ModelGateReport with the full evaluation result.
        """
        start = time.monotonic()
        champion = self.registry.get_champion()

        # Hard gates — all must pass
        failures = self._hard_gates(metrics)

        if failures:
            self.registry.reject(metrics, reason="; ".join(failures))
            return ModelGateReport(
                passed=False,
                promoted=False,
                challenger_metrics=metrics,
                champion_metrics=champion,
                gate_failures=failures,
                rejection_reason="; ".join(failures),
                duration_s=time.monotonic() - start,
            )

        # Champion comparison
        if champion is None:
            # No champion yet — first model that passes hard gates becomes champion
            self.registry.promote(metrics)
            return ModelGateReport(
                passed=True,
                promoted=True,
                challenger_metrics=metrics,
                champion_metrics=None,
                gate_failures=[],
                promotion_reason="First model — no champion to compare",
                duration_s=time.monotonic() - start,
            )

        delta = metrics.auc - champion.auc
        if delta >= self.thresholds.champion_delta:
            self.registry.promote(metrics)
            return ModelGateReport(
                passed=True,
                promoted=True,
                challenger_metrics=metrics,
                champion_metrics=champion,
                gate_failures=[],
                promotion_reason=(
                    f"AUC gain {delta:.4f} >= delta {self.thresholds.champion_delta}"
                ),
                duration_s=time.monotonic() - start,
            )
        else:
            reason = (
                f"AUC delta {delta:.4f} < required {self.thresholds.champion_delta} "
                f"(challenger={metrics.auc:.4f}, champion={champion.auc:.4f})"
            )
            self.registry.reject(metrics, reason=reason)
            return ModelGateReport(
                passed=False,
                promoted=False,
                challenger_metrics=metrics,
                champion_metrics=champion,
                gate_failures=[reason],
                rejection_reason=reason,
                duration_s=time.monotonic() - start,
            )

    def _hard_gates(self, metrics: ModelMetrics) -> list[str]:
        """Run absolute threshold checks. Returns failure messages."""
        failures: list[str] = []

        if metrics.auc < self.thresholds.min_auc:
            failures.append(
                f"AUC {metrics.auc:.4f} < min_auc {self.thresholds.min_auc}"
            )

        if metrics.ece > self.thresholds.max_ece:
            failures.append(
                f"ECE {metrics.ece:.4f} > max_ece {self.thresholds.max_ece}"
            )

        if metrics.slice_auc_gap > self.thresholds.max_slice_gap:
            failures.append(
                f"slice_auc_gap {metrics.slice_auc_gap:.4f} > max_slice_gap {self.thresholds.max_slice_gap}"
            )

        if self.thresholds.max_cost is not None and metrics.cost_at_threshold > self.thresholds.max_cost:
            failures.append(
                f"cost_at_threshold {metrics.cost_at_threshold:.0f} > max_cost {self.thresholds.max_cost:.0f}"
            )

        return failures


# ── Metrics from model ────────────────────────────────────────────────────────

def compute_model_metrics(
    model: Any,
    X_test: Any,
    y_test: Any,
    *,
    model_version: str = "unknown",
    cost_fp: float = 2_000.0,
    cost_fn: float = 8_000.0,
    threshold: float = 0.5,
    slice_column: Any | None = None,
) -> ModelMetrics:
    """Compute ModelMetrics from a fitted model and test split.

    Args:
        model:          Fitted sklearn-compatible estimator.
        X_test:         Test features DataFrame or array.
        y_test:         True labels.
        model_version:  Version string to embed in metrics.
        cost_fp:        Business cost per false positive.
        cost_fn:        Business cost per false negative.
        threshold:      Decision threshold for cost computation.
        slice_column:   Optional Series for slice AUC gap computation.

    Returns:
        ModelMetrics with AUC, ECE, Brier, slice_gap, and cost.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    proba = model.predict_proba(X_test)[:, 1]
    y_arr = np.array(y_test)

    auc = float(roc_auc_score(y_arr, proba))
    brier = float(brier_score_loss(y_arr, proba))

    # ECE (10 bins)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    n = len(y_arr)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() > 0:
            frac = float(y_arr[mask].mean())
            mean_p = float(proba[mask].mean())
            ece += (mask.sum() / n) * abs(frac - mean_p)

    # Slice AUC gap
    slice_gap = 0.0
    if slice_column is not None:
        groups = np.unique(np.array(slice_column))
        aucs: list[float] = []
        for g in groups:
            mask = np.array(slice_column) == g
            if mask.sum() >= 30 and len(np.unique(y_arr[mask])) > 1:
                aucs.append(float(roc_auc_score(y_arr[mask], proba[mask])))
        if len(aucs) >= 2:
            slice_gap = float(max(aucs) - min(aucs))

    # Business cost
    preds = (proba >= threshold).astype(int)
    fp = int(((preds == 1) & (y_arr == 0)).sum())
    fn = int(((preds == 0) & (y_arr == 1)).sum())
    cost = fp * cost_fp + fn * cost_fn

    return ModelMetrics(
        auc=auc,
        ece=float(ece),
        brier=brier,
        slice_auc_gap=slice_gap,
        cost_at_threshold=cost,
        n_test=len(y_arr),
        model_version=model_version,
    )
