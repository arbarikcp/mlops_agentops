"""SLO / SLI / Error Budget checker for the credit-risk ML service.

Day 53 — defines SLOs for operational, ML quality, and business dimensions.
Each SLO has a target, a window, and produces a BudgetStatus (GREEN/YELLOW/RED/EXHAUSTED).

Classes:
  SLOType        — LATENCY / ERROR_RATE / MODEL_QUALITY / FEATURE_FRESHNESS / BUSINESS
  BudgetStatus   — GREEN / YELLOW / RED / EXHAUSTED
  SLODefinition  — target + window for one SLO
  SLOResult      — observed value, compliance, budget consumption
  SLOReport      — aggregated across all SLOs
  SLOChecker     — evaluates metrics against SLO definitions

See: docs/phase7/day53_slo_monitoring_gate.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ── Enumerations ──────────────────────────────────────────────────────────────

class SLOType(str, Enum):
    LATENCY           = "latency"
    ERROR_RATE        = "error_rate"
    MODEL_QUALITY     = "model_quality"
    FEATURE_FRESHNESS = "feature_freshness"
    BUSINESS          = "business"


class BudgetStatus(str, Enum):
    GREEN     = "green"     # > 50% budget remaining
    YELLOW    = "yellow"    # 10–50% remaining
    RED       = "red"       # < 10% remaining
    EXHAUSTED = "exhausted" # 0% remaining


def _budget_status(consumed_pct: float) -> BudgetStatus:
    if consumed_pct >= 1.0:
        return BudgetStatus.EXHAUSTED
    if consumed_pct >= 0.90:
        return BudgetStatus.RED
    if consumed_pct >= 0.50:
        return BudgetStatus.YELLOW
    return BudgetStatus.GREEN


# ── SLODefinition ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SLODefinition:
    """Definition of one SLO.

    Attributes:
        name:         Human-readable SLO identifier.
        slo_type:     Category of SLO.
        target:       SLO compliance target (e.g., 0.999 = 99.9%).
        window_hours: Evaluation window in hours (e.g., 720 = 30 days).
        description:  What this SLO measures.
    """

    name: str
    slo_type: SLOType
    target: float
    window_hours: float = 720.0  # 30 days
    description: str = ""

    def __post_init__(self) -> None:
        if not (0.0 < self.target <= 1.0):
            raise ValueError(f"SLO target must be in (0, 1], got {self.target}")
        if self.window_hours <= 0:
            raise ValueError(f"window_hours must be positive, got {self.window_hours}")

    @property
    def error_budget_hours(self) -> float:
        """Total allowed failure hours in the window."""
        return (1.0 - self.target) * self.window_hours


# ── SLOResult ─────────────────────────────────────────────────────────────────

@dataclass
class SLOResult:
    """Result of evaluating one SLO against observed metrics.

    Attributes:
        slo_name:           Identifies which SLO was checked.
        slo_type:           Type of SLO.
        current_value:      Observed metric value (semantics depend on type).
        target:             SLO target value.
        compliant:          True if current_value meets the target.
        budget_consumed_pct: Fraction of error budget consumed (0–1+).
        budget_status:      GREEN / YELLOW / RED / EXHAUSTED.
        description:        Human-readable explanation.
    """

    slo_name: str
    slo_type: SLOType
    current_value: float
    target: float
    compliant: bool
    budget_consumed_pct: float
    budget_status: BudgetStatus
    description: str = ""

    @property
    def blocks_gate(self) -> bool:
        return self.budget_status in (BudgetStatus.RED, BudgetStatus.EXHAUSTED)


# ── SLOReport ─────────────────────────────────────────────────────────────────

@dataclass
class SLOReport:
    """Aggregated SLO report across all checked SLOs.

    Attributes:
        results:          All SLOResult objects.
        overall_compliant: True if all SLOs are compliant and no budget is exhausted.
    """

    results: list[SLOResult] = field(default_factory=list)
    overall_compliant: bool = True

    def by_type(self, slo_type: SLOType) -> list[SLOResult]:
        return [r for r in self.results if r.slo_type == slo_type]

    def violations(self) -> list[SLOResult]:
        return [r for r in self.results if not r.compliant]

    def gate_blocking(self) -> list[SLOResult]:
        return [r for r in self.results if r.blocks_gate]

    def budget_summary(self) -> dict[str, str]:
        return {r.slo_name: r.budget_status.value for r in self.results}

    def summary(self) -> str:
        status = "COMPLIANT ✅" if self.overall_compliant else "VIOLATION ❌"
        lines = [
            f"SLOReport: {status}",
            f"  SLOs checked: {len(self.results)}",
            f"  Violations: {len(self.violations())}",
            f"  Gate-blocking: {len(self.gate_blocking())}",
        ]
        for r in self.results:
            icon = "✅" if r.compliant else "❌"
            lines.append(
                f"  {icon} [{r.budget_status.value.upper()}] {r.slo_name}: "
                f"{r.current_value:.4f} vs target {r.target:.4f}"
            )
        return "\n".join(lines)


# ── SLOChecker ─────────────────────────────────────────────────────────────────

class SLOChecker:
    """Evaluates observed metrics against registered SLO definitions.

    Built-in SLOs for the credit risk service:
      - p99 latency < 500ms  (99.9% of requests)
      - error rate < 1%      (99.9% of requests)
      - model AUC ≥ 0.72     (95% of daily evaluations)
      - feature freshness < 26h (99% of hours)
      - approval rate 60–80% (99% of hours)

    Args:
        latency_p99_ms:   Maximum acceptable p99 latency in milliseconds.
        max_error_rate:   Maximum acceptable error rate fraction.
        min_auc:          Minimum acceptable model AUC.
        max_freshness_h:  Maximum acceptable feature age in hours.
        min_approval:     Minimum acceptable approval rate.
        max_approval:     Maximum acceptable approval rate.
        max_default_rate: Maximum acceptable default rate.
        window_hours:     Error budget window (default: 720h = 30 days).
    """

    def __init__(
        self,
        latency_p99_ms: float = 500.0,
        max_error_rate: float = 0.01,
        min_auc: float = 0.72,
        max_freshness_h: float = 26.0,
        min_approval: float = 0.60,
        max_approval: float = 0.80,
        max_default_rate: float = 0.35,
        window_hours: float = 720.0,
    ) -> None:
        self.latency_p99_ms = latency_p99_ms
        self.max_error_rate = max_error_rate
        self.min_auc = min_auc
        self.max_freshness_h = max_freshness_h
        self.min_approval = min_approval
        self.max_approval = max_approval
        self.max_default_rate = max_default_rate
        self.window_hours = window_hours

    # ── Individual SLO checks ─────────────────────────────────────────────────

    def check_latency(self, p99_ms: float) -> SLOResult:
        """SLO: p99 latency must be below latency_p99_ms."""
        compliant = p99_ms <= self.latency_p99_ms
        # Budget consumed proportional to excess latency (capped at 1.0)
        consumed = min(1.0, max(0.0, (p99_ms - self.latency_p99_ms) / self.latency_p99_ms)) if not compliant else 0.0
        return SLOResult(
            slo_name="p99_latency",
            slo_type=SLOType.LATENCY,
            current_value=p99_ms,
            target=self.latency_p99_ms,
            compliant=compliant,
            budget_consumed_pct=consumed,
            budget_status=_budget_status(consumed),
            description=f"p99 latency {p99_ms:.1f}ms vs SLO {self.latency_p99_ms:.1f}ms",
        )

    def check_error_rate(self, error_rate: float) -> SLOResult:
        """SLO: request error rate must be below max_error_rate."""
        compliant = error_rate <= self.max_error_rate
        consumed = min(1.0, error_rate / max(self.max_error_rate, 1e-9)) if not compliant else error_rate / max(self.max_error_rate, 1e-9)
        return SLOResult(
            slo_name="error_rate",
            slo_type=SLOType.ERROR_RATE,
            current_value=error_rate,
            target=self.max_error_rate,
            compliant=compliant,
            budget_consumed_pct=consumed,
            budget_status=_budget_status(consumed),
            description=f"Error rate {error_rate:.3%} vs SLO {self.max_error_rate:.3%}",
        )

    def check_model_quality(self, auc: float) -> SLOResult:
        """SLO: model AUC must be at or above min_auc."""
        compliant = auc >= self.min_auc
        deficit = max(0.0, self.min_auc - auc)
        consumed = min(1.0, deficit / max(self.min_auc, 1e-9))
        return SLOResult(
            slo_name="model_auc",
            slo_type=SLOType.MODEL_QUALITY,
            current_value=auc,
            target=self.min_auc,
            compliant=compliant,
            budget_consumed_pct=consumed,
            budget_status=_budget_status(consumed),
            description=f"Model AUC {auc:.4f} vs SLO {self.min_auc:.4f}",
        )

    def check_feature_freshness(self, max_age_hours: float) -> SLOResult:
        """SLO: maximum feature age must be below max_freshness_h."""
        compliant = max_age_hours <= self.max_freshness_h
        consumed = min(1.0, max_age_hours / max(self.max_freshness_h, 1e-9)) if not compliant else 0.0
        return SLOResult(
            slo_name="feature_freshness",
            slo_type=SLOType.FEATURE_FRESHNESS,
            current_value=max_age_hours,
            target=self.max_freshness_h,
            compliant=compliant,
            budget_consumed_pct=consumed,
            budget_status=_budget_status(consumed),
            description=f"Feature age {max_age_hours:.1f}h vs SLO {self.max_freshness_h:.1f}h",
        )

    def check_approval_rate(self, rate: float) -> SLOResult:
        """SLO: approval rate must be between min_approval and max_approval."""
        compliant = self.min_approval <= rate <= self.max_approval
        if compliant:
            consumed = 0.0
        elif rate < self.min_approval:
            consumed = min(1.0, (self.min_approval - rate) / self.min_approval)
        else:
            consumed = min(1.0, (rate - self.max_approval) / (1.0 - self.max_approval + 1e-9))
        return SLOResult(
            slo_name="approval_rate",
            slo_type=SLOType.BUSINESS,
            current_value=rate,
            target=self.min_approval,
            compliant=compliant,
            budget_consumed_pct=consumed,
            budget_status=_budget_status(consumed),
            description=f"Approval rate {rate:.1%} vs SLO [{self.min_approval:.0%}–{self.max_approval:.0%}]",
        )

    def check_default_rate(self, rate: float) -> SLOResult:
        """SLO: observed default rate must be below max_default_rate."""
        compliant = rate <= self.max_default_rate
        consumed = min(1.0, rate / max(self.max_default_rate, 1e-9)) if not compliant else 0.0
        return SLOResult(
            slo_name="default_rate",
            slo_type=SLOType.BUSINESS,
            current_value=rate,
            target=self.max_default_rate,
            compliant=compliant,
            budget_consumed_pct=consumed,
            budget_status=_budget_status(consumed),
            description=f"Default rate {rate:.1%} vs SLO {self.max_default_rate:.1%}",
        )

    # ── Full run ──────────────────────────────────────────────────────────────

    def run_all(self, metrics: dict[str, float]) -> SLOReport:
        """Run all SLO checks against a metrics dictionary.

        Expected keys (all optional — missing keys skip the SLO):
          p99_latency_ms, error_rate, model_auc,
          max_feature_age_hours, approval_rate, default_rate

        Returns:
            SLOReport with overall_compliant=True if all SLOs pass.
        """
        results: list[SLOResult] = []

        if "p99_latency_ms" in metrics:
            results.append(self.check_latency(metrics["p99_latency_ms"]))
        if "error_rate" in metrics:
            results.append(self.check_error_rate(metrics["error_rate"]))
        if "model_auc" in metrics:
            results.append(self.check_model_quality(metrics["model_auc"]))
        if "max_feature_age_hours" in metrics:
            results.append(self.check_feature_freshness(metrics["max_feature_age_hours"]))
        if "approval_rate" in metrics:
            results.append(self.check_approval_rate(metrics["approval_rate"]))
        if "default_rate" in metrics:
            results.append(self.check_default_rate(metrics["default_rate"]))

        overall = all(r.compliant for r in results)
        return SLOReport(results=results, overall_compliant=overall)
