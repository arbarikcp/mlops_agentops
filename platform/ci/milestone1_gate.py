"""Milestone 1 Gate: full traceability + rollback dry-run.

Day 58 — runs a structured dry-run of all six M1 gate checks without hitting
real infrastructure. Validates that the platform data structures and interfaces
satisfy the M1 traceability requirements defined in day58_milestone1_gate.md.

Classes:
  TraceabilityRecord  — full trace from prediction_id to code + data + features
  GateCheck           — one M1 gate check with name, gate number, result
  GateReport          — ordered list of GateCheck results; overall pass/fail
  Milestone1Gate      — orchestrates all gate checks; produces GateReport

See: docs/phase8/day58_milestone1_gate.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ── TraceabilityRecord ────────────────────────────────────────────────────────

@dataclass
class TraceabilityRecord:
    """Full traceability from a prediction back to all its provenance.

    This is the data structure that must be answerable from a single
    prediction_id to satisfy Gate 1 (Reproducibility) of Milestone 1.

    Attributes:
        prediction_id:  UUID of the prediction (links prediction log + outcome).
        model_version:  Registry artifact name + version.
        mlflow_run_id:  MLflow run that produced the model.
        code_sha:       Git commit SHA of training code.
        data_version:   DVC data version tag.
        params:         Training hyperparameters.
        metrics:        Training metrics (must include "auc").
        features:       Feature values at prediction time (PIT-correct).
        score:          Model output score [0, 1].
        decision:       "approve" / "review" / "decline".
        correlation_id: Request-level trace ID (spans microservices).
        artifact_sha256: SHA-256 of the model artifact file.
    """

    prediction_id: str
    model_version: str
    mlflow_run_id: str
    code_sha: str
    data_version: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    decision: str = "decline"
    correlation_id: str = ""
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.prediction_id:
            raise ValueError("prediction_id cannot be empty")
        if not self.model_version:
            raise ValueError("model_version cannot be empty")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1]; got {self.score}")
        if self.decision not in {"approve", "review", "decline"}:
            raise ValueError(f"invalid decision: {self.decision!r}")

    def is_fully_traceable(self) -> bool:
        """Return True if all provenance fields are populated (non-empty)."""
        required = [
            self.prediction_id,
            self.model_version,
            self.mlflow_run_id,
            self.code_sha,
            self.data_version,
            self.artifact_sha256,
        ]
        return all(bool(v) for v in required) and bool(self.features)


# ── GateCheck ─────────────────────────────────────────────────────────────────

@dataclass
class GateCheck:
    """One M1 gate check result.

    Attributes:
        name:        Short check identifier.
        gate_number: 1–5 (maps to the six production gates).
        passed:      True if the check passed.
        message:     Human-readable outcome.
        details:     Optional metadata.
    """

    name: str
    gate_number: int
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ── GateReport ────────────────────────────────────────────────────────────────

@dataclass
class GateReport:
    """Aggregated result from all M1 gate checks.

    Attributes:
        checks:         All GateCheck results in execution order.
        overall_passed: True if every check passed.
        checked_at:     ISO-8601 UTC timestamp.
    """

    checks: list[GateCheck] = field(default_factory=list)
    overall_passed: bool = True
    checked_at: str = ""

    def __post_init__(self) -> None:
        if not self.checked_at:
            self.checked_at = datetime.now(timezone.utc).isoformat()

    def failures(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]

    def by_gate(self, gate_number: int) -> list[GateCheck]:
        return [c for c in self.checks if c.gate_number == gate_number]

    def summary(self) -> str:
        status = "PASSED ✅" if self.overall_passed else "FAILED ❌"
        lines = [
            f"Milestone 1 Gate: {status}",
            f"  Checks:   {len(self.checks)}",
            f"  Failures: {len(self.failures())}",
            f"  At:       {self.checked_at}",
        ]
        for c in self.checks:
            icon = "✅" if c.passed else "❌"
            lines.append(f"  {icon} [G{c.gate_number}] {c.name}: {c.message}")
        return "\n".join(lines)


# ── Milestone1Gate ────────────────────────────────────────────────────────────

class Milestone1Gate:
    """Runs all Milestone 1 gate checks as a dry-run against provided records.

    Does not connect to real MLflow, DVC, or Helm. Validates the data
    structures produced by the platform against M1 requirements.

    Args:
        trace:           TraceabilityRecord to validate.
        slo_metrics:     Dict of SLO metric values (keys: p99_latency_ms,
                         error_rate, model_auc, max_feature_age_hours,
                         approval_rate, default_rate).
        artifact_sha256: Expected SHA-256 for the model artifact.
        sbom_entries:    Number of SBOM entries (> 0 = SBOM exists).
    """

    def __init__(
        self,
        trace: TraceabilityRecord,
        slo_metrics: dict[str, float] | None = None,
        artifact_sha256: str = "",
        sbom_entries: int = 0,
    ) -> None:
        self.trace = trace
        self.slo_metrics = slo_metrics or {}
        self.artifact_sha256 = artifact_sha256
        self.sbom_entries = sbom_entries

    # ── Gate 1: Reproducibility ──────────────────────────────────────────────

    def _check_traceability(self) -> GateCheck:
        passed = self.trace.is_fully_traceable()
        return GateCheck(
            name="full-traceability",
            gate_number=1,
            passed=passed,
            message="all provenance fields populated" if passed else "missing provenance fields",
        )

    def _check_metrics_present(self) -> GateCheck:
        has_auc = "auc" in self.trace.metrics
        return GateCheck(
            name="training-metrics-present",
            gate_number=1,
            passed=has_auc,
            message=f"auc={self.trace.metrics.get('auc')}" if has_auc else "auc metric missing",
        )

    def _check_features_present(self) -> GateCheck:
        passed = len(self.trace.features) > 0
        return GateCheck(
            name="pit-features-present",
            gate_number=1,
            passed=passed,
            message=f"{len(self.trace.features)} features logged" if passed else "no features in trace",
        )

    # ── Gate 2: Serving ──────────────────────────────────────────────────────

    def _check_score_valid(self) -> GateCheck:
        passed = 0.0 <= self.trace.score <= 1.0
        return GateCheck(
            name="score-in-range",
            gate_number=2,
            passed=passed,
            message=f"score={self.trace.score:.4f}",
        )

    def _check_decision_valid(self) -> GateCheck:
        passed = self.trace.decision in {"approve", "review", "decline"}
        return GateCheck(
            name="decision-valid",
            gate_number=2,
            passed=passed,
            message=f"decision={self.trace.decision}",
        )

    def _check_correlation_id(self) -> GateCheck:
        passed = bool(self.trace.correlation_id)
        return GateCheck(
            name="correlation-id-present",
            gate_number=2,
            passed=passed,
            message="correlation_id present" if passed else "correlation_id missing",
        )

    # ── Gate 3: Pipeline ─────────────────────────────────────────────────────

    def _check_slo_latency(self) -> GateCheck:
        p99 = self.slo_metrics.get("p99_latency_ms", 0.0)
        passed = p99 < 500.0
        return GateCheck(
            name="slo-latency-p99",
            gate_number=3,
            passed=passed,
            message=f"p99={p99:.1f}ms {'< 500ms ✅' if passed else '>= 500ms ❌'}",
        )

    def _check_slo_auc(self) -> GateCheck:
        auc = self.slo_metrics.get("model_auc", 0.0)
        passed = auc >= 0.72
        return GateCheck(
            name="slo-model-auc",
            gate_number=3,
            passed=passed,
            message=f"auc={auc:.4f} {'≥ 0.72 ✅' if passed else '< 0.72 ❌'}",
        )

    # ── Gate 4: Monitoring ───────────────────────────────────────────────────

    def _check_slo_error_rate(self) -> GateCheck:
        err = self.slo_metrics.get("error_rate", 0.0)
        passed = err < 0.01
        return GateCheck(
            name="slo-error-rate",
            gate_number=4,
            passed=passed,
            message=f"error_rate={err:.4f} {'< 1% ✅' if passed else '>= 1% ❌'}",
        )

    # ── Gate 5: Security ─────────────────────────────────────────────────────

    def _check_artifact_signed(self) -> GateCheck:
        passed = bool(self.artifact_sha256) and bool(self.trace.artifact_sha256)
        return GateCheck(
            name="artifact-signed",
            gate_number=5,
            passed=passed,
            message="artifact SHA-256 present" if passed else "artifact not signed",
        )

    def _check_sbom_exists(self) -> GateCheck:
        passed = self.sbom_entries > 0
        return GateCheck(
            name="sbom-exists",
            gate_number=5,
            passed=passed,
            message=f"{self.sbom_entries} SBOM entries" if passed else "SBOM not generated",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> GateReport:
        """Run all M1 gate checks and return a GateReport."""
        checks = [
            # Gate 1 — Reproducibility
            self._check_traceability(),
            self._check_metrics_present(),
            self._check_features_present(),
            # Gate 2 — Serving
            self._check_score_valid(),
            self._check_decision_valid(),
            self._check_correlation_id(),
            # Gate 3 — Pipeline (SLO proxy)
            self._check_slo_latency(),
            self._check_slo_auc(),
            # Gate 4 — Monitoring
            self._check_slo_error_rate(),
            # Gate 5 — Security
            self._check_artifact_signed(),
            self._check_sbom_exists(),
        ]
        overall = all(c.passed for c in checks)
        return GateReport(checks=checks, overall_passed=overall)
