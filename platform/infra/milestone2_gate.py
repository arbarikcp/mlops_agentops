"""Milestone 2 Gate — 12+ checks across 6 production gates for Phase 12.

Day 90: The Milestone 2 Gate validates that the full AWS Cloud MLOps backbone
is production-ready across six dimensions:
  1. REPRODUCIBILITY — trace endpoint → model package → training job → data
  2. SERVING         — endpoint health, latency, and rollback capability
  3. PIPELINE        — SageMaker Pipeline or Argo Workflow ran successfully
  4. MONITORING      — drift and bias monitors are active
  5. SECURITY        — KMS + IAM + PrivateLink controls in place
  6. PORTABILITY     — core layer is cloud-agnostic (score >= 0.6)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class GateDimension(str, Enum):
    REPRODUCIBILITY = "reproducibility"
    SERVING = "serving"
    PIPELINE = "pipeline"
    MONITORING = "monitoring"
    SECURITY = "security"
    PORTABILITY = "portability"


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    WARN = "warn"


# ── Gate check ────────────────────────────────────────────────────────────────


@dataclass
class M2GateCheck:
    """A single check within the Milestone 2 Gate.

    Each check maps to a specific production requirement. Checks are
    grouped by GateDimension for clarity in the report.
    """

    check_id: str
    dimension: GateDimension
    description: str
    required: bool = True  # if False, FAIL becomes WARN
    status: CheckStatus = CheckStatus.SKIP
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.check_id:
            raise ValueError("check_id must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")

    def run(self, predicate: Callable[[], bool], detail: str = "") -> "M2GateCheck":
        """Execute the check predicate and record result."""
        try:
            passed = predicate()
            self.status = CheckStatus.PASS if passed else (
                CheckStatus.FAIL if self.required else CheckStatus.WARN
            )
            self.detail = detail
        except Exception as exc:
            self.status = CheckStatus.FAIL if self.required else CheckStatus.WARN
            self.detail = f"Exception: {exc}"
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkId": self.check_id,
            "dimension": self.dimension.value,
            "description": self.description,
            "required": self.required,
            "status": self.status.value,
            "detail": self.detail,
        }


# ── Gate Report ───────────────────────────────────────────────────────────────


@dataclass
class M2GateReport:
    """Complete Milestone 2 Gate report."""

    gate_name: str
    environment: str
    checks: list[M2GateCheck] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.gate_name:
            raise ValueError("gate_name must not be empty")
        if not self.environment:
            raise ValueError("environment must not be empty")

    @property
    def passed_checks(self) -> list[M2GateCheck]:
        return [c for c in self.checks if c.status == CheckStatus.PASS]

    @property
    def failed_checks(self) -> list[M2GateCheck]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def warn_checks(self) -> list[M2GateCheck]:
        return [c for c in self.checks if c.status == CheckStatus.WARN]

    @property
    def is_passed(self) -> bool:
        """Gate passes if all required checks pass (warns are OK)."""
        return len(self.failed_checks) == 0

    def by_dimension(self) -> dict[str, list[M2GateCheck]]:
        result: dict[str, list[M2GateCheck]] = {}
        for check in self.checks:
            result.setdefault(check.dimension.value, []).append(check)
        return result

    def dimension_pass_rate(self, dim: GateDimension) -> float:
        dim_checks = [c for c in self.checks if c.dimension == dim]
        if not dim_checks:
            return 0.0
        passed = sum(1 for c in dim_checks if c.status == CheckStatus.PASS)
        return passed / len(dim_checks)

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.checks),
            "passed": len(self.passed_checks),
            "failed": len(self.failed_checks),
            "warnings": len(self.warn_checks),
            "gateStatus": "PASSED" if self.is_passed else "FAILED",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "gateName": self.gate_name,
            "environment": self.environment,
            "summary": self.summary(),
            "dimensionScores": {
                dim.value: round(self.dimension_pass_rate(dim), 2)
                for dim in GateDimension
            },
            "checks": [c.to_dict() for c in self.checks],
        }


# ── Milestone 2 Gate ──────────────────────────────────────────────────────────


@dataclass
class Milestone2Gate:
    """Milestone 2 Gate — validates the AWS Cloud MLOps backbone is production-ready.

    Runs 12+ checks across 6 dimensions. Each check is implemented as a
    predicate function over the deployment context dict.

    The gate is designed for dry-run mode (no real AWS calls) — predicates
    validate structural properties of the configuration, not live service state.
    """

    environment: str
    deployment_context: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.environment:
            raise ValueError("environment must not be empty")
        if not self.deployment_context:
            raise ValueError("deployment_context must not be empty")

    def _build_checks(self) -> list[M2GateCheck]:
        ctx = self.deployment_context
        checks: list[M2GateCheck] = []

        # ── 1. REPRODUCIBILITY (3 checks) ─────────────────────────────────────
        checks.append(M2GateCheck(
            "R-01", GateDimension.REPRODUCIBILITY,
            "Model package ARN traceable to training job",
        ).run(
            lambda: bool(ctx.get("model_package_arn")),
            detail=f"model_package_arn={ctx.get('model_package_arn', 'MISSING')}",
        ))
        checks.append(M2GateCheck(
            "R-02", GateDimension.REPRODUCIBILITY,
            "Training job linked to data version (DVC commit)",
        ).run(
            lambda: bool(ctx.get("dvc_commit_hash")),
            detail=f"dvc_commit={ctx.get('dvc_commit_hash', 'MISSING')}",
        ))
        checks.append(M2GateCheck(
            "R-03", GateDimension.REPRODUCIBILITY,
            "Training image SHA pinned in model package metadata",
        ).run(
            lambda: bool(ctx.get("image_sha")),
            detail=f"image_sha={ctx.get('image_sha', 'MISSING')}",
        ))

        # ── 2. SERVING (3 checks) ──────────────────────────────────────────────
        checks.append(M2GateCheck(
            "S-01", GateDimension.SERVING,
            "SageMaker endpoint in InService state",
        ).run(
            lambda: ctx.get("endpoint_status") == "InService",
            detail=f"endpoint_status={ctx.get('endpoint_status', 'MISSING')}",
        ))
        checks.append(M2GateCheck(
            "S-02", GateDimension.SERVING,
            "Endpoint p99 latency < 200ms (last 1h)",
        ).run(
            lambda: float(ctx.get("endpoint_p99_ms", 999)) < 200,
            detail=f"p99_ms={ctx.get('endpoint_p99_ms', 'MISSING')}",
        ))
        checks.append(M2GateCheck(
            "S-03", GateDimension.SERVING,
            "Rollback endpoint config available",
        ).run(
            lambda: bool(ctx.get("previous_endpoint_config")),
            detail=f"rollback_config={ctx.get('previous_endpoint_config', 'MISSING')}",
        ))

        # ── 3. PIPELINE (2 checks) ────────────────────────────────────────────
        checks.append(M2GateCheck(
            "P-01", GateDimension.PIPELINE,
            "SageMaker Pipeline last execution status = Succeeded",
        ).run(
            lambda: ctx.get("pipeline_last_status") == "Succeeded",
            detail=f"pipeline_status={ctx.get('pipeline_last_status', 'MISSING')}",
        ))
        checks.append(M2GateCheck(
            "P-02", GateDimension.PIPELINE,
            "Pipeline execution ID linked to model package (lineage)",
        ).run(
            lambda: bool(ctx.get("pipeline_execution_id")),
            detail=f"execution_id={ctx.get('pipeline_execution_id', 'MISSING')}",
        ))

        # ── 4. MONITORING (2 checks) ──────────────────────────────────────────
        checks.append(M2GateCheck(
            "M-01", GateDimension.MONITORING,
            "Data quality monitor schedule is active",
        ).run(
            lambda: ctx.get("dq_monitor_status") == "Scheduled",
            detail=f"dq_monitor={ctx.get('dq_monitor_status', 'MISSING')}",
        ))
        checks.append(M2GateCheck(
            "M-02", GateDimension.MONITORING,
            "Clarify bias monitor has run within last 7 days",
        ).run(
            lambda: ctx.get("clarify_last_run_days", 99) <= 7,
            detail=f"clarify_last_run_days={ctx.get('clarify_last_run_days', 'MISSING')}",
        ))

        # ── 5. SECURITY (3 checks) ────────────────────────────────────────────
        checks.append(M2GateCheck(
            "SEC-01", GateDimension.SECURITY,
            "Model artifacts encrypted with KMS CMK",
        ).run(
            lambda: bool(ctx.get("kms_key_arn")),
            detail=f"kms_key={ctx.get('kms_key_arn', 'MISSING')}",
        ))
        checks.append(M2GateCheck(
            "SEC-02", GateDimension.SECURITY,
            "SageMaker VPC mode enabled (PrivateLink endpoints present)",
        ).run(
            lambda: bool(ctx.get("privatelink_endpoints")) and len(ctx["privatelink_endpoints"]) >= 3,
            detail=f"privatelink_count={len(ctx.get('privatelink_endpoints', []))}",
        ))
        checks.append(M2GateCheck(
            "SEC-03", GateDimension.SECURITY,
            "IAM roles follow least-privilege (no wildcard actions on sensitive resources)",
        ).run(
            lambda: not ctx.get("iam_has_wildcard_actions", True),
            detail=f"wildcard_actions={ctx.get('iam_has_wildcard_actions', 'MISSING')}",
        ))

        # ── 6. PORTABILITY (2 checks) ─────────────────────────────────────────
        checks.append(M2GateCheck(
            "PORT-01", GateDimension.PORTABILITY,
            "Portability score >= 0.6 (majority of components cloud-agnostic)",
        ).run(
            lambda: float(ctx.get("portability_score", 0)) >= 0.6,
            detail=f"portability_score={ctx.get('portability_score', 'MISSING')}",
        ))
        checks.append(M2GateCheck(
            "PORT-02", GateDimension.PORTABILITY,
            "Core serving layer uses KServe (cloud-agnostic) or documented cloud-specific justification",
            required=False,  # advisory — WARN if missing, not FAIL
        ).run(
            lambda: ctx.get("serving_is_portable", False),
            detail=f"serving_portable={ctx.get('serving_is_portable', 'MISSING')}",
        ))

        return checks

    def run(self) -> M2GateReport:
        """Execute all checks and return the gate report."""
        report = M2GateReport(
            gate_name=f"milestone2-gate-{self.environment}",
            environment=self.environment,
        )
        report.checks = self._build_checks()
        return report

    @classmethod
    def default_context(
        cls,
        model_package_arn: str = "arn:aws:sagemaker:us-east-1:123456789012:model-package/credit-risk-models/1",
        endpoint_status: str = "InService",
        endpoint_p99_ms: float = 120.0,
        pipeline_status: str = "Succeeded",
        dq_monitor_status: str = "Scheduled",
        kms_key_arn: str = "arn:aws:kms:us-east-1:123456789012:key/mrk-abc123",
        portability_score: float = 0.7,
    ) -> dict[str, Any]:
        """Return a passing deployment context for dry-run testing."""
        return {
            # Reproducibility
            "model_package_arn": model_package_arn,
            "dvc_commit_hash": "abc123def456",
            "image_sha": "sha256:deadbeef1234",
            # Serving
            "endpoint_status": endpoint_status,
            "endpoint_p99_ms": endpoint_p99_ms,
            "previous_endpoint_config": "credit-risk-endpoint-config-v1",
            # Pipeline
            "pipeline_last_status": pipeline_status,
            "pipeline_execution_id": "arn:aws:sagemaker:us-east-1:123456789012:pipeline-execution/abc",
            # Monitoring
            "dq_monitor_status": dq_monitor_status,
            "clarify_last_run_days": 3,
            # Security
            "kms_key_arn": kms_key_arn,
            "privatelink_endpoints": ["sagemaker.api", "ecr.api", "ecr.dkr", "s3", "sts"],
            "iam_has_wildcard_actions": False,
            # Portability
            "portability_score": portability_score,
            "serving_is_portable": False,  # SageMaker endpoints are cloud-specific
        }

    @classmethod
    def dry_run(cls, environment: str = "prod") -> M2GateReport:
        """Run the gate against the default passing context."""
        gate = cls(environment=environment, deployment_context=cls.default_context())
        return gate.run()
