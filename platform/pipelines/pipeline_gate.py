"""Pipeline Gate dry-run: combines idempotency proof, retry-safety, lineage audit.

Runs the full Pipeline gate check in one call:
  - PipelineGateConfig:  what to check
  - PipelineGateReport:  typed result with per-check breakdown
  - PipelineGateRunner:  runs all checks against a SimpleDag + DagRunResult

Also contains OrchestrationSurvey: a structured comparison of Prefect,
Metaflow, Argo, SageMaker Pipelines, and Vertex AI Pipelines with
a recommend() method.

See: docs/phase5/day37_survey_pipeline_gate.md
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from pipelines.failure_modes import (
    IdempotencyProof,
    LineageAuditor,
    RetryChecker,
)

log = logging.getLogger(__name__)


# ── Pipeline Gate Config ──────────────────────────────────────────────────────

@dataclass
class PipelineGateConfig:
    """Configuration for a Pipeline gate dry-run.

    Attributes:
        required_assets:         Asset keys that must appear in lineage.
        steps_requiring_cleanup: Step names that must have cleanup_fn registered.
        idempotency_fn:          Function to test idempotency (optional).
        idempotency_inputs:      Inputs to pass to idempotency_fn.
        idempotency_run_count:   How many times to run the idempotency test.
    """

    required_assets: list[str] = field(default_factory=lambda: [
        "raw_credit_data",
        "validated_data",
        "feature_dataset",
        "trained_model",
        "validation_report",
        "champion_model",
    ])
    steps_requiring_cleanup: list[str] = field(default_factory=list)
    idempotency_fn: Any | None = None
    idempotency_inputs: Any = None
    idempotency_run_count: int = 2


# ── Pipeline Gate Report ──────────────────────────────────────────────────────

@dataclass
class PipelineGateReport:
    """Result of a Pipeline gate dry-run.

    Attributes:
        passed:               True if ALL gate checks passed.
        idempotency_passed:   Idempotency check result (None if not configured).
        retry_safety_passed:  All inspected steps are retry-safe.
        lineage_passed:       All required assets are in lineage with checksums.
        issues:               All failure messages across all checks.
        warnings:             Non-blocking concerns.
        duration_s:           Wall-clock seconds for the gate run.
        retry_reports:        Per-step retry safety reports.
    """

    passed: bool
    idempotency_passed: bool | None
    retry_safety_passed: bool
    lineage_passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    retry_reports: list[Any] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASSED ✅" if self.passed else "FAILED ❌"
        lines = [f"Pipeline Gate: {status}"]
        lines.append(f"  Idempotency:   {'✅' if self.idempotency_passed else '⚠️  (not configured)' if self.idempotency_passed is None else '❌'}")
        lines.append(f"  Retry safety:  {'✅' if self.retry_safety_passed else '❌'}")
        lines.append(f"  Lineage:       {'✅' if self.lineage_passed else '❌'}")
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"    - {issue}")
        return "\n".join(lines)


# ── Pipeline Gate Runner ──────────────────────────────────────────────────────

class PipelineGateRunner:
    """Runs the Pipeline gate dry-run.

    Args:
        config: PipelineGateConfig with all check parameters.
    """

    def __init__(self, config: PipelineGateConfig | None = None) -> None:
        self.config = config or PipelineGateConfig()

    def run(
        self,
        dag: Any | None = None,
        run_result: Any | None = None,
    ) -> PipelineGateReport:
        """Execute all gate checks.

        Args:
            dag:        SimpleDag to inspect for retry safety (optional).
            run_result: DagRunResult to audit for lineage (optional).

        Returns:
            PipelineGateReport with full results.
        """
        start = time.monotonic()
        issues: list[str] = []
        warnings: list[str] = []
        retry_reports: list[Any] = []

        # 1. Idempotency
        idempotency_passed: bool | None = None
        if self.config.idempotency_fn is not None:
            proof = IdempotencyProof(
                fn=self.config.idempotency_fn,
                run_count=self.config.idempotency_run_count,
            )
            result = proof.prove(inputs=self.config.idempotency_inputs)
            idempotency_passed = result.is_idempotent
            if not result.is_idempotent:
                issues.append(f"Idempotency: {result.failure_reason}")

        # 2. Retry safety
        retry_safety_passed = True
        if dag is not None:
            checker = RetryChecker()
            retry_reports = checker.check_all(dag)
            for report in retry_reports:
                if not report.is_retry_safe:
                    # Steps requiring cleanup that don't have it are issues
                    if report.step_name in self.config.steps_requiring_cleanup:
                        for issue in report.issues:
                            issues.append(f"Retry safety [{report.step_name}]: {issue}")
                        retry_safety_passed = False
                    else:
                        for issue in report.issues:
                            warnings.append(f"Retry warning [{report.step_name}]: {issue}")

        # 3. Lineage completeness
        lineage_passed = True
        if run_result is not None:
            auditor = LineageAuditor(required_assets=self.config.required_assets)
            lineage_report = auditor.audit(run_result)
            lineage_passed = lineage_report.complete
            for issue in lineage_report.issues:
                issues.append(f"Lineage: {issue}")

        duration = time.monotonic() - start
        passed = (
            (idempotency_passed is not False)
            and retry_safety_passed
            and lineage_passed
        )

        report = PipelineGateReport(
            passed=passed,
            idempotency_passed=idempotency_passed,
            retry_safety_passed=retry_safety_passed,
            lineage_passed=lineage_passed,
            issues=issues,
            warnings=warnings,
            duration_s=duration,
            retry_reports=retry_reports,
        )

        log.info("Pipeline gate %s in %.2fs", "PASSED" if passed else "FAILED", duration)
        return report


# ── Orchestration Survey ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrchestratorProfile:
    """Profile for one orchestration tool."""

    name: str
    mental_model: str
    asset_centric: bool
    step_caching: bool
    ml_native: bool
    local_dev_score: int    # 1–3 (3 = best)
    k8s_native: bool
    cloud_portability: int  # 1–3
    best_for: str


class OrchestrationSurvey:
    """Structured comparison of orchestration tools with a recommend() method.

    Based on Day 37 theory — see docs/phase5/day37_survey_pipeline_gate.md.
    """

    PROFILES: list[OrchestratorProfile] = [
        OrchestratorProfile(
            name="Dagster",
            mental_model="Asset graph — track what data exists",
            asset_centric=True,
            step_caching=True,
            ml_native=True,
            local_dev_score=3,
            k8s_native=False,
            cloud_portability=3,
            best_for="ML teams needing asset tracking + local dev",
        ),
        OrchestratorProfile(
            name="Prefect",
            mental_model="Flow/Task — Python-native workflows",
            asset_centric=False,
            step_caching=False,
            ml_native=False,
            local_dev_score=3,
            k8s_native=False,
            cloud_portability=3,
            best_for="Python-first teams, general-purpose workflows",
        ),
        OrchestratorProfile(
            name="Metaflow",
            mental_model="FlowSpec class with @step methods",
            asset_centric=False,
            step_caching=False,
            ml_native=True,
            local_dev_score=2,
            k8s_native=False,
            cloud_portability=2,
            best_for="AWS data science teams, parameter sweeps",
        ),
        OrchestratorProfile(
            name="Argo Workflows",
            mental_model="YAML DAG of containerised pods",
            asset_centric=False,
            step_caching=False,
            ml_native=False,
            local_dev_score=1,
            k8s_native=True,
            cloud_portability=3,
            best_for="K8s-native teams, large-scale parallel jobs",
        ),
        OrchestratorProfile(
            name="SageMaker Pipelines",
            mental_model="Python SDK → AWS managed execution",
            asset_centric=False,
            step_caching=False,
            ml_native=True,
            local_dev_score=1,
            k8s_native=False,
            cloud_portability=1,
            best_for="AWS-only orgs with SageMaker serving",
        ),
        OrchestratorProfile(
            name="Vertex AI Pipelines",
            mental_model="KFP components compiled to Vertex AI",
            asset_centric=False,
            step_caching=False,
            ml_native=True,
            local_dev_score=1,
            k8s_native=False,
            cloud_portability=2,
            best_for="GCP-first teams, BigQuery-heavy pipelines",
        ),
    ]

    def all_tools(self) -> list[str]:
        return [p.name for p in self.PROFILES]

    def get_profile(self, name: str) -> OrchestratorProfile | None:
        for p in self.PROFILES:
            if p.name.lower() == name.lower():
                return p
        return None

    def recommend(
        self,
        *,
        need_asset_centric: bool = False,
        need_step_caching: bool = False,
        need_ml_native: bool = False,
        need_k8s_native: bool = False,
        cloud: str | None = None,
        need_local_dev: bool = True,
    ) -> list[OrchestratorProfile]:
        """Recommend tools based on requirements.

        Args:
            need_asset_centric:  Must track assets as first-class.
            need_step_caching:   Must cache unchanged steps.
            need_ml_native:      Must have ML-specific features.
            need_k8s_native:     Must run natively on K8s.
            cloud:               "aws", "gcp", or None (cloud-agnostic).
            need_local_dev:      Must have good local dev experience.

        Returns:
            Sorted list of matching OrchestratorProfile objects.
        """
        results: list[tuple[int, OrchestratorProfile]] = []

        cloud_mapping = {
            "aws": "SageMaker Pipelines",
            "gcp": "Vertex AI Pipelines",
        }

        for profile in self.PROFILES:
            score = 0
            disqualified = False

            if need_asset_centric and not profile.asset_centric:
                disqualified = True
            if need_step_caching and not profile.step_caching:
                disqualified = True
            if need_k8s_native and not profile.k8s_native:
                disqualified = True

            if disqualified:
                continue

            # Score positive criteria
            if need_ml_native and profile.ml_native:
                score += 2
            if need_local_dev and profile.local_dev_score == 3:
                score += 2
            elif need_local_dev and profile.local_dev_score == 2:
                score += 1
            if cloud and cloud_mapping.get(cloud) == profile.name:
                score += 3
            if profile.asset_centric:
                score += 1

            results.append((score, profile))

        results.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in results]

    def comparison_table(self) -> list[dict[str, Any]]:
        """Return a list of dicts suitable for a comparison table."""
        return [
            {
                "tool": p.name,
                "asset_centric": p.asset_centric,
                "step_caching": p.step_caching,
                "ml_native": p.ml_native,
                "local_dev_score": p.local_dev_score,
                "k8s_native": p.k8s_native,
                "cloud_portability": p.cloud_portability,
                "best_for": p.best_for,
            }
            for p in self.PROFILES
        ]
