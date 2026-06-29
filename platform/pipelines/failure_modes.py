"""Pipeline failure modes: idempotency proof, retry-safety, lineage audit.

Tools for verifying and documenting pipeline reliability:
  - FailureClass:       Enum for transient / deterministic / corruption
  - FailureClassifier:  Classifies exceptions into FailureClass
  - IdempotencyResult:  Output of running a step N times and comparing checksums
  - IdempotencyProof:   Runs a step fn multiple times and compares outputs
  - RetryCheckReport:   Result of verifying a DagStep is retry-safe
  - RetryChecker:       Inspects a DagStep for retry-safety signals
  - LineageAuditReport: Result of auditing materializations in a DagRunResult
  - LineageAuditor:     Verifies lineage completeness for a pipeline run

See: docs/phase5/day36_failure_modes.md
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

log = logging.getLogger(__name__)


# ── Failure Classification ────────────────────────────────────────────────────

class FailureClass(str, Enum):
    TRANSIENT = "TRANSIENT"        # retry safe — network, resource
    DETERMINISTIC = "DETERMINISTIC"  # do not retry — code bug, validation
    CORRUPTION = "CORRUPTION"      # dangerous — partial write, shared state


# Default exception mappings
_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)

_DETERMINISTIC_ERRORS: tuple[type[Exception], ...] = (
    ValueError,
    TypeError,
    KeyError,
    AttributeError,
)


class FailureClassifier:
    """Classifies exceptions into FailureClass.

    Args:
        transient_errors:    Exception types that are safe to retry.
        deterministic_errors: Exception types that should not be retried.
        default_class:       Class for unrecognised exceptions.
    """

    def __init__(
        self,
        transient_errors: tuple[type[Exception], ...] = _TRANSIENT_ERRORS,
        deterministic_errors: tuple[type[Exception], ...] = _DETERMINISTIC_ERRORS,
        default_class: FailureClass = FailureClass.TRANSIENT,
    ) -> None:
        self.transient_errors = transient_errors
        self.deterministic_errors = deterministic_errors
        self.default_class = default_class

    def classify(self, exc: Exception) -> FailureClass:
        """Return the FailureClass for the given exception.

        Priority: DETERMINISTIC > TRANSIENT > default.
        """
        if isinstance(exc, self.deterministic_errors):
            return FailureClass.DETERMINISTIC
        if isinstance(exc, self.transient_errors):
            return FailureClass.TRANSIENT
        return self.default_class

    def is_retryable(self, exc: Exception) -> bool:
        """Return True if this exception should trigger a retry."""
        return self.classify(exc) == FailureClass.TRANSIENT


# ── Idempotency Proof ─────────────────────────────────────────────────────────

@dataclass
class IdempotencyResult:
    """Result of an idempotency proof run.

    Attributes:
        is_idempotent:    True if all runs produced the same checksum.
        run_count:        Number of times the step was executed.
        checksums:        SHA-256 checksums of each run's output.
        failure_reason:   Description of what differed (if not idempotent).
    """

    is_idempotent: bool
    run_count: int
    checksums: list[str]
    failure_reason: str | None = None

    @property
    def unique_checksums(self) -> int:
        return len(set(self.checksums))


def _stable_checksum(value: Any) -> str:
    """Compute a stable SHA-256 checksum for any Python value."""
    try:
        import json
        content = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        content = repr(value)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class IdempotencyProof:
    """Proves idempotency by running a function multiple times and comparing outputs.

    Args:
        fn:            Function to test — called with the same `inputs` each time.
        checksum_fn:   How to fingerprint the output (default: stable JSON hash).
        run_count:     Number of times to run (default: 2).
    """

    def __init__(
        self,
        fn: Callable[[Any], Any],
        checksum_fn: Callable[[Any], str] | None = None,
        run_count: int = 2,
    ) -> None:
        if run_count < 2:
            raise ValueError("run_count must be >= 2 to prove idempotency")
        self.fn = fn
        self.checksum_fn = checksum_fn or _stable_checksum
        self.run_count = run_count

    def prove(self, inputs: Any = None) -> IdempotencyResult:
        """Run fn `run_count` times and compare output checksums.

        Args:
            inputs: Argument(s) passed to fn each time.

        Returns:
            IdempotencyResult with pass/fail and checksums.
        """
        checksums: list[str] = []
        for i in range(self.run_count):
            try:
                if inputs is None:
                    output = self.fn()
                else:
                    output = self.fn(inputs)
                checksums.append(self.checksum_fn(output))
            except Exception as exc:  # noqa: BLE001
                return IdempotencyResult(
                    is_idempotent=False,
                    run_count=i + 1,
                    checksums=checksums,
                    failure_reason=f"Run {i+1} raised exception: {exc}",
                )

        unique = set(checksums)
        is_idempotent = len(unique) == 1

        return IdempotencyResult(
            is_idempotent=is_idempotent,
            run_count=self.run_count,
            checksums=checksums,
            failure_reason=(
                None if is_idempotent
                else f"Checksums differed across {self.run_count} runs: {checksums}"
            ),
        )


# ── Retry Safety Checker ──────────────────────────────────────────────────────

@dataclass
class RetryCheckReport:
    """Result of inspecting a DagStep for retry safety.

    Attributes:
        step_name:        Name of the inspected step.
        is_retry_safe:    True if all safety criteria are met.
        has_cleanup:      True if the step has a cleanup_fn registered.
        max_attempts:     Number of retry attempts configured.
        issues:           List of retry-safety concern messages.
        suggestions:      Actionable suggestions to improve retry safety.
    """

    step_name: str
    is_retry_safe: bool
    has_cleanup: bool
    max_attempts: int
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class RetryChecker:
    """Inspects DagStep objects for retry-safety signals.

    Checks:
        1. Has a cleanup_fn registered
        2. max_attempts > 1 (retry is configured)
        3. delay_seconds >= 0 (won't spin-loop)
    """

    def check(self, step: Any) -> RetryCheckReport:
        """Inspect a DagStep and return a RetryCheckReport.

        Args:
            step: A pipelines.dag.DagStep instance.

        Returns:
            RetryCheckReport with issues and suggestions.
        """
        issues: list[str] = []
        suggestions: list[str] = []

        has_cleanup = step.cleanup_fn is not None
        max_attempts = step.retry_policy.max_attempts
        delay = step.retry_policy.delay_seconds

        if not has_cleanup and max_attempts > 1:
            issues.append("Step retries but has no cleanup_fn — partial outputs may persist")
            suggestions.append("Register a cleanup_fn that deletes temporary output paths")

        if max_attempts == 1:
            issues.append("Step has max_attempts=1 — no retry configured")
            suggestions.append("Increase max_attempts for transient-failure-prone steps (e.g. S3 writes)")

        if delay < 0:
            issues.append("delay_seconds < 0 — invalid configuration")

        is_safe = len(issues) == 0

        return RetryCheckReport(
            step_name=step.name,
            is_retry_safe=is_safe,
            has_cleanup=has_cleanup,
            max_attempts=max_attempts,
            issues=issues,
            suggestions=suggestions,
        )

    def check_all(self, dag: Any) -> list[RetryCheckReport]:
        """Check all steps in a SimpleDag."""
        return [self.check(step) for step in dag._steps]


# ── Lineage Auditor ───────────────────────────────────────────────────────────

@dataclass
class LineageAuditReport:
    """Result of auditing lineage completeness for a pipeline run.

    Attributes:
        complete:             True if all required assets were materialised.
        run_id:               Run ID audited.
        materialised_assets:  Asset keys present in the run.
        missing_assets:       Required assets not found in the run.
        missing_checksums:    Assets that have no checksum recorded.
        chain:                Ordered list of materialisation dicts (the lineage chain).
        issues:               Audit failure messages.
    """

    complete: bool
    run_id: str
    materialised_assets: list[str]
    missing_assets: list[str]
    missing_checksums: list[str]
    chain: list[dict[str, Any]]
    issues: list[str] = field(default_factory=list)


class LineageAuditor:
    """Audits lineage completeness for a DagRunResult.

    Args:
        required_assets: Asset keys that MUST appear in lineage for the run
                         to be considered complete.
    """

    def __init__(self, required_assets: list[str] | None = None) -> None:
        self.required_assets = required_assets or [
            "raw_credit_data",
            "validated_data",
            "feature_dataset",
            "trained_model",
            "validation_report",
            "champion_model",
        ]

    def audit(self, run_result: Any) -> LineageAuditReport:
        """Audit a DagRunResult for lineage completeness.

        Args:
            run_result: A pipelines.dag.DagRunResult instance.

        Returns:
            LineageAuditReport with completeness status and issues.
        """
        mats = run_result.materializations
        materialised_keys = [m.asset_key for m in mats]
        chain = [m.to_dict() for m in mats]

        missing_assets = [k for k in self.required_assets if k not in materialised_keys]
        missing_checksums = [
            m.asset_key for m in mats if not m.checksum
        ]

        issues: list[str] = []
        for asset in missing_assets:
            issues.append(f"Required asset '{asset}' not in lineage")
        for asset in missing_checksums:
            issues.append(f"Asset '{asset}' has no checksum — cannot verify integrity")

        complete = len(issues) == 0
        if complete:
            log.info("Lineage audit PASSED (run_id=%s, assets=%d)", run_result.run_id, len(mats))
        else:
            log.warning(
                "Lineage audit FAILED (run_id=%s): %s", run_result.run_id, issues[:3]
            )

        return LineageAuditReport(
            complete=complete,
            run_id=run_result.run_id,
            materialised_assets=materialised_keys,
            missing_assets=missing_assets,
            missing_checksums=missing_checksums,
            chain=chain,
            issues=issues,
        )

    def require(self, asset_key: str) -> "LineageAuditor":
        """Add an asset to the required set. Returns self for chaining."""
        if asset_key not in self.required_assets:
            self.required_assets.append(asset_key)
        return self
