"""DAG primitives — steps, assets, run context, retry, backfill planning.

Implements core orchestration concepts without requiring Dagster/Airflow:
  - StepResult / AssetMaterialization: typed outputs from each pipeline step
  - RunContext: carries run_id, partition, config, and lineage across steps
  - RetryPolicy: exponential back-off with max attempts
  - DagStep: a callable unit that respects RetryPolicy and cleans up on failure
  - SimpleDag: ordered list of steps with topological dependency tracking
  - BackfillPlanner: computes which partitions need to be (re-)run

See: docs/phase5/day31_orchestration_principles.md
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

log = logging.getLogger(__name__)


# ── Run Status ────────────────────────────────────────────────────────────────

class StepStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


# ── Typed step output ─────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Output record for a single pipeline step.

    Attributes:
        step_name:    Name of the step that produced this result.
        status:       Terminal status (SUCCESS / FAILED / SKIPPED).
        output:       Arbitrary output value from the step function.
        error:        Exception message if status is FAILED.
        duration_s:   Wall-clock seconds the step took.
        attempt:      Which attempt succeeded (1 = first try).
        metadata:     Free-form key-value pairs for lineage.
    """

    step_name: str
    status: StepStatus
    output: Any = None
    error: str | None = None
    duration_s: float = 0.0
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == StepStatus.SUCCESS

    @property
    def failed(self) -> bool:
        return self.status == StepStatus.FAILED


@dataclass
class AssetMaterialization:
    """Records that a data asset was written to a persistent location.

    Attributes:
        asset_key:      Logical name of the asset (e.g. "feature_dataset").
        path:           File system or object-store path.
        row_count:      Number of rows (for tabular assets).
        checksum:       SHA-256 of the written file for lineage.
        run_id:         Run that produced this materialisation.
        partition:      Time partition key (e.g. "2024-01").
        extra:          Additional metadata (schema version, model version, etc.)
    """

    asset_key: str
    path: str
    row_count: int = 0
    checksum: str | None = None
    run_id: str = ""
    partition: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_key": self.asset_key,
            "path": self.path,
            "row_count": self.row_count,
            "checksum": self.checksum,
            "run_id": self.run_id,
            "partition": self.partition,
            **self.extra,
        }


# ── Run Context ───────────────────────────────────────────────────────────────

@dataclass
class RunContext:
    """Shared state passed through every step in a pipeline run.

    Attributes:
        run_id:         Unique ID for this pipeline execution.
        partition:      Time partition key for this run (None for unpartitioned).
        config:         User-supplied config dict (overrides defaults).
        materializations: Asset materializations recorded so far this run.
        step_results:   Ordered list of StepResult for every executed step.
    """

    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    partition: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    materializations: list[AssetMaterialization] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)

    def record_materialization(self, mat: AssetMaterialization) -> None:
        """Attach a run_id and record the materialisation."""
        mat.run_id = self.run_id
        self.materializations.append(mat)
        log.info(
            "Asset materialised: key=%s path=%s rows=%d run_id=%s",
            mat.asset_key, mat.path, mat.row_count, self.run_id,
        )

    def get_materialization(self, asset_key: str) -> AssetMaterialization | None:
        """Return the most recent materialisation of an asset in this run."""
        for mat in reversed(self.materializations):
            if mat.asset_key == asset_key:
                return mat
        return None

    def lineage_summary(self) -> list[dict[str, Any]]:
        """Return ordered list of asset materialisation dicts for audit."""
        return [m.to_dict() for m in self.materializations]


# ── Retry Policy ─────────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """Defines retry behaviour for a DagStep.

    Attributes:
        max_attempts:    Total attempts (1 = no retry).
        delay_seconds:   Initial delay before first retry.
        backoff_factor:  Multiply delay by this factor each attempt.
        retryable_errors: Exception types that trigger a retry.
                          If empty, retries on any exception.
    """

    max_attempts: int = 3
    delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    retryable_errors: tuple[type[Exception], ...] = ()

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be >= 0")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")

    def should_retry(self, exc: Exception, attempt: int) -> bool:
        """Return True if the exception is retryable and attempts remain."""
        if attempt >= self.max_attempts:
            return False
        if self.retryable_errors:
            return isinstance(exc, self.retryable_errors)
        return True

    def wait_seconds(self, attempt: int) -> float:
        """Seconds to wait before attempt `attempt` (1-indexed, first retry = 2)."""
        return self.delay_seconds * (self.backoff_factor ** (attempt - 1))


# ── DAG Step ─────────────────────────────────────────────────────────────────

@dataclass
class DagStep:
    """A single unit of computation in the DAG with retry and cleanup.

    The step function receives (ctx: RunContext, **kwargs) and should:
      - Compute an output
      - Record any AssetMaterialization on ctx
      - Return a value (or None)

    Cleanup is called with (ctx, error) if the step fails, to remove partial
    outputs before the next retry attempt.

    Args:
        name:           Step name (must be unique in the DAG).
        fn:             The function to execute: (RunContext, **kwargs) → Any.
        retry_policy:   How to handle failures.
        cleanup_fn:     Optional cleanup on failure: (RunContext, Exception) → None.
        depends_on:     Names of steps that must succeed before this one runs.
    """

    name: str
    fn: Callable[..., Any]
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    cleanup_fn: Callable[..., None] | None = None
    depends_on: list[str] = field(default_factory=list)

    def run(self, ctx: RunContext, **kwargs: Any) -> StepResult:
        """Execute the step with retry logic.

        Returns a StepResult regardless of success or failure.
        """
        log.info("Step %r starting (run_id=%s)", self.name, ctx.run_id)
        start = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(1, self.retry_policy.max_attempts + 1):
            try:
                output = self.fn(ctx, **kwargs)
                duration = time.monotonic() - start
                result = StepResult(
                    step_name=self.name,
                    status=StepStatus.SUCCESS,
                    output=output,
                    duration_s=duration,
                    attempt=attempt,
                )
                log.info(
                    "Step %r succeeded (attempt=%d, %.2fs)", self.name, attempt, duration
                )
                ctx.step_results.append(result)
                return result

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning(
                    "Step %r failed attempt %d/%d: %s",
                    self.name, attempt, self.retry_policy.max_attempts, exc,
                )
                if self.cleanup_fn is not None:
                    try:
                        self.cleanup_fn(ctx, exc)
                    except Exception as cleanup_exc:  # noqa: BLE001
                        log.error("Cleanup for step %r failed: %s", self.name, cleanup_exc)

                if self.retry_policy.should_retry(exc, attempt):
                    wait = self.retry_policy.wait_seconds(attempt)
                    log.info("Retrying step %r in %.1fs...", self.name, wait)
                    time.sleep(wait)
                else:
                    break

        duration = time.monotonic() - start
        result = StepResult(
            step_name=self.name,
            status=StepStatus.FAILED,
            error=str(last_error),
            duration_s=duration,
            attempt=self.retry_policy.max_attempts,
        )
        ctx.step_results.append(result)
        return result


# ── Simple DAG ────────────────────────────────────────────────────────────────

@dataclass
class DagRunResult:
    """Result of running a SimpleDag.

    Attributes:
        run_id:         ID of the run.
        succeeded:      True if all non-skipped steps succeeded.
        step_results:   Per-step results in execution order.
        materializations: Asset materializations produced.
        duration_s:     Total wall-clock time.
    """

    run_id: str
    succeeded: bool
    step_results: list[StepResult]
    materializations: list[AssetMaterialization]
    duration_s: float

    @property
    def failed_steps(self) -> list[str]:
        return [r.step_name for r in self.step_results if r.failed]

    @property
    def skipped_steps(self) -> list[str]:
        return [r.step_name for r in self.step_results if r.status == StepStatus.SKIPPED]


class SimpleDag:
    """Ordered list of DagSteps with dependency-based execution.

    Steps are executed in registration order. A step is SKIPPED if any of
    its dependencies failed.

    Args:
        name: Human-readable name for this DAG.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._steps: list[DagStep] = []

    def add_step(self, step: DagStep) -> "SimpleDag":
        """Append a step. Returns self for chaining."""
        self._steps.append(step)
        return self

    @property
    def step_names(self) -> list[str]:
        return [s.name for s in self._steps]

    def run(
        self,
        partition: str | None = None,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> DagRunResult:
        """Execute all steps respecting dependencies.

        Args:
            partition: Time partition key (passed into RunContext).
            config:    Override config values.
            run_id:    Explicit run ID (auto-generated if None).

        Returns:
            DagRunResult with all step outcomes and materializations.
        """
        ctx = RunContext(
            run_id=run_id or str(uuid.uuid4()),
            partition=partition,
            config=config or {},
        )
        log.info("DAG %r starting (run_id=%s, partition=%s)", self.name, ctx.run_id, partition)
        start = time.monotonic()

        failed_names: set[str] = set()

        for step in self._steps:
            # Check if any dependency failed
            blocking = [d for d in step.depends_on if d in failed_names]
            if blocking:
                log.warning(
                    "Step %r skipped — dependencies failed: %s", step.name, blocking
                )
                ctx.step_results.append(StepResult(
                    step_name=step.name,
                    status=StepStatus.SKIPPED,
                    metadata={"blocked_by": blocking},
                ))
                continue

            result = step.run(ctx)
            if result.failed:
                failed_names.add(step.name)

        total = time.monotonic() - start
        succeeded = len(failed_names) == 0
        log.info(
            "DAG %r finished in %.2fs — %s (failed: %s)",
            self.name, total, "SUCCESS" if succeeded else "FAILED", list(failed_names),
        )

        return DagRunResult(
            run_id=ctx.run_id,
            succeeded=succeeded,
            step_results=ctx.step_results,
            materializations=ctx.materializations,
            duration_s=total,
        )


# ── Backfill Planner ──────────────────────────────────────────────────────────

class PartitionStatus(str, Enum):
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    MISSING = "MISSING"


@dataclass
class BackfillPlan:
    """Result of planning a backfill run.

    Attributes:
        partitions_to_run:  Partitions that need to be executed.
        partitions_to_skip: Partitions already complete.
        status_map:         partition → PartitionStatus for all input partitions.
    """

    partitions_to_run: list[str]
    partitions_to_skip: list[str]
    status_map: dict[str, PartitionStatus]


def plan_backfill(
    all_partitions: list[str],
    completed: set[str] | None = None,
    failed: set[str] | None = None,
    *,
    rerun_failed: bool = True,
) -> BackfillPlan:
    """Compute which partitions to run and which to skip.

    Args:
        all_partitions: Full list of partition keys to consider.
        completed:      Partitions that already completed successfully.
        failed:         Partitions that previously failed.
        rerun_failed:   If True, failed partitions are included in to_run.

    Returns:
        BackfillPlan with the execution plan.
    """
    completed = completed or set()
    failed = failed or set()

    status_map: dict[str, PartitionStatus] = {}
    to_run: list[str] = []
    to_skip: list[str] = []

    for p in all_partitions:
        if p in completed:
            status_map[p] = PartitionStatus.COMPLETE
            to_skip.append(p)
        elif p in failed:
            status_map[p] = PartitionStatus.FAILED
            if rerun_failed:
                to_run.append(p)
            else:
                to_skip.append(p)
        else:
            status_map[p] = PartitionStatus.MISSING
            to_run.append(p)

    return BackfillPlan(
        partitions_to_run=to_run,
        partitions_to_skip=to_skip,
        status_map=status_map,
    )


# ── Utility ───────────────────────────────────────────────────────────────────

def asset_checksum(content: bytes) -> str:
    """SHA-256 hex digest of bytes — used to fingerprint asset content."""
    return hashlib.sha256(content).hexdigest()
