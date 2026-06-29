"""ML CI/CD pipeline orchestrator: code, data, and model CI stages.

Day 54 — defines the three-axis ML CI pipeline as pure-Python stages so the
same logic runs in GitLab CI jobs and in unit tests without external dependencies.

Classes:
  CIStage        — individual CI stage (name, fn, blocking)
  CIResult       — pass/fail outcome with duration and detail
  CIPipelineRun  — ordered collection of CIResult objects
  MLCIPipeline   — registers stages; runs code/data/model CI; produces a CIPipelineRun

See: docs/phase8/day54_cicd_for_ml.md
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ── CIAxis ────────────────────────────────────────────────────────────────────

class CIAxis(str, Enum):
    CODE  = "code"
    DATA  = "data"
    MODEL = "model"


# ── CIStage ───────────────────────────────────────────────────────────────────

@dataclass
class CIStage:
    """One named CI stage.

    Attributes:
        name:     Unique stage identifier.
        axis:     CODE / DATA / MODEL.
        fn:       Callable() -> dict — runs the check; returns details dict.
        blocking: If True, a failure stops subsequent stages in the same run.
    """

    name: str
    axis: CIAxis
    fn: Callable[[], dict[str, Any]]
    blocking: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CIStage name cannot be empty")


# ── CIResult ──────────────────────────────────────────────────────────────────

@dataclass
class CIResult:
    """Outcome of one CI stage execution.

    Attributes:
        stage_name: Stage that was run.
        axis:       CODE / DATA / MODEL.
        passed:     True if stage reported success.
        duration_s: Wall-clock time in seconds.
        details:    Arbitrary metadata from the stage function.
        error:      Exception message if stage raised unexpectedly.
    """

    stage_name: str
    axis: CIAxis
    passed: bool
    duration_s: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def blocking_failure(self) -> bool:
        return not self.passed


# ── CIPipelineRun ─────────────────────────────────────────────────────────────

@dataclass
class CIPipelineRun:
    """Ordered results from one full CI pipeline execution.

    Attributes:
        results:       All CIResult objects in execution order.
        overall_passed: True if all blocking stages passed.
    """

    results: list[CIResult] = field(default_factory=list)
    overall_passed: bool = True

    def by_axis(self, axis: CIAxis) -> list[CIResult]:
        return [r for r in self.results if r.axis == axis]

    def failures(self) -> list[CIResult]:
        return [r for r in self.results if not r.passed]

    def total_duration_s(self) -> float:
        return sum(r.duration_s for r in self.results)

    def summary(self) -> str:
        status = "PASSED ✅" if self.overall_passed else "FAILED ❌"
        lines = [
            f"CIPipelineRun: {status}",
            f"  Stages:   {len(self.results)}",
            f"  Failures: {len(self.failures())}",
            f"  Duration: {self.total_duration_s():.2f}s",
        ]
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            lines.append(f"  {icon} [{r.axis.value}] {r.stage_name} ({r.duration_s:.3f}s)")
        return "\n".join(lines)


# ── MLCIPipeline ──────────────────────────────────────────────────────────────

class MLCIPipeline:
    """Registers and runs three-axis (code/data/model) CI stages.

    Stages execute in registration order. If a blocking stage fails,
    subsequent stages in the same axis are skipped.

    Usage::

        pipeline = MLCIPipeline()
        pipeline.register(CIStage("lint", CIAxis.CODE, run_lint))
        pipeline.register(CIStage("schema", CIAxis.DATA, run_schema_check))
        run = pipeline.run()
        assert run.overall_passed
    """

    def __init__(self) -> None:
        self._stages: list[CIStage] = []

    def register(self, stage: CIStage) -> None:
        """Add a stage to the pipeline (appended to execution order)."""
        self._stages.append(stage)

    def stage_names(self) -> list[str]:
        return [s.name for s in self._stages]

    def __len__(self) -> int:
        return len(self._stages)

    def run(self, axes: list[CIAxis] | None = None) -> CIPipelineRun:
        """Execute all registered stages (or only stages matching `axes`).

        Args:
            axes: Filter to run only stages of these axes. None = run all.

        Returns:
            CIPipelineRun with all results and overall_passed flag.
        """
        results: list[CIResult] = []
        failed_axes: set[CIAxis] = set()
        overall_passed = True

        for stage in self._stages:
            if axes is not None and stage.axis not in axes:
                continue

            # Skip if a blocking stage in this axis already failed
            if stage.axis in failed_axes:
                results.append(CIResult(
                    stage_name=stage.name,
                    axis=stage.axis,
                    passed=False,
                    error="skipped — earlier blocking stage failed",
                ))
                overall_passed = False
                continue

            t0 = time.monotonic()
            try:
                details = stage.fn()
                passed = bool(details.get("passed", True))
                result = CIResult(
                    stage_name=stage.name,
                    axis=stage.axis,
                    passed=passed,
                    duration_s=time.monotonic() - t0,
                    details=details,
                )
            except Exception as exc:  # noqa: BLE001
                result = CIResult(
                    stage_name=stage.name,
                    axis=stage.axis,
                    passed=False,
                    duration_s=time.monotonic() - t0,
                    error=str(exc),
                )

            results.append(result)

            if not result.passed:
                overall_passed = False
                if stage.blocking:
                    failed_axes.add(stage.axis)

        return CIPipelineRun(results=results, overall_passed=overall_passed)
