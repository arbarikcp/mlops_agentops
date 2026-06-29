"""Tests for ci/ml_pipeline.py — CIStage, CIResult, CIPipelineRun, MLCIPipeline."""
from __future__ import annotations

import pytest

from ci.ml_pipeline import CIAxis, CIPipelineRun, CIResult, CIStage, MLCIPipeline


def _pass_fn() -> dict:
    return {"passed": True, "detail": "ok"}


def _fail_fn() -> dict:
    return {"passed": False, "detail": "broken"}


def _raise_fn() -> dict:
    raise RuntimeError("unexpected error")


# ── CIStage ────────────────────────────────────────────────────────────────────

class TestCIStage:
    def test_basic(self) -> None:
        s = CIStage("lint", CIAxis.CODE, _pass_fn)
        assert s.name == "lint"
        assert s.blocking

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            CIStage("", CIAxis.CODE, _pass_fn)


# ── CIResult ───────────────────────────────────────────────────────────────────

class TestCIResult:
    def test_blocking_failure_when_failed(self) -> None:
        r = CIResult("lint", CIAxis.CODE, passed=False)
        assert r.blocking_failure

    def test_not_blocking_when_passed(self) -> None:
        r = CIResult("lint", CIAxis.CODE, passed=True)
        assert not r.blocking_failure


# ── CIPipelineRun ──────────────────────────────────────────────────────────────

class TestCIPipelineRun:
    def _run(self, results: list[CIResult]) -> CIPipelineRun:
        overall = all(r.passed for r in results)
        return CIPipelineRun(results=results, overall_passed=overall)

    def test_by_axis(self) -> None:
        run = self._run([
            CIResult("lint", CIAxis.CODE, True),
            CIResult("schema", CIAxis.DATA, False),
        ])
        assert len(run.by_axis(CIAxis.CODE)) == 1
        assert len(run.by_axis(CIAxis.DATA)) == 1

    def test_failures(self) -> None:
        run = self._run([
            CIResult("lint", CIAxis.CODE, True),
            CIResult("schema", CIAxis.DATA, False),
        ])
        assert len(run.failures()) == 1

    def test_total_duration(self) -> None:
        run = self._run([
            CIResult("a", CIAxis.CODE, True, duration_s=0.5),
            CIResult("b", CIAxis.DATA, True, duration_s=0.3),
        ])
        assert run.total_duration_s() == pytest.approx(0.8)

    def test_summary_passed(self) -> None:
        run = CIPipelineRun([], overall_passed=True)
        assert "PASSED" in run.summary()

    def test_summary_failed(self) -> None:
        run = CIPipelineRun([CIResult("x", CIAxis.CODE, False)], overall_passed=False)
        assert "FAILED" in run.summary()


# ── MLCIPipeline ───────────────────────────────────────────────────────────────

class TestMLCIPipeline:
    def test_register_and_len(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("lint", CIAxis.CODE, _pass_fn))
        assert len(p) == 1

    def test_run_all_pass(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("lint", CIAxis.CODE, _pass_fn))
        p.register(CIStage("schema", CIAxis.DATA, _pass_fn))
        run = p.run()
        assert run.overall_passed
        assert len(run.results) == 2

    def test_run_one_failure(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("lint", CIAxis.CODE, _fail_fn))
        run = p.run()
        assert not run.overall_passed

    def test_blocking_failure_skips_same_axis(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("lint", CIAxis.CODE, _fail_fn, blocking=True))
        p.register(CIStage("type-check", CIAxis.CODE, _pass_fn, blocking=True))
        run = p.run()
        # Second stage skipped because first CODE stage failed
        skipped = [r for r in run.results if "skipped" in r.error]
        assert len(skipped) == 1

    def test_non_blocking_failure_does_not_skip(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("lint", CIAxis.CODE, _fail_fn, blocking=False))
        p.register(CIStage("unit", CIAxis.CODE, _pass_fn, blocking=True))
        run = p.run()
        assert len(run.results) == 2
        assert run.results[1].passed  # second stage ran

    def test_exception_captured_as_failure(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("bad", CIAxis.MODEL, _raise_fn))
        run = p.run()
        assert not run.overall_passed
        assert "unexpected error" in run.results[0].error

    def test_axis_filter(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("lint", CIAxis.CODE, _pass_fn))
        p.register(CIStage("schema", CIAxis.DATA, _pass_fn))
        p.register(CIStage("smoke", CIAxis.MODEL, _pass_fn))
        run = p.run(axes=[CIAxis.CODE, CIAxis.DATA])
        assert len(run.results) == 2
        axes_ran = {r.axis for r in run.results}
        assert CIAxis.MODEL not in axes_ran

    def test_blocking_failure_does_not_block_other_axis(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("lint", CIAxis.CODE, _fail_fn, blocking=True))
        p.register(CIStage("schema", CIAxis.DATA, _pass_fn))  # different axis
        run = p.run()
        # DATA stage should still run
        data_results = [r for r in run.results if r.axis == CIAxis.DATA]
        assert len(data_results) == 1
        assert data_results[0].passed

    def test_stage_names(self) -> None:
        p = MLCIPipeline()
        p.register(CIStage("a", CIAxis.CODE, _pass_fn))
        p.register(CIStage("b", CIAxis.DATA, _pass_fn))
        assert p.stage_names() == ["a", "b"]
