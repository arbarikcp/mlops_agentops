"""Tests for pipelines/dag.py — DAG primitives, retry, backfill."""
from __future__ import annotations

import time

import pytest

from pipelines.dag import (
    AssetMaterialization,
    BackfillPlan,
    DagStep,
    DagRunResult,
    PartitionStatus,
    RetryPolicy,
    RunContext,
    SimpleDag,
    StepResult,
    StepStatus,
    asset_checksum,
    plan_backfill,
)


# ── RetryPolicy ───────────────────────────────────────────────────────────────

class TestRetryPolicy:
    def test_defaults(self) -> None:
        rp = RetryPolicy()
        assert rp.max_attempts == 3
        assert rp.delay_seconds == 1.0
        assert rp.backoff_factor == 2.0

    def test_invalid_max_attempts_raises(self) -> None:
        with pytest.raises(ValueError, match="max_attempts"):
            RetryPolicy(max_attempts=0)

    def test_invalid_delay_raises(self) -> None:
        with pytest.raises(ValueError, match="delay_seconds"):
            RetryPolicy(delay_seconds=-1)

    def test_invalid_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match="backoff_factor"):
            RetryPolicy(backoff_factor=0.5)

    def test_should_retry_within_attempts(self) -> None:
        rp = RetryPolicy(max_attempts=3)
        assert rp.should_retry(RuntimeError("x"), attempt=1) is True
        assert rp.should_retry(RuntimeError("x"), attempt=2) is True

    def test_should_not_retry_at_max_attempts(self) -> None:
        rp = RetryPolicy(max_attempts=3)
        assert rp.should_retry(RuntimeError("x"), attempt=3) is False

    def test_retryable_errors_filters(self) -> None:
        rp = RetryPolicy(max_attempts=3, retryable_errors=(ConnectionError,))
        assert rp.should_retry(ConnectionError(), attempt=1) is True
        assert rp.should_retry(ValueError(), attempt=1) is False

    def test_wait_seconds_exponential(self) -> None:
        rp = RetryPolicy(delay_seconds=1.0, backoff_factor=2.0)
        assert rp.wait_seconds(1) == 1.0
        assert rp.wait_seconds(2) == 2.0
        assert rp.wait_seconds(3) == 4.0


# ── RunContext ─────────────────────────────────────────────────────────────────

class TestRunContext:
    def test_run_id_generated(self) -> None:
        ctx = RunContext()
        assert len(ctx.run_id) > 0

    def test_explicit_run_id(self) -> None:
        ctx = RunContext(run_id="test-run-123")
        assert ctx.run_id == "test-run-123"

    def test_record_materialization(self) -> None:
        ctx = RunContext(run_id="r1")
        mat = AssetMaterialization(asset_key="features", path="features.parquet", row_count=1000)
        ctx.record_materialization(mat)
        assert len(ctx.materializations) == 1
        assert mat.run_id == "r1"

    def test_get_materialization(self) -> None:
        ctx = RunContext()
        mat = AssetMaterialization(asset_key="model", path="model.pkl")
        ctx.record_materialization(mat)
        result = ctx.get_materialization("model")
        assert result is not None
        assert result.asset_key == "model"

    def test_get_materialization_missing_returns_none(self) -> None:
        ctx = RunContext()
        assert ctx.get_materialization("nonexistent") is None

    def test_get_materialization_returns_latest(self) -> None:
        ctx = RunContext()
        ctx.record_materialization(AssetMaterialization("feat", "v1.parquet"))
        ctx.record_materialization(AssetMaterialization("feat", "v2.parquet"))
        assert ctx.get_materialization("feat").path == "v2.parquet"

    def test_lineage_summary(self) -> None:
        ctx = RunContext(run_id="r42")
        ctx.record_materialization(AssetMaterialization("raw", "raw.csv", row_count=500))
        summary = ctx.lineage_summary()
        assert len(summary) == 1
        assert summary[0]["asset_key"] == "raw"
        assert summary[0]["row_count"] == 500
        assert summary[0]["run_id"] == "r42"


# ── StepResult ────────────────────────────────────────────────────────────────

class TestStepResult:
    def test_succeeded_property(self) -> None:
        r = StepResult("step1", StepStatus.SUCCESS)
        assert r.succeeded is True
        assert r.failed is False

    def test_failed_property(self) -> None:
        r = StepResult("step1", StepStatus.FAILED, error="boom")
        assert r.failed is True
        assert r.succeeded is False


# ── DagStep ───────────────────────────────────────────────────────────────────

class TestDagStep:
    def test_successful_step(self) -> None:
        step = DagStep("s1", fn=lambda ctx, **kw: 42, retry_policy=RetryPolicy(max_attempts=1))
        ctx = RunContext()
        result = step.run(ctx)
        assert result.succeeded
        assert result.output == 42
        assert result.attempt == 1

    def test_failed_step_after_max_attempts(self) -> None:
        call_count = {"n": 0}

        def flaky(ctx, **kw):
            call_count["n"] += 1
            raise RuntimeError("transient error")

        step = DagStep(
            "s1", fn=flaky,
            retry_policy=RetryPolicy(max_attempts=2, delay_seconds=0, backoff_factor=1),
        )
        ctx = RunContext()
        result = step.run(ctx)
        assert result.failed
        assert "transient error" in result.error
        assert call_count["n"] == 2

    def test_retry_succeeds_on_second_attempt(self) -> None:
        attempt_store = {"n": 0}

        def flaky_then_ok(ctx, **kw):
            attempt_store["n"] += 1
            if attempt_store["n"] < 2:
                raise RuntimeError("first attempt fails")
            return "ok"

        step = DagStep(
            "s1", fn=flaky_then_ok,
            retry_policy=RetryPolicy(max_attempts=3, delay_seconds=0, backoff_factor=1),
        )
        ctx = RunContext()
        result = step.run(ctx)
        assert result.succeeded
        assert result.attempt == 2
        assert result.output == "ok"

    def test_cleanup_called_on_failure(self) -> None:
        cleaned = {"called": False}

        def bad_fn(ctx, **kw):
            raise RuntimeError("fail")

        def cleanup(ctx, exc):
            cleaned["called"] = True

        step = DagStep(
            "s1", fn=bad_fn,
            retry_policy=RetryPolicy(max_attempts=1),
            cleanup_fn=cleanup,
        )
        step.run(RunContext())
        assert cleaned["called"] is True

    def test_step_result_appended_to_context(self) -> None:
        step = DagStep("s1", fn=lambda ctx, **kw: None, retry_policy=RetryPolicy(max_attempts=1))
        ctx = RunContext()
        step.run(ctx)
        assert len(ctx.step_results) == 1
        assert ctx.step_results[0].step_name == "s1"


# ── SimpleDag ─────────────────────────────────────────────────────────────────

class TestSimpleDag:
    def _ok_step(self, name: str, output: Any = None, depends_on: list | None = None) -> DagStep:
        return DagStep(
            name=name,
            fn=lambda ctx, **kw: output,
            retry_policy=RetryPolicy(max_attempts=1),
            depends_on=depends_on or [],
        )

    def _fail_step(self, name: str, depends_on: list | None = None) -> DagStep:
        return DagStep(
            name=name,
            fn=lambda ctx, **kw: (_ for _ in ()).throw(RuntimeError("bang")),
            retry_policy=RetryPolicy(max_attempts=1),
            depends_on=depends_on or [],
        )

    def test_all_steps_succeed(self) -> None:
        dag = SimpleDag("test_dag")
        dag.add_step(self._ok_step("a"))
        dag.add_step(self._ok_step("b", depends_on=["a"]))
        result = dag.run()
        assert result.succeeded
        assert result.failed_steps == []

    def test_step_skipped_when_dependency_fails(self) -> None:
        dag = SimpleDag("test_dag")
        dag.add_step(self._fail_step("a"))
        dag.add_step(self._ok_step("b", depends_on=["a"]))
        result = dag.run()
        assert not result.succeeded
        assert "a" in result.failed_steps
        assert "b" in result.skipped_steps

    def test_independent_step_runs_despite_sibling_failure(self) -> None:
        dag = SimpleDag("test_dag")
        dag.add_step(self._fail_step("a"))
        dag.add_step(self._ok_step("b"))  # no dependency on a
        result = dag.run()
        # a failed, b succeeded, overall failed because a failed
        assert not result.succeeded
        assert "a" in result.failed_steps
        assert "b" not in result.failed_steps
        assert "b" not in result.skipped_steps

    def test_run_id_in_result(self) -> None:
        dag = SimpleDag("d")
        dag.add_step(self._ok_step("s"))
        result = dag.run(run_id="fixed-id")
        assert result.run_id == "fixed-id"

    def test_partition_passed_to_context(self) -> None:
        captured: dict = {}

        def capture(ctx, **kw):
            captured["partition"] = ctx.partition

        dag = SimpleDag("d")
        dag.add_step(DagStep("s", fn=capture, retry_policy=RetryPolicy(max_attempts=1)))
        dag.run(partition="2024-01")
        assert captured["partition"] == "2024-01"

    def test_materializations_in_result(self) -> None:
        def mat_step(ctx, **kw):
            ctx.record_materialization(
                AssetMaterialization("feat", "feat.parquet", row_count=100)
            )

        dag = SimpleDag("d")
        dag.add_step(DagStep("s", fn=mat_step, retry_policy=RetryPolicy(max_attempts=1)))
        result = dag.run()
        assert len(result.materializations) == 1
        assert result.materializations[0].asset_key == "feat"

    def test_step_names(self) -> None:
        dag = SimpleDag("d")
        dag.add_step(self._ok_step("first"))
        dag.add_step(self._ok_step("second"))
        assert dag.step_names == ["first", "second"]

    def test_chaining_add_step(self) -> None:
        dag = SimpleDag("d")
        result = dag.add_step(self._ok_step("a")).add_step(self._ok_step("b"))
        assert result is dag
        assert len(dag.step_names) == 2


# ── BackfillPlanner ───────────────────────────────────────────────────────────

class TestBackfillPlanner:
    def test_all_missing_partitions_run(self) -> None:
        plan = plan_backfill(["2024-01", "2024-02", "2024-03"])
        assert set(plan.partitions_to_run) == {"2024-01", "2024-02", "2024-03"}
        assert plan.partitions_to_skip == []

    def test_completed_partitions_skipped(self) -> None:
        plan = plan_backfill(
            ["2024-01", "2024-02", "2024-03"],
            completed={"2024-01", "2024-02"},
        )
        assert plan.partitions_to_run == ["2024-03"]
        assert set(plan.partitions_to_skip) == {"2024-01", "2024-02"}

    def test_failed_partitions_rerun_by_default(self) -> None:
        plan = plan_backfill(
            ["2024-01", "2024-02"],
            failed={"2024-01"},
        )
        assert "2024-01" in plan.partitions_to_run

    def test_failed_partitions_skipped_when_disabled(self) -> None:
        plan = plan_backfill(
            ["2024-01", "2024-02"],
            failed={"2024-01"},
            rerun_failed=False,
        )
        assert "2024-01" in plan.partitions_to_skip
        assert "2024-01" not in plan.partitions_to_run

    def test_status_map_all_states(self) -> None:
        plan = plan_backfill(
            ["a", "b", "c"],
            completed={"a"},
            failed={"b"},
        )
        assert plan.status_map["a"] == PartitionStatus.COMPLETE
        assert plan.status_map["b"] == PartitionStatus.FAILED
        assert plan.status_map["c"] == PartitionStatus.MISSING

    def test_empty_partitions(self) -> None:
        plan = plan_backfill([])
        assert plan.partitions_to_run == []
        assert plan.partitions_to_skip == []


# ── asset_checksum ────────────────────────────────────────────────────────────

class TestAssetChecksum:
    def test_same_content_same_checksum(self) -> None:
        assert asset_checksum(b"hello") == asset_checksum(b"hello")

    def test_different_content_different_checksum(self) -> None:
        assert asset_checksum(b"hello") != asset_checksum(b"world")

    def test_checksum_is_64_hex_chars(self) -> None:
        h = asset_checksum(b"test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ── type annotation used inside class method ──────────────────────────────────
from typing import Any  # noqa: E402
