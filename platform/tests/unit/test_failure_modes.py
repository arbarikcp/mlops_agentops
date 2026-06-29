"""Tests for pipelines/failure_modes.py — idempotency proof, retry-safety, lineage audit."""
from __future__ import annotations

import pytest

from pipelines.dag import (
    AssetMaterialization,
    DagStep,
    DagRunResult,
    RetryPolicy,
    RunContext,
    SimpleDag,
    StepResult,
    StepStatus,
)
from pipelines.failure_modes import (
    FailureClass,
    FailureClassifier,
    IdempotencyProof,
    IdempotencyResult,
    LineageAuditReport,
    LineageAuditor,
    RetryCheckReport,
    RetryChecker,
    _stable_checksum,
)


# ── FailureClassifier ──────────────────────────────────────────────────────────

class TestFailureClassifier:
    def test_value_error_is_deterministic(self) -> None:
        fc = FailureClassifier()
        assert fc.classify(ValueError("bad value")) == FailureClass.DETERMINISTIC

    def test_connection_error_is_transient(self) -> None:
        fc = FailureClassifier()
        assert fc.classify(ConnectionError("timeout")) == FailureClass.TRANSIENT

    def test_os_error_is_transient(self) -> None:
        fc = FailureClassifier()
        assert fc.classify(OSError("disk error")) == FailureClass.TRANSIENT

    def test_type_error_is_deterministic(self) -> None:
        fc = FailureClassifier()
        assert fc.classify(TypeError("type mismatch")) == FailureClass.DETERMINISTIC

    def test_unknown_error_uses_default(self) -> None:
        fc = FailureClassifier(default_class=FailureClass.CORRUPTION)
        class WeirdError(Exception): pass
        assert fc.classify(WeirdError()) == FailureClass.CORRUPTION

    def test_is_retryable_transient(self) -> None:
        fc = FailureClassifier()
        assert fc.is_retryable(ConnectionError()) is True

    def test_is_retryable_deterministic(self) -> None:
        fc = FailureClassifier()
        assert fc.is_retryable(ValueError()) is False

    def test_custom_transient_errors(self) -> None:
        class MyTransient(Exception): pass
        fc = FailureClassifier(transient_errors=(MyTransient,))
        assert fc.classify(MyTransient()) == FailureClass.TRANSIENT

    def test_deterministic_takes_priority_over_transient(self) -> None:
        # If an error matches both lists, DETERMINISTIC wins
        class Overlap(ValueError, ConnectionError): pass
        fc = FailureClassifier()
        assert fc.classify(Overlap()) == FailureClass.DETERMINISTIC


# ── _stable_checksum ───────────────────────────────────────────────────────────

class TestStableChecksum:
    def test_same_input_same_checksum(self) -> None:
        assert _stable_checksum({"a": 1}) == _stable_checksum({"a": 1})

    def test_different_inputs_different_checksum(self) -> None:
        assert _stable_checksum({"a": 1}) != _stable_checksum({"a": 2})

    def test_works_on_non_json(self) -> None:
        h = _stable_checksum(object())
        assert len(h) == 16


# ── IdempotencyProof ───────────────────────────────────────────────────────────

class TestIdempotencyProof:
    def test_idempotent_fn_passes(self) -> None:
        proof = IdempotencyProof(fn=lambda x: x * 2, run_count=3)
        result = proof.prove(inputs=5)
        assert result.is_idempotent is True
        assert result.unique_checksums == 1
        assert result.run_count == 3

    def test_non_idempotent_fn_fails(self) -> None:
        counter = {"n": 0}

        def non_idempotent(x):
            counter["n"] += 1
            return counter["n"]   # always different

        proof = IdempotencyProof(fn=non_idempotent, run_count=2)
        result = proof.prove(inputs=42)
        assert result.is_idempotent is False
        assert result.failure_reason is not None
        assert result.unique_checksums == 2

    def test_exception_in_fn_captured(self) -> None:
        calls = {"n": 0}

        def flaky(x):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise RuntimeError("boom")
            return x

        proof = IdempotencyProof(fn=flaky, run_count=3)
        result = proof.prove(inputs=1)
        assert result.is_idempotent is False
        assert "exception" in result.failure_reason.lower()

    def test_invalid_run_count_raises(self) -> None:
        with pytest.raises(ValueError, match="run_count"):
            IdempotencyProof(fn=lambda x: x, run_count=1)

    def test_checksums_list_length(self) -> None:
        proof = IdempotencyProof(fn=lambda x: x, run_count=4)
        result = proof.prove(inputs="hello")
        assert len(result.checksums) == 4

    def test_custom_checksum_fn(self) -> None:
        custom = lambda v: "always_same"  # noqa: E731
        proof = IdempotencyProof(fn=lambda x: x, checksum_fn=custom, run_count=2)
        result = proof.prove(inputs={"anything": True})
        assert result.is_idempotent is True

    def test_no_input_callable(self) -> None:
        counter = {"n": 0}

        def fn():
            return 42

        proof = IdempotencyProof(fn=fn, run_count=2)
        result = proof.prove()
        assert result.is_idempotent is True


# ── RetryChecker ───────────────────────────────────────────────────────────────

class TestRetryChecker:
    def _make_step(self, name="s", max_attempts=3, cleanup=None, delay=1.0) -> DagStep:
        return DagStep(
            name=name,
            fn=lambda ctx, **kw: None,
            retry_policy=RetryPolicy(max_attempts=max_attempts, delay_seconds=delay),
            cleanup_fn=cleanup,
        )

    def test_no_retry_no_cleanup_has_issues(self) -> None:
        step = self._make_step(max_attempts=1, cleanup=None)
        checker = RetryChecker()
        report = checker.check(step)
        assert report.is_retry_safe is False
        assert any("max_attempts=1" in i for i in report.issues)

    def test_retry_without_cleanup_has_issue(self) -> None:
        step = self._make_step(max_attempts=3, cleanup=None)
        checker = RetryChecker()
        report = checker.check(step)
        assert report.is_retry_safe is False
        assert any("cleanup" in i for i in report.issues)

    def test_retry_with_cleanup_is_safe(self) -> None:
        step = self._make_step(max_attempts=3, cleanup=lambda ctx, exc: None)
        checker = RetryChecker()
        report = checker.check(step)
        assert report.is_retry_safe is True
        assert report.has_cleanup is True
        assert report.issues == []

    def test_suggestions_provided_on_issue(self) -> None:
        step = self._make_step(max_attempts=1)
        report = RetryChecker().check(step)
        assert len(report.suggestions) > 0

    def test_check_all_runs_on_all_steps(self) -> None:
        dag = SimpleDag("d")
        dag.add_step(self._make_step("a", max_attempts=3, cleanup=lambda c, e: None))
        dag.add_step(self._make_step("b", max_attempts=1))
        reports = RetryChecker().check_all(dag)
        assert len(reports) == 2
        names = {r.step_name for r in reports}
        assert names == {"a", "b"}

    def test_step_name_in_report(self) -> None:
        step = self._make_step("my_step")
        report = RetryChecker().check(step)
        assert report.step_name == "my_step"


# ── LineageAuditor ────────────────────────────────────────────────────────────

def _make_run_result(asset_keys: list[str], add_checksums: bool = True) -> DagRunResult:
    """Helper to create a fake DagRunResult with given materializations."""
    mats = [
        AssetMaterialization(
            asset_key=k,
            path=f"{k}.parquet",
            checksum="abc123" if add_checksums else None,
        )
        for k in asset_keys
    ]
    return DagRunResult(
        run_id="test-run",
        succeeded=True,
        step_results=[],
        materializations=mats,
        duration_s=1.0,
    )


class TestLineageAuditor:
    def test_complete_lineage_passes(self) -> None:
        result = _make_run_result([
            "raw_credit_data", "validated_data", "feature_dataset",
            "trained_model", "validation_report", "champion_model",
        ])
        auditor = LineageAuditor()
        report = auditor.audit(result)
        assert report.complete is True
        assert report.missing_assets == []

    def test_missing_asset_detected(self) -> None:
        result = _make_run_result([
            "raw_credit_data", "validated_data", "feature_dataset",
            "trained_model", "validation_report",
            # champion_model MISSING
        ])
        auditor = LineageAuditor()
        report = auditor.audit(result)
        assert report.complete is False
        assert "champion_model" in report.missing_assets

    def test_missing_checksum_detected(self) -> None:
        result = _make_run_result([
            "raw_credit_data", "validated_data", "feature_dataset",
            "trained_model", "validation_report", "champion_model",
        ], add_checksums=False)
        auditor = LineageAuditor()
        report = auditor.audit(result)
        assert report.complete is False
        assert len(report.missing_checksums) == 6

    def test_chain_contains_all_assets(self) -> None:
        keys = ["raw_credit_data", "validated_data", "feature_dataset",
                "trained_model", "validation_report", "champion_model"]
        result = _make_run_result(keys)
        report = LineageAuditor().audit(result)
        chain_keys = [c["asset_key"] for c in report.chain]
        for k in keys:
            assert k in chain_keys

    def test_run_id_in_report(self) -> None:
        result = _make_run_result(["raw_credit_data"])
        result.run_id = "my-run-42"
        report = LineageAuditor(required_assets=["raw_credit_data"]).audit(result)
        assert report.run_id == "my-run-42"

    def test_require_chaining(self) -> None:
        auditor = LineageAuditor(required_assets=[])
        returned = auditor.require("new_asset")
        assert returned is auditor
        assert "new_asset" in auditor.required_assets

    def test_custom_required_assets(self) -> None:
        result = _make_run_result(["raw_credit_data", "trained_model"])
        auditor = LineageAuditor(required_assets=["raw_credit_data", "trained_model"])
        report = auditor.audit(result)
        assert report.complete is True

    def test_issues_message_contains_asset_name(self) -> None:
        result = _make_run_result(["raw_credit_data"])
        auditor = LineageAuditor(required_assets=["raw_credit_data", "trained_model"])
        report = auditor.audit(result)
        assert any("trained_model" in issue for issue in report.issues)
