"""Tests for ci/milestone1_gate.py — TraceabilityRecord, GateCheck, GateReport, Milestone1Gate."""
from __future__ import annotations

import pytest

from ci.milestone1_gate import (
    GateCheck,
    GateReport,
    Milestone1Gate,
    TraceabilityRecord,
)

_SHA = "a" * 64  # valid SHA-256 hex

# ── Helpers ────────────────────────────────────────────────────────────────────

def _full_trace(**overrides) -> TraceabilityRecord:
    defaults = dict(
        prediction_id="pred-001",
        model_version="credit-risk-v1.2",
        mlflow_run_id="abc123",
        code_sha="6c6a398",
        data_version="v1",
        params={"C": 0.1},
        metrics={"auc": 0.847},
        features={"age": 35, "income": 60_000},
        score=0.731,
        decision="decline",
        correlation_id="req-001",
        artifact_sha256=_SHA,
    )
    return TraceabilityRecord(**{**defaults, **overrides})


def _gate() -> Milestone1Gate:
    return Milestone1Gate(
        trace=_full_trace(),
        slo_metrics={
            "p99_latency_ms": 220.0,
            "error_rate": 0.003,
            "model_auc": 0.847,
            "max_feature_age_hours": 24.0,
            "approval_rate": 0.65,
            "default_rate": 0.12,
        },
        artifact_sha256=_SHA,
        sbom_entries=5,
    )


# ── TraceabilityRecord ─────────────────────────────────────────────────────────

class TestTraceabilityRecord:
    def test_fully_traceable(self) -> None:
        assert _full_trace().is_fully_traceable()

    def test_not_traceable_missing_run_id(self) -> None:
        assert not _full_trace(mlflow_run_id="").is_fully_traceable()

    def test_not_traceable_missing_features(self) -> None:
        assert not _full_trace(features={}).is_fully_traceable()

    def test_empty_prediction_id_raises(self) -> None:
        with pytest.raises(ValueError, match="prediction_id"):
            _full_trace(prediction_id="")

    def test_empty_model_version_raises(self) -> None:
        with pytest.raises(ValueError, match="model_version"):
            _full_trace(model_version="")

    def test_invalid_score_raises(self) -> None:
        with pytest.raises(ValueError, match="score"):
            _full_trace(score=1.5)

    def test_invalid_decision_raises(self) -> None:
        with pytest.raises(ValueError, match="decision"):
            _full_trace(decision="maybe")

    def test_valid_decisions(self) -> None:
        for d in ("approve", "review", "decline"):
            r = _full_trace(decision=d)
            assert r.decision == d


# ── GateCheck ──────────────────────────────────────────────────────────────────

class TestGateCheck:
    def test_basic(self) -> None:
        c = GateCheck("test", gate_number=1, passed=True, message="ok")
        assert c.passed
        assert c.gate_number == 1

    def test_failure(self) -> None:
        c = GateCheck("test", gate_number=2, passed=False)
        assert not c.passed


# ── GateReport ─────────────────────────────────────────────────────────────────

class TestGateReport:
    def test_failures(self) -> None:
        checks = [
            GateCheck("a", 1, True),
            GateCheck("b", 2, False),
        ]
        r = GateReport(checks=checks, overall_passed=False)
        assert len(r.failures()) == 1

    def test_by_gate(self) -> None:
        checks = [
            GateCheck("a", 1, True),
            GateCheck("b", 1, True),
            GateCheck("c", 2, False),
        ]
        r = GateReport(checks=checks, overall_passed=False)
        assert len(r.by_gate(1)) == 2
        assert len(r.by_gate(2)) == 1

    def test_summary_passed(self) -> None:
        r = GateReport(checks=[], overall_passed=True)
        assert "PASSED" in r.summary()

    def test_summary_failed(self) -> None:
        r = GateReport(checks=[GateCheck("x", 1, False)], overall_passed=False)
        assert "FAILED" in r.summary()

    def test_checked_at_auto_set(self) -> None:
        r = GateReport()
        assert "T" in r.checked_at  # ISO format with T separator


# ── Milestone1Gate ─────────────────────────────────────────────────────────────

class TestMilestone1Gate:
    def test_all_pass(self) -> None:
        report = _gate().run()
        assert report.overall_passed
        assert len(report.failures()) == 0

    def test_checks_count(self) -> None:
        report = _gate().run()
        assert len(report.checks) == 11  # 3+3+2+1+2

    def test_gate1_checks_present(self) -> None:
        report = _gate().run()
        g1 = report.by_gate(1)
        assert len(g1) == 3
        names = [c.name for c in g1]
        assert "full-traceability" in names

    def test_gate5_sbom_check(self) -> None:
        gate = Milestone1Gate(
            trace=_full_trace(),
            artifact_sha256=_SHA,
            sbom_entries=0,  # no SBOM
        )
        report = gate.run()
        sbom_check = next(c for c in report.checks if c.name == "sbom-exists")
        assert not sbom_check.passed

    def test_missing_traceability_fails_gate1(self) -> None:
        gate = Milestone1Gate(
            trace=_full_trace(mlflow_run_id=""),
            artifact_sha256=_SHA,
            sbom_entries=5,
        )
        report = gate.run()
        assert not report.overall_passed
        g1_failures = [c for c in report.by_gate(1) if not c.passed]
        assert len(g1_failures) >= 1

    def test_high_latency_fails(self) -> None:
        gate = Milestone1Gate(
            trace=_full_trace(),
            slo_metrics={"p99_latency_ms": 800.0, "error_rate": 0.0, "model_auc": 0.85},
            artifact_sha256=_SHA,
            sbom_entries=3,
        )
        report = gate.run()
        latency_check = next(c for c in report.checks if c.name == "slo-latency-p99")
        assert not latency_check.passed

    def test_low_auc_fails(self) -> None:
        gate = Milestone1Gate(
            trace=_full_trace(),
            slo_metrics={"p99_latency_ms": 100.0, "error_rate": 0.0, "model_auc": 0.65},
            artifact_sha256=_SHA,
            sbom_entries=3,
        )
        report = gate.run()
        auc_check = next(c for c in report.checks if c.name == "slo-model-auc")
        assert not auc_check.passed

    def test_no_artifact_sha_fails_signing_check(self) -> None:
        gate = Milestone1Gate(
            trace=_full_trace(artifact_sha256=""),
            artifact_sha256="",
            sbom_entries=5,
        )
        report = gate.run()
        sign_check = next(c for c in report.checks if c.name == "artifact-signed")
        assert not sign_check.passed

    def test_no_correlation_id_fails(self) -> None:
        gate = Milestone1Gate(
            trace=_full_trace(correlation_id=""),
            artifact_sha256=_SHA,
            sbom_entries=5,
        )
        report = gate.run()
        corr_check = next(c for c in report.checks if c.name == "correlation-id-present")
        assert not corr_check.passed

    def test_missing_metrics_fails(self) -> None:
        gate = Milestone1Gate(
            trace=_full_trace(metrics={}),  # no AUC
            artifact_sha256=_SHA,
            sbom_entries=5,
        )
        report = gate.run()
        metrics_check = next(c for c in report.checks if c.name == "training-metrics-present")
        assert not metrics_check.passed

    def test_report_type(self) -> None:
        report = _gate().run()
        assert isinstance(report, GateReport)
