"""Tests for monitoring/slo.py — SLODefinition, SLOResult, SLOReport, SLOChecker."""
from __future__ import annotations

import pytest

from monitoring.slo import (
    BudgetStatus,
    SLOChecker,
    SLODefinition,
    SLOReport,
    SLOResult,
    SLOType,
    _budget_status,
)


# ── _budget_status ─────────────────────────────────────────────────────────────

class TestBudgetStatus:
    def test_green_below_half(self) -> None:
        assert _budget_status(0.0) == BudgetStatus.GREEN
        assert _budget_status(0.49) == BudgetStatus.GREEN

    def test_yellow_half_to_ninety(self) -> None:
        assert _budget_status(0.50) == BudgetStatus.YELLOW
        assert _budget_status(0.89) == BudgetStatus.YELLOW

    def test_red_ninety_to_hundred(self) -> None:
        assert _budget_status(0.90) == BudgetStatus.RED
        assert _budget_status(0.99) == BudgetStatus.RED

    def test_exhausted_at_one(self) -> None:
        assert _budget_status(1.0) == BudgetStatus.EXHAUSTED
        assert _budget_status(1.5) == BudgetStatus.EXHAUSTED


# ── SLODefinition ──────────────────────────────────────────────────────────────

class TestSLODefinition:
    def test_error_budget_hours(self) -> None:
        slo = SLODefinition("latency", SLOType.LATENCY, target=0.999, window_hours=720.0)
        assert slo.error_budget_hours == pytest.approx(0.001 * 720.0)

    def test_invalid_target_raises(self) -> None:
        with pytest.raises(ValueError, match="target"):
            SLODefinition("x", SLOType.LATENCY, target=1.5)

    def test_zero_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_hours"):
            SLODefinition("x", SLOType.LATENCY, target=0.99, window_hours=0)


# ── SLOResult ──────────────────────────────────────────────────────────────────

class TestSLOResult:
    def test_blocks_gate_red(self) -> None:
        r = SLOResult("x", SLOType.LATENCY, 600, 500, False, 0.95, BudgetStatus.RED)
        assert r.blocks_gate

    def test_blocks_gate_exhausted(self) -> None:
        r = SLOResult("x", SLOType.LATENCY, 600, 500, False, 1.0, BudgetStatus.EXHAUSTED)
        assert r.blocks_gate

    def test_does_not_block_when_yellow(self) -> None:
        r = SLOResult("x", SLOType.LATENCY, 550, 500, False, 0.60, BudgetStatus.YELLOW)
        assert not r.blocks_gate

    def test_does_not_block_when_compliant(self) -> None:
        r = SLOResult("x", SLOType.LATENCY, 400, 500, True, 0.0, BudgetStatus.GREEN)
        assert not r.blocks_gate


# ── SLOReport ──────────────────────────────────────────────────────────────────

class TestSLOReport:
    def _result(self, name, compliant, status=BudgetStatus.GREEN, stype=SLOType.LATENCY):
        return SLOResult(name, stype, 0.5, 0.5, compliant, 0.0, status)

    def test_violations(self) -> None:
        report = SLOReport([
            self._result("a", True),
            self._result("b", False),
        ])
        assert len(report.violations()) == 1

    def test_gate_blocking(self) -> None:
        report = SLOReport([
            self._result("a", False, BudgetStatus.RED),
            self._result("b", False, BudgetStatus.YELLOW),
        ])
        assert len(report.gate_blocking()) == 1

    def test_budget_summary_keys(self) -> None:
        report = SLOReport([self._result("p99", True)])
        summary = report.budget_summary()
        assert "p99" in summary

    def test_summary_compliant(self) -> None:
        report = SLOReport([], overall_compliant=True)
        assert "COMPLIANT" in report.summary()

    def test_summary_violation(self) -> None:
        report = SLOReport([self._result("x", False)], overall_compliant=False)
        assert "VIOLATION" in report.summary()

    def test_by_type(self) -> None:
        report = SLOReport([
            self._result("a", True, stype=SLOType.LATENCY),
            self._result("b", True, stype=SLOType.BUSINESS),
        ])
        assert len(report.by_type(SLOType.LATENCY)) == 1


# ── SLOChecker — individual checks ────────────────────────────────────────────

class TestSLOCheckerLatency:
    def test_compliant(self) -> None:
        r = SLOChecker().check_latency(400.0)
        assert r.compliant
        assert r.budget_status == BudgetStatus.GREEN

    def test_violation(self) -> None:
        r = SLOChecker().check_latency(600.0)
        assert not r.compliant

    def test_exactly_at_threshold(self) -> None:
        r = SLOChecker(latency_p99_ms=500.0).check_latency(500.0)
        assert r.compliant


class TestSLOCheckerErrorRate:
    def test_compliant(self) -> None:
        r = SLOChecker().check_error_rate(0.005)
        assert r.compliant

    def test_violation(self) -> None:
        r = SLOChecker().check_error_rate(0.05)
        assert not r.compliant

    def test_zero_rate_green(self) -> None:
        r = SLOChecker().check_error_rate(0.0)
        assert r.compliant
        assert r.budget_status == BudgetStatus.GREEN


class TestSLOCheckerModelQuality:
    def test_compliant(self) -> None:
        r = SLOChecker().check_model_quality(0.80)
        assert r.compliant

    def test_violation(self) -> None:
        r = SLOChecker().check_model_quality(0.65)
        assert not r.compliant

    def test_exactly_at_threshold(self) -> None:
        r = SLOChecker(min_auc=0.72).check_model_quality(0.72)
        assert r.compliant


class TestSLOCheckerFreshness:
    def test_compliant(self) -> None:
        r = SLOChecker().check_feature_freshness(10.0)
        assert r.compliant

    def test_violation(self) -> None:
        r = SLOChecker().check_feature_freshness(30.0)
        assert not r.compliant


class TestSLOCheckerApprovalRate:
    def test_in_band_compliant(self) -> None:
        r = SLOChecker().check_approval_rate(0.70)
        assert r.compliant

    def test_below_min_violation(self) -> None:
        r = SLOChecker().check_approval_rate(0.50)
        assert not r.compliant

    def test_above_max_violation(self) -> None:
        r = SLOChecker().check_approval_rate(0.90)
        assert not r.compliant


class TestSLOCheckerDefaultRate:
    def test_compliant(self) -> None:
        r = SLOChecker().check_default_rate(0.22)
        assert r.compliant

    def test_violation(self) -> None:
        r = SLOChecker().check_default_rate(0.40)
        assert not r.compliant


# ── SLOChecker — run_all ──────────────────────────────────────────────────────

class TestSLOCheckerRunAll:
    def test_all_pass(self) -> None:
        metrics = {
            "p99_latency_ms": 200.0,
            "error_rate": 0.005,
            "model_auc": 0.80,
            "max_feature_age_hours": 5.0,
            "approval_rate": 0.70,
            "default_rate": 0.22,
        }
        report = SLOChecker().run_all(metrics)
        assert report.overall_compliant
        assert len(report.results) == 6

    def test_one_violation(self) -> None:
        metrics = {
            "p99_latency_ms": 800.0,  # fails
            "error_rate": 0.005,
            "model_auc": 0.80,
        }
        report = SLOChecker().run_all(metrics)
        assert not report.overall_compliant
        assert len(report.violations()) == 1

    def test_missing_keys_skipped(self) -> None:
        report = SLOChecker().run_all({"model_auc": 0.80})
        assert len(report.results) == 1

    def test_empty_metrics_all_pass(self) -> None:
        report = SLOChecker().run_all({})
        assert report.overall_compliant
        assert len(report.results) == 0
