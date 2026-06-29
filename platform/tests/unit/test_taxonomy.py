"""Tests for monitoring/taxonomy.py — MonitorType, MonitorResult, Monitor, MonitorRegistry."""
from __future__ import annotations

import pytest

from monitoring.taxonomy import (
    Monitor,
    MonitorRegistry,
    MonitorResult,
    MonitorType,
    Severity,
    alert_channel,
)


def _passing_result(name: str = "test", monitor_type: MonitorType = MonitorType.ML) -> MonitorResult:
    return MonitorResult(
        name=name,
        monitor_type=monitor_type,
        passed=True,
        value=0.8,
        threshold=0.7,
        severity=Severity.WARNING,
        message="ok",
    )


def _failing_result(
    name: str = "test",
    monitor_type: MonitorType = MonitorType.ML,
    severity: Severity = Severity.WARNING,
) -> MonitorResult:
    return MonitorResult(
        name=name,
        monitor_type=monitor_type,
        passed=False,
        value=0.5,
        threshold=0.7,
        severity=severity,
        message="failed",
    )


# ── MonitorType ────────────────────────────────────────────────────────────────

class TestMonitorType:
    def test_values(self) -> None:
        assert MonitorType.OPERATIONAL == "operational"
        assert MonitorType.ML == "ml"
        assert MonitorType.BUSINESS == "business"


# ── alert_channel ──────────────────────────────────────────────────────────────

class TestAlertChannel:
    def test_operational_channel(self) -> None:
        assert alert_channel(MonitorType.OPERATIONAL) == "#oncall-infra"

    def test_ml_channel(self) -> None:
        assert alert_channel(MonitorType.ML) == "#ml-alerts"

    def test_business_channel(self) -> None:
        assert alert_channel(MonitorType.BUSINESS) == "#business-risk"


# ── MonitorResult ──────────────────────────────────────────────────────────────

class TestMonitorResult:
    def test_channel_auto_set(self) -> None:
        r = _passing_result(monitor_type=MonitorType.BUSINESS)
        assert r.channel == "#business-risk"

    def test_explicit_channel_preserved(self) -> None:
        r = MonitorResult("x", MonitorType.ML, True, 0.8, 0.7, Severity.INFO, channel="#custom")
        assert r.channel == "#custom"

    def test_should_alert_false_when_passed(self) -> None:
        r = _passing_result()
        assert not r.should_alert

    def test_should_alert_true_when_failed_warning(self) -> None:
        r = _failing_result(severity=Severity.WARNING)
        assert r.should_alert

    def test_should_alert_false_when_info(self) -> None:
        r = _failing_result(severity=Severity.INFO)
        assert not r.should_alert

    def test_blocks_gate_only_for_critical(self) -> None:
        warning = _failing_result(severity=Severity.WARNING)
        critical = _failing_result(severity=Severity.CRITICAL)
        assert not warning.blocks_gate
        assert critical.blocks_gate

    def test_blocks_gate_false_when_passed(self) -> None:
        r = MonitorResult("x", MonitorType.ML, True, 0.8, 0.7, Severity.CRITICAL)
        assert not r.blocks_gate


# ── Monitor ────────────────────────────────────────────────────────────────────

class TestMonitor:
    def test_run_returns_result(self) -> None:
        m = Monitor("m1", MonitorType.ML, lambda: _passing_result("m1"))
        result = m.run()
        assert result.passed

    def test_run_exception_returns_critical_failure(self) -> None:
        def bad_check() -> MonitorResult:
            raise RuntimeError("broken")

        m = Monitor("m_bad", MonitorType.OPERATIONAL, bad_check)
        result = m.run()
        assert not result.passed
        assert result.severity == Severity.CRITICAL
        assert "broken" in result.message

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Monitor("", MonitorType.ML, lambda: _passing_result())


# ── MonitorRegistry ────────────────────────────────────────────────────────────

class TestMonitorRegistry:
    def _make_registry(self) -> tuple[MonitorRegistry, list[bool]]:
        registry = MonitorRegistry()
        calls: list[bool] = []

        def make_check(name: str, passed: bool, mtype: MonitorType):
            def check() -> MonitorResult:
                calls.append(passed)
                return MonitorResult(name, mtype, passed, 0.5, 0.7, Severity.WARNING)
            return check

        registry.register(Monitor("op1", MonitorType.OPERATIONAL, make_check("op1", True, MonitorType.OPERATIONAL)))
        registry.register(Monitor("ml1", MonitorType.ML, make_check("ml1", False, MonitorType.ML)))
        registry.register(Monitor("biz1", MonitorType.BUSINESS, make_check("biz1", True, MonitorType.BUSINESS)))
        return registry, calls

    def test_register_and_len(self) -> None:
        registry = MonitorRegistry()
        registry.register(Monitor("a", MonitorType.ML, lambda: _passing_result("a")))
        assert len(registry) == 1

    def test_register_overwrites_same_name(self) -> None:
        registry = MonitorRegistry()
        registry.register(Monitor("a", MonitorType.ML, lambda: _passing_result("a")))
        registry.register(Monitor("a", MonitorType.ML, lambda: _passing_result("a")))
        assert len(registry) == 1

    def test_run_all_returns_all_results(self) -> None:
        registry, _ = self._make_registry()
        results = registry.run_all()
        assert len(results) == 3

    def test_failed_results_filters_correctly(self) -> None:
        registry, _ = self._make_registry()
        results = registry.run_all()
        failed = MonitorRegistry.failed_results(results)
        assert len(failed) == 1
        assert failed[0].name == "ml1"

    def test_run_by_type(self) -> None:
        registry, _ = self._make_registry()
        results = registry.run_by_type(MonitorType.ML)
        assert len(results) == 1
        assert results[0].monitor_type == MonitorType.ML

    def test_gate_blocking_only_critical(self) -> None:
        registry = MonitorRegistry()
        registry.register(Monitor("w", MonitorType.ML,
            lambda: MonitorResult("w", MonitorType.ML, False, 0.5, 0.7, Severity.WARNING)))
        registry.register(Monitor("c", MonitorType.OPERATIONAL,
            lambda: MonitorResult("c", MonitorType.OPERATIONAL, False, 0.5, 0.7, Severity.CRITICAL)))
        results = registry.run_all()
        blocking = MonitorRegistry.gate_blocking_results(results)
        assert len(blocking) == 1
        assert blocking[0].name == "c"

    def test_unregister(self) -> None:
        registry = MonitorRegistry()
        registry.register(Monitor("a", MonitorType.ML, lambda: _passing_result("a")))
        registry.unregister("a")
        assert len(registry) == 0

    def test_names(self) -> None:
        registry, _ = self._make_registry()
        assert set(registry.names()) == {"op1", "ml1", "biz1"}
