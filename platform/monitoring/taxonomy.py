"""Monitoring taxonomy: OPERATIONAL / ML / BUSINESS monitors with typed alert routing.

Day 46 — establishes the three-pillar taxonomy enforced throughout Phase 7.
Each monitor has a type that determines its alert channel, owner, and SLO.

Classes:
  MonitorType     — OPERATIONAL / ML / BUSINESS
  Severity        — INFO / WARNING / CRITICAL
  MonitorResult   — outcome of one monitor run (value, threshold, passed, channel)
  Monitor         — callable wrapper with metadata
  MonitorRegistry — runs all registered monitors; routes failures by type

See: docs/phase7/day46_monitoring_taxonomy.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Enumerations ──────────────────────────────────────────────────────────────

class MonitorType(str, Enum):
    OPERATIONAL = "operational"
    ML = "ml"
    BUSINESS = "business"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ── Alert channel mapping — one channel per type, enforced centrally ──────────

_ALERT_CHANNELS: dict[MonitorType, str] = {
    MonitorType.OPERATIONAL: "#oncall-infra",
    MonitorType.ML:          "#ml-alerts",
    MonitorType.BUSINESS:    "#business-risk",
}


def alert_channel(monitor_type: MonitorType) -> str:
    return _ALERT_CHANNELS[monitor_type]


# ── MonitorResult ─────────────────────────────────────────────────────────────

@dataclass
class MonitorResult:
    """Outcome of one monitor check run.

    Attributes:
        name:         Monitor identifier.
        monitor_type: OPERATIONAL / ML / BUSINESS.
        passed:       True if value is within acceptable range.
        value:        Observed metric value.
        threshold:    Acceptable limit (semantics depend on monitor).
        severity:     INFO / WARNING / CRITICAL (inherited from Monitor).
        message:      Human-readable explanation.
        channel:      Alert channel derived from monitor_type.
        metadata:     Arbitrary extra context (e.g., feature names, run IDs).
    """

    name: str
    monitor_type: MonitorType
    passed: bool
    value: float
    threshold: float
    severity: Severity
    message: str = ""
    channel: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.channel:
            self.channel = alert_channel(self.monitor_type)

    @property
    def should_alert(self) -> bool:
        return not self.passed and self.severity != Severity.INFO

    @property
    def blocks_gate(self) -> bool:
        return not self.passed and self.severity == Severity.CRITICAL


# ── Monitor ───────────────────────────────────────────────────────────────────

@dataclass
class Monitor:
    """A single named monitor wrapping a check callable.

    Args:
        name:         Unique identifier.
        monitor_type: OPERATIONAL / ML / BUSINESS.
        check_fn:     Callable() -> MonitorResult. Must handle its own exceptions.
        threshold:    Threshold to pass into check_fn (informational; check_fn may ignore it).
        severity:     Severity level for failures.
        description:  Optional human description for dashboards.
    """

    name: str
    monitor_type: MonitorType
    check_fn: Callable[[], MonitorResult]
    threshold: float = 0.0
    severity: Severity = Severity.WARNING
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Monitor name cannot be empty")

    def run(self) -> MonitorResult:
        """Execute the check_fn and return its MonitorResult.

        If check_fn raises, returns a CRITICAL failure result so the registry
        always gets a result (no uncaught exceptions propagate to the scheduler).
        """
        try:
            result = self.check_fn()
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("Monitor '%s' raised: %s", self.name, exc)
            return MonitorResult(
                name=self.name,
                monitor_type=self.monitor_type,
                passed=False,
                value=float("nan"),
                threshold=self.threshold,
                severity=Severity.CRITICAL,
                message=f"Monitor raised exception: {exc}",
            )


# ── MonitorRegistry ───────────────────────────────────────────────────────────

class MonitorRegistry:
    """Registers and runs a collection of monitors; routes failures by type.

    Usage::

        registry = MonitorRegistry()
        registry.register(Monitor("latency_p99", MonitorType.OPERATIONAL, check_latency))
        registry.register(Monitor("auc_decay",   MonitorType.ML,          check_auc))
        results = registry.run_all()
        for r in registry.failed_results(results):
            send_alert(r.channel, r.message)
    """

    def __init__(self) -> None:
        self._monitors: dict[str, Monitor] = {}

    def register(self, monitor: Monitor) -> None:
        """Register a monitor. Overwrites if name already registered."""
        if not monitor.name:
            raise ValueError("Monitor name cannot be empty")
        self._monitors[monitor.name] = monitor
        logger.debug("Registered monitor '%s' (type=%s)", monitor.name, monitor.monitor_type)

    def unregister(self, name: str) -> None:
        self._monitors.pop(name, None)

    def names(self) -> list[str]:
        return list(self._monitors.keys())

    def __len__(self) -> int:
        return len(self._monitors)

    def run_all(self) -> list[MonitorResult]:
        """Run every registered monitor. Returns one result per monitor."""
        results: list[MonitorResult] = []
        for monitor in self._monitors.values():
            result = monitor.run()
            results.append(result)
            if not result.passed:
                logger.warning(
                    "Monitor '%s' FAILED (severity=%s, channel=%s): %s",
                    result.name, result.severity, result.channel, result.message,
                )
        return results

    def run_by_type(self, monitor_type: MonitorType) -> list[MonitorResult]:
        """Run only monitors of a specific type."""
        results: list[MonitorResult] = []
        for monitor in self._monitors.values():
            if monitor.monitor_type == monitor_type:
                results.append(monitor.run())
        return results

    @staticmethod
    def failed_results(results: list[MonitorResult]) -> list[MonitorResult]:
        return [r for r in results if not r.passed]

    @staticmethod
    def gate_blocking_results(results: list[MonitorResult]) -> list[MonitorResult]:
        """Results that block a promotion gate (CRITICAL + failed)."""
        return [r for r in results if r.blocks_gate]

    @staticmethod
    def alert_channel(monitor_type: MonitorType) -> str:
        return _ALERT_CHANNELS[monitor_type]
