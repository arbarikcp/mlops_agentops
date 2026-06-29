"""Prometheus-compatible ML metrics collector.

Day 49 — collects ML-specific metrics (latency, drift, AUC, freshness, business KPIs)
and formats them in Prometheus text exposition format so the FastAPI /metrics endpoint
can serve them without requiring the prometheus_client library.

Classes:
  MetricSnapshot       — one metric observation (name, value, labels, type)
  MLMetricsCollector   — records observations; formats text exposition

See: docs/phase7/day49_prometheus.md
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


# ── MetricSnapshot ─────────────────────────────────────────────────────────────

@dataclass
class MetricSnapshot:
    """One Prometheus metric observation.

    Attributes:
        name:        Metric name (snake_case, no prefix).
        value:       Current numeric value.
        labels:      Prometheus label dict (e.g., {"feature": "pay_ratio"}).
        help_text:   HELP string shown in text exposition.
        metric_type: "counter" / "gauge" / "histogram" / "summary".
        timestamp:   Unix epoch when recorded (informational).
    """

    name: str
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    help_text: str = ""
    metric_type: str = "gauge"
    timestamp: float = field(default_factory=time.time)


# ── MLMetricsCollector ────────────────────────────────────────────────────────

class MLMetricsCollector:
    """Thread-safe ML metrics collector with Prometheus text exposition output.

    All metrics are stored in-memory as `MetricSnapshot` objects.
    Call `collect()` to retrieve snapshots; call `format_text_exposition()` to
    get the string that a `/metrics` HTTP endpoint should return.

    Args:
        prefix: Prefix prepended to all metric names (default: "mlops").
    """

    def __init__(self, prefix: str = "mlops") -> None:
        if not prefix:
            raise ValueError("prefix cannot be empty")
        self.prefix = prefix
        self._lock = Lock()
        # Counters (cumulative)
        self._prediction_count: int = 0
        self._error_count: int = 0
        self._latency_sum_ms: float = 0.0
        self._latency_count: int = 0
        # Latency histogram buckets (ms): 5, 10, 25, 50, 100, 250, 500, 1000, +Inf
        self._latency_buckets = {5: 0, 10: 0, 25: 0, 50: 0, 100: 0, 250: 0, 500: 0, 1000: 0}
        # Gauges (latest values keyed by label fingerprint)
        self._drift_scores: dict[str, dict[str, Any]] = {}  # {feature: {psi, severity}}
        self._auc: float | None = None
        self._freshness: dict[str, float] = {}  # {view_name: age_hours}
        self._approval_rate: float | None = None
        self._default_rate: float | None = None

    # ── Record methods ─────────────────────────────────────────────────────────

    def record_prediction(self, latency_ms: float, error: bool = False) -> None:
        """Record one prediction request with latency and error flag."""
        with self._lock:
            self._prediction_count += 1
            if error:
                self._error_count += 1
            self._latency_sum_ms += latency_ms
            self._latency_count += 1
            for bucket in self._latency_buckets:
                if latency_ms <= bucket:
                    self._latency_buckets[bucket] += 1

    def record_drift(self, feature_name: str, psi: float, severity: str = "none") -> None:
        """Record current PSI drift score for a feature."""
        with self._lock:
            self._drift_scores[feature_name] = {"psi": psi, "severity": severity.lower()}

    def record_auc(self, auc: float) -> None:
        """Record current model AUC."""
        with self._lock:
            self._auc = auc

    def record_feature_freshness(self, view_name: str, age_hours: float) -> None:
        """Record how many hours since a feature view was last materialised."""
        with self._lock:
            self._freshness[view_name] = age_hours

    def record_approval_rate(self, rate: float) -> None:
        """Record current approval rate (0–1)."""
        with self._lock:
            self._approval_rate = rate

    def record_default_rate(self, rate: float) -> None:
        """Record current observed default rate (0–1)."""
        with self._lock:
            self._default_rate = rate

    # ── Collect ───────────────────────────────────────────────────────────────

    def collect(self) -> list[MetricSnapshot]:
        """Return all current metric snapshots."""
        with self._lock:
            snapshots: list[MetricSnapshot] = []

            # Operational counters
            snapshots.append(MetricSnapshot(
                name=f"{self.prefix}_prediction_requests_total",
                value=float(self._prediction_count),
                help_text="Total prediction requests served",
                metric_type="counter",
            ))
            snapshots.append(MetricSnapshot(
                name=f"{self.prefix}_prediction_errors_total",
                value=float(self._error_count),
                help_text="Total prediction errors",
                metric_type="counter",
            ))

            # Latency gauge (average)
            avg_latency = (
                self._latency_sum_ms / self._latency_count
                if self._latency_count > 0 else 0.0
            )
            snapshots.append(MetricSnapshot(
                name=f"{self.prefix}_prediction_latency_ms_avg",
                value=avg_latency,
                help_text="Average prediction latency milliseconds",
                metric_type="gauge",
            ))

            # Latency histogram buckets
            cumulative = 0
            for bucket_ms, count in sorted(self._latency_buckets.items()):
                cumulative += count
                snapshots.append(MetricSnapshot(
                    name=f"{self.prefix}_prediction_latency_ms_bucket",
                    value=float(cumulative),
                    labels={"le": str(bucket_ms)},
                    help_text="Prediction latency histogram",
                    metric_type="histogram",
                ))
            snapshots.append(MetricSnapshot(
                name=f"{self.prefix}_prediction_latency_ms_bucket",
                value=float(self._latency_count),
                labels={"le": "+Inf"},
                help_text="Prediction latency histogram",
                metric_type="histogram",
            ))
            snapshots.append(MetricSnapshot(
                name=f"{self.prefix}_prediction_latency_ms_sum",
                value=self._latency_sum_ms,
                help_text="Sum of prediction latencies",
                metric_type="gauge",
            ))
            snapshots.append(MetricSnapshot(
                name=f"{self.prefix}_prediction_latency_ms_count",
                value=float(self._latency_count),
                help_text="Count of latency observations",
                metric_type="gauge",
            ))

            # Drift gauges
            for feature, info in self._drift_scores.items():
                snapshots.append(MetricSnapshot(
                    name=f"{self.prefix}_drift_score",
                    value=info["psi"],
                    labels={"feature": feature, "severity": info["severity"]},
                    help_text="Current PSI drift score per feature",
                    metric_type="gauge",
                ))

            # AUC gauge
            if self._auc is not None:
                snapshots.append(MetricSnapshot(
                    name=f"{self.prefix}_model_auc",
                    value=self._auc,
                    help_text="Current model AUC on labeled feedback",
                    metric_type="gauge",
                ))

            # Feature freshness
            for view, age in self._freshness.items():
                snapshots.append(MetricSnapshot(
                    name=f"{self.prefix}_feature_freshness_hours",
                    value=age,
                    labels={"view": view},
                    help_text="Hours since feature view was last materialised",
                    metric_type="gauge",
                ))

            # Business KPIs
            if self._approval_rate is not None:
                snapshots.append(MetricSnapshot(
                    name=f"{self.prefix}_approval_rate",
                    value=self._approval_rate,
                    help_text="Current approval rate",
                    metric_type="gauge",
                ))
            if self._default_rate is not None:
                snapshots.append(MetricSnapshot(
                    name=f"{self.prefix}_default_rate",
                    value=self._default_rate,
                    help_text="Current observed default rate",
                    metric_type="gauge",
                ))

            return snapshots

    # ── Text exposition ───────────────────────────────────────────────────────

    def format_text_exposition(self) -> str:
        """Format collected metrics as Prometheus text exposition (content-type text/plain)."""
        snapshots = self.collect()
        lines: list[str] = []
        seen_names: set[str] = set()

        for snap in snapshots:
            if snap.name not in seen_names:
                lines.append(f"# HELP {snap.name} {snap.help_text}")
                lines.append(f"# TYPE {snap.name} {snap.metric_type}")
                seen_names.add(snap.name)

            if snap.labels:
                label_str = ",".join(f'{k}="{v}"' for k, v in sorted(snap.labels.items()))
                lines.append(f"{snap.name}{{{label_str}}} {snap.value}")
            else:
                lines.append(f"{snap.name} {snap.value}")

        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        """Reset all metrics (useful in tests)."""
        with self._lock:
            self._prediction_count = 0
            self._error_count = 0
            self._latency_sum_ms = 0.0
            self._latency_count = 0
            self._latency_buckets = {k: 0 for k in self._latency_buckets}
            self._drift_scores = {}
            self._auc = None
            self._freshness = {}
            self._approval_rate = None
            self._default_rate = None
