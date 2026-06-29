"""Load test utilities: latency profiler, load test config/result, Locust scenario.

This module provides:
    LatencyProfiler   — contextmanager-based per-label latency collection
    LoadTestConfig    — parameterised load test profile
    LoadTestResult    — stores p50/p95/p99 and SLA pass/fail decision
    LoadTestRunner    — runs a synchronous load simulation against a callable
    locustfile snippet — see serving/locustfile.py (generated separately)

Does NOT require Locust to be installed. The Locust scenario is in locustfile.py.
LatencyProfiler and LoadTestRunner work standalone.

See: docs/phase4/day29_load_testing.md for theory.

Usage:
    from serving.load_test import LatencyProfiler, LoadTestRunner, LoadTestConfig

    profiler = LatencyProfiler()
    with profiler.measure("onnx_inference"):
        scores = model.predict(X)

    config = LoadTestConfig(target_rps=100, hold_seconds=30)
    runner = LoadTestRunner(config, predict_fn=lambda x: runner.predict_single(x))
    result = runner.run(features_df)
    assert result.passed_sla, f"p99={result.p99_ms}ms > {config.p99_threshold_ms}ms"
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Generator

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── LatencyProfiler ────────────────────────────────────────────────────────────

class LatencyProfiler:
    """Collects per-label latency samples and reports percentile statistics.

    Use as a context manager to time code blocks, or call record() directly.

    Example:
        profiler = LatencyProfiler()
        with profiler.measure("inference"):
            result = model.predict(X)
        print(profiler.report())
    """

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, label: str) -> Generator[None, None, None]:
        """Time a code block and record its latency under `label`."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(label, elapsed_ms)

    def record(self, label: str, latency_ms: float) -> None:
        """Record a latency sample directly."""
        if label not in self._timings:
            self._timings[label] = []
        self._timings[label].append(latency_ms)

    def percentile(self, label: str, p: float) -> float:
        """Return the p-th percentile for a label (0.0 if no samples)."""
        samples = self._timings.get(label, [])
        if not samples:
            return 0.0
        return float(np.percentile(samples, p))

    def n_samples(self, label: str) -> int:
        """Number of samples recorded for a label."""
        return len(self._timings.get(label, []))

    def labels(self) -> list[str]:
        """All labels that have been recorded."""
        return list(self._timings.keys())

    def report(self) -> dict[str, dict[str, float]]:
        """Return per-label p50/p95/p99/mean/n summary dict."""
        result: dict[str, dict[str, float]] = {}
        for label, samples in self._timings.items():
            arr = np.array(samples)
            result[label] = {
                "n": float(len(arr)),
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }
        return result

    def reset(self, label: str | None = None) -> None:
        """Clear samples for a specific label or all labels."""
        if label:
            self._timings.pop(label, None)
        else:
            self._timings.clear()


# ── LoadTestConfig ─────────────────────────────────────────────────────────────

@dataclass
class LoadTestConfig:
    """Configuration for a synchronous load simulation.

    Attributes:
        target_rps:            Target requests per second.
        ramp_seconds:          Seconds to ramp from 0 to target_rps.
        hold_seconds:          Seconds to hold at target_rps.
        p95_threshold_ms:      SLA: p95 latency must be below this.
        p99_threshold_ms:      SLA: p99 latency must be below this.
        error_rate_threshold:  SLA: error rate must be below this fraction.
        max_requests:          Hard limit on total requests (for fast unit tests).
    """

    target_rps: int = 50
    ramp_seconds: int = 10
    hold_seconds: int = 30
    p95_threshold_ms: float = 200.0
    p99_threshold_ms: float = 500.0
    error_rate_threshold: float = 0.01
    max_requests: int | None = None

    def __post_init__(self) -> None:
        if self.target_rps < 1:
            raise ValueError("target_rps must be >= 1")
        if self.p95_threshold_ms <= 0 or self.p99_threshold_ms <= 0:
            raise ValueError("Latency thresholds must be positive")
        if not 0 < self.error_rate_threshold <= 1:
            raise ValueError("error_rate_threshold must be in (0, 1]")


# ── LoadTestResult ─────────────────────────────────────────────────────────────

@dataclass
class LoadTestResult:
    """Summary of a completed load test run.

    Attributes:
        rps_achieved:   Actual requests per second achieved.
        p50_ms:         Median latency in milliseconds.
        p95_ms:         95th percentile latency.
        p99_ms:         99th percentile latency.
        error_rate:     Fraction of requests that failed (error / total).
        n_requests:     Total requests made.
        n_errors:       Number of failed requests.
        passed_sla:     True if all SLA thresholds are met.
        violations:     List of SLA violations (empty if passed_sla=True).
    """

    rps_achieved: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    n_requests: int
    n_errors: int
    passed_sla: bool
    violations: list[str]

    def summary(self) -> str:
        lines = [
            f"RPS: {self.rps_achieved:.1f}",
            f"p50: {self.p50_ms:.1f}ms | p95: {self.p95_ms:.1f}ms | p99: {self.p99_ms:.1f}ms",
            f"Errors: {self.n_errors}/{self.n_requests} ({self.error_rate * 100:.2f}%)",
            f"SLA passed: {self.passed_sla}",
        ]
        if self.violations:
            for v in self.violations:
                lines.append(f"  VIOLATION: {v}")
        return "\n".join(lines)


# ── LoadTestRunner ─────────────────────────────────────────────────────────────

class LoadTestRunner:
    """Runs a synchronous load simulation against a callable.

    Simulates load by calling predict_fn N times and collecting latencies.
    Not a true distributed load tester (use Locust for that) — useful for
    quick validation of inference performance locally.

    Args:
        config:      LoadTestConfig specifying RPS, duration, SLA thresholds.
        predict_fn:  Callable(dict) → any; should raise on error.
    """

    def __init__(
        self,
        config: LoadTestConfig,
        predict_fn: Callable[[dict], object],
    ) -> None:
        self.config = config
        self.predict_fn = predict_fn
        self.profiler = LatencyProfiler()

    def run(self, sample_features: dict | pd.DataFrame) -> LoadTestResult:
        """Execute the load simulation and return a LoadTestResult.

        Args:
            sample_features: Feature input to send to predict_fn.
                             If DataFrame, a random row is sampled each call.

        Returns:
            LoadTestResult with p50/p95/p99 and SLA pass/fail.
        """
        total_requests = (
            self.config.max_requests
            or (self.config.ramp_seconds + self.config.hold_seconds) * self.config.target_rps
        )
        total_requests = max(1, total_requests)

        n_errors = 0
        self.profiler.reset()

        rng = np.random.default_rng(42)

        log.info("Starting load test: %d requests, target=%d RPS", total_requests, self.config.target_rps)
        wall_start = time.perf_counter()

        for i in range(total_requests):
            if isinstance(sample_features, pd.DataFrame):
                idx = int(rng.integers(0, len(sample_features)))
                features = sample_features.iloc[idx].to_dict()
            else:
                features = sample_features

            try:
                with self.profiler.measure("predict"):
                    self.predict_fn(features)
            except Exception:
                n_errors += 1

        wall_elapsed = time.perf_counter() - wall_start
        rps_achieved = total_requests / max(wall_elapsed, 1e-9)

        latencies = self.profiler._timings.get("predict", [])
        if latencies:
            arr = np.array(latencies)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
        else:
            p50 = p95 = p99 = 0.0

        error_rate = n_errors / max(total_requests, 1)
        violations: list[str] = []

        if p95 > self.config.p95_threshold_ms:
            violations.append(f"p95={p95:.1f}ms > threshold={self.config.p95_threshold_ms}ms")
        if p99 > self.config.p99_threshold_ms:
            violations.append(f"p99={p99:.1f}ms > threshold={self.config.p99_threshold_ms}ms")
        if error_rate > self.config.error_rate_threshold:
            violations.append(f"error_rate={error_rate:.3f} > threshold={self.config.error_rate_threshold}")

        result = LoadTestResult(
            rps_achieved=rps_achieved,
            p50_ms=p50,
            p95_ms=p95,
            p99_ms=p99,
            error_rate=error_rate,
            n_requests=total_requests,
            n_errors=n_errors,
            passed_sla=len(violations) == 0,
            violations=violations,
        )

        log.info("Load test complete:\n%s", result.summary())
        return result
