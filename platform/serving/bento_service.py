"""BentoML-style service abstraction: runners, adaptive batching, bento packaging.

This module implements the BentoML serving concepts WITHOUT requiring the bentoml
package to be installed. It provides:

    AdaptiveBatcher  — queues rows, dispatches in batches when full or timeout hits
    RunnerConfig     — configuration for a single runner (batch size, latency budget)
    BentoServiceConfig — full service configuration for packaging
    BentoPackager    — generates bento.yaml and directory structure for deployment

The AdaptiveBatcher is the core concept from Day 26: it reduces inference overhead
by grouping individual requests into a single model call.

See: docs/phase4/day26_bentoml.md for theory.

Usage:
    from serving.bento_service import AdaptiveBatcher, RunnerConfig

    config = RunnerConfig(max_batch_size=128, max_latency_ms=10)
    batcher = AdaptiveBatcher(config, predict_fn=runner.predict_batch)

    # In production: await batcher.submit(features_df)
    # Here (sync test): batcher.flush_now(batch_df)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import json

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class RunnerConfig:
    """Configuration for a BentoML-style model runner.

    Attributes:
        max_batch_size:  Maximum number of rows per batch dispatch.
        max_latency_ms:  Maximum milliseconds to wait for a batch to fill.
                         When this expires, a partial batch is dispatched.
        n_workers:       Number of parallel runner processes (default 1).
        name:            Runner identifier (used in metrics and logs).
    """

    max_batch_size: int = 128
    max_latency_ms: int = 10
    n_workers: int = 1
    name: str = "credit_risk_runner"

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_latency_ms < 0:
            raise ValueError("max_latency_ms must be >= 0")
        if self.n_workers < 1:
            raise ValueError("n_workers must be >= 1")


@dataclass
class BatchDispatchResult:
    """Result of a single batch dispatch.

    Attributes:
        n_rows:       Number of rows processed.
        scores:       Positive-class probabilities, shape [n_rows].
        latency_ms:   Wall-clock time for the dispatch in milliseconds.
        batch_id:     Monotonically increasing batch counter.
        waited_ms:    Time items waited in queue before dispatch.
    """

    n_rows: int
    scores: np.ndarray
    latency_ms: float
    batch_id: int
    waited_ms: float


class AdaptiveBatcher:
    """Synchronous adaptive batcher — groups rows and dispatches in batches.

    The real BentoML AdaptiveBatcher is async (asyncio.Queue). This implementation
    is synchronous for testability. It accumulates rows in an internal queue and
    dispatches when either:
        - Queue reaches `max_batch_size`, OR
        - `max_latency_ms` has elapsed since the first item was enqueued.

    In production with BentoML, this is handled transparently per-request.
    Here we expose `submit()` and `flush_now()` to make the batching behaviour
    observable in tests and notebooks.

    Args:
        config:     RunnerConfig specifying batch size and latency budget.
        predict_fn: Callable (DataFrame → ndarray of probabilities).
    """

    def __init__(
        self,
        config: RunnerConfig,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
    ) -> None:
        self._config = config
        self._predict_fn = predict_fn
        self._queue: list[pd.DataFrame] = []
        self._queue_start_time: float | None = None
        self._batch_counter: int = 0
        self._total_rows_processed: int = 0
        self._dispatch_latencies: list[float] = []

    @property
    def queue_size(self) -> int:
        """Current number of rows waiting in the queue."""
        return sum(len(df) for df in self._queue)

    @property
    def n_batches_dispatched(self) -> int:
        return self._batch_counter

    def submit(self, df: pd.DataFrame) -> bool:
        """Add rows to the queue. Returns True if a dispatch was triggered.

        A dispatch is triggered when the queue reaches max_batch_size OR
        when max_latency_ms has elapsed since the first enqueue.

        Args:
            df: One or more feature rows to score.

        Returns:
            True if a batch was dispatched; False if rows are still queued.
        """
        if self._queue_start_time is None:
            self._queue_start_time = time.perf_counter()

        self._queue.append(df)
        elapsed_ms = (time.perf_counter() - self._queue_start_time) * 1000

        should_dispatch = (
            self.queue_size >= self._config.max_batch_size
            or elapsed_ms >= self._config.max_latency_ms
        )

        if should_dispatch:
            self._dispatch()
            return True
        return False

    def flush_now(self, df: pd.DataFrame | None = None) -> BatchDispatchResult:
        """Force an immediate dispatch of the current queue (plus optional df).

        Useful for end-of-window flushing and testing.

        Args:
            df: Optional additional rows to include before flushing.

        Returns:
            BatchDispatchResult with scores and timing.
        """
        if df is not None:
            self._queue.append(df)
            if self._queue_start_time is None:
                self._queue_start_time = time.perf_counter()

        return self._dispatch()

    def _dispatch(self) -> BatchDispatchResult:
        """Internal: concatenate queue, call predict_fn, reset queue."""
        if not self._queue:
            raise RuntimeError("Cannot dispatch an empty queue")

        waited_ms = 0.0
        if self._queue_start_time is not None:
            waited_ms = (time.perf_counter() - self._queue_start_time) * 1000

        batch_df = pd.concat(self._queue, ignore_index=True)
        n_rows = len(batch_df)

        start = time.perf_counter()
        scores = self._predict_fn(batch_df)
        latency_ms = (time.perf_counter() - start) * 1000

        self._batch_counter += 1
        self._total_rows_processed += n_rows
        self._dispatch_latencies.append(latency_ms)

        batch_id = self._batch_counter
        log.info(
            "Batch %d dispatched: %d rows, latency=%.1fms, waited=%.1fms",
            batch_id, n_rows, latency_ms, waited_ms,
        )

        # Reset queue
        self._queue = []
        self._queue_start_time = None

        return BatchDispatchResult(
            n_rows=n_rows,
            scores=np.asarray(scores),
            latency_ms=latency_ms,
            batch_id=batch_id,
            waited_ms=waited_ms,
        )

    def stats(self) -> dict[str, Any]:
        """Return throughput and latency stats for all dispatched batches."""
        if not self._dispatch_latencies:
            return {
                "n_batches": 0,
                "total_rows": 0,
                "mean_batch_latency_ms": 0.0,
                "p99_batch_latency_ms": 0.0,
            }
        arr = np.array(self._dispatch_latencies)
        return {
            "n_batches": self._batch_counter,
            "total_rows": self._total_rows_processed,
            "mean_batch_latency_ms": float(arr.mean()),
            "p99_batch_latency_ms": float(np.percentile(arr, 99)),
        }


@dataclass
class BentoServiceConfig:
    """Configuration for packaging a model as a Bento.

    Attributes:
        name:          Service name (used as image name).
        version:       Semantic version string.
        model_path:    Path to the model artifact (.onnx or .pkl).
        labels:        Key-value labels for the bento (team, env, etc.).
        runner_config: Runner configuration (batch size, latency).
    """

    name: str
    version: str
    model_path: Path
    labels: dict[str, str] = field(default_factory=dict)
    runner_config: RunnerConfig = field(default_factory=RunnerConfig)


class BentoPackager:
    """Generates a Bento directory structure for deployment.

    Writes:
        {output_dir}/bento.yaml     ← service metadata
        {output_dir}/runner.json    ← runner configuration
        {output_dir}/README.md      ← auto-generated service description

    The actual bentoml build step (bentoml CLI) would use these files.
    This class lets us test and inspect the packaging logic without the
    bentoml package installed.
    """

    def package(self, config: BentoServiceConfig, output_dir: Path) -> Path:
        """Write the Bento manifest to output_dir.

        Args:
            config:     BentoServiceConfig describing the service.
            output_dir: Directory to write the bento manifest into.

        Returns:
            Path to the written bento.yaml.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        bento_yaml = {
            "service": f"{config.name}:{config.version}",
            "name": config.name,
            "version": config.version,
            "model": str(config.model_path),
            "labels": config.labels,
            "runner": {
                "name": config.runner_config.name,
                "max_batch_size": config.runner_config.max_batch_size,
                "max_latency_ms": config.runner_config.max_latency_ms,
                "n_workers": config.runner_config.n_workers,
            },
        }

        bento_path = output_dir / "bento.yaml"
        import yaml  # PyYAML is already a project dependency
        with open(bento_path, "w") as f:
            yaml.dump(bento_yaml, f, default_flow_style=False)

        runner_path = output_dir / "runner.json"
        runner_path.write_text(json.dumps({
            "name": config.runner_config.name,
            "max_batch_size": config.runner_config.max_batch_size,
            "max_latency_ms": config.runner_config.max_latency_ms,
        }, indent=2))

        log.info(
            "Bento manifest written to %s (service=%s:%s)",
            output_dir, config.name, config.version,
        )
        return bento_path
