"""Inference abstractions: online single-row, batch chunked, latency tracking.

Three patterns covered:
    Online  — one prediction per HTTP request, caller waits (p99 SLA driven)
    Batch   — chunked prediction over a large DataFrame (scheduled job)
    Streaming — not implemented here; see docs/phase4/day23_inference_patterns.md

ModelRunner holds a pre-loaded ONNX session. Load once at startup; reuse.
Never reload the model inside the request handler — cold load costs 100–500ms.

See: docs/phase4/day23_inference_patterns.md for theory.

Usage:
    from serving.inference import ModelRunner

    runner = ModelRunner(model_path=Path("models/credit_risk_model.onnx"),
                         model_version="v1.0", threshold=0.5)
    runner.load()

    result = runner.predict_single(features_dict)
    results = runner.predict_batch(features_df, chunk_size=512)
    print(runner.latency.summary())
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 512
_DEFAULT_THRESHOLD = 0.5


@dataclass
class PredictionResult:
    """Single-row inference result.

    Attributes:
        applicant_id:  Row identifier (0 if not provided).
        score:         Positive-class probability from the model.
        label:         Binary decision (1 = default, 0 = no default).
        latency_ms:    Wall-clock time for this prediction in milliseconds.
        model_version: Version tag from ModelRunner.
        timestamp:     ISO-8601 timestamp of the prediction.
    """

    applicant_id: int
    score: float
    label: int
    latency_ms: float
    model_version: str
    timestamp: str


class LatencyTracker:
    """Collects latency samples and computes percentile statistics.

    Thread-safety: not thread-safe. Use one instance per worker process.
    """

    def __init__(self, max_samples: int = 10_000) -> None:
        self._samples: list[float] = []
        self._max_samples = max_samples

    def record(self, latency_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        if len(self._samples) >= self._max_samples:
            self._samples.pop(0)
        self._samples.append(latency_ms)

    def percentile(self, p: float) -> float:
        """Return the p-th percentile of recorded latencies.

        Args:
            p: Percentile in [0, 100].

        Returns:
            Latency in ms, or 0.0 if no samples.
        """
        if not self._samples:
            return 0.0
        return float(np.percentile(self._samples, p))

    def summary(self) -> dict[str, float]:
        """Return p50/p90/p95/p99 latency summary."""
        if not self._samples:
            return {"n": 0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
        arr = np.array(self._samples, dtype=float)
        return {
            "n": len(arr),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def reset(self) -> None:
        """Clear all recorded samples."""
        self._samples = []

    @property
    def n_samples(self) -> int:
        return len(self._samples)


class ModelRunner:
    """Wraps an ONNX InferenceSession for online and batch inference.

    Load once at application startup with `runner.load()`.
    The session is kept in memory and reused for all subsequent calls.

    Args:
        model_path:    Path to the .onnx model file.
        model_version: Human-readable version tag (stored in PredictionResult).
        threshold:     Decision threshold for binary label (default 0.5).
        feature_names: Optional ordered column list; enforces column alignment.
    """

    def __init__(
        self,
        model_path: Path,
        model_version: str = "unknown",
        threshold: float = _DEFAULT_THRESHOLD,
        feature_names: list[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_version = model_version
        self.threshold = threshold
        self.feature_names = feature_names
        self._session: Any = None
        self._input_name: str = "float_input"
        self.latency = LatencyTracker()

    def load(self) -> None:
        """Load the ONNX model into an InferenceSession.

        Must be called before any predict_* method.

        Raises:
            ImportError: onnxruntime not installed.
            FileNotFoundError: .onnx file does not exist.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required. Install with: uv add onnxruntime"
            ) from exc

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self._session = ort.InferenceSession(str(self.model_path))
        self._input_name = self._session.get_inputs()[0].name
        log.info("Loaded ONNX model from %s (version=%s)", self.model_path, self.model_version)

    def warm_up(self, n_warmup_requests: int = 3) -> None:
        """Send synthetic requests through the model to trigger JIT compilation.

        Call after load() and before accepting real traffic. The first N
        ONNX requests are slower due to graph compilation; warm-up absorbs that.

        Args:
            n_warmup_requests: Number of dummy single-row predictions to run.
        """
        if self._session is None:
            raise RuntimeError("Call runner.load() before warm_up()")

        n_features = self._session.get_inputs()[0].shape[1] or 39
        dummy = np.zeros((1, n_features), dtype=np.float32)
        for _ in range(n_warmup_requests):
            self._session.run(None, {self._input_name: dummy})
        log.info("Warm-up complete (%d dummy requests)", n_warmup_requests)

    def predict_single(
        self,
        features: dict[str, float] | pd.Series | np.ndarray,
        *,
        applicant_id: int = 0,
    ) -> PredictionResult:
        """Score a single applicant.

        Args:
            features:     Feature values — dict, pd.Series, or 1-D ndarray.
            applicant_id: Optional row identifier for the result.

        Returns:
            PredictionResult with score, label, and latency.

        Raises:
            RuntimeError: If load() has not been called.
        """
        if self._session is None:
            raise RuntimeError("Call runner.load() before predict_single()")

        start = time.perf_counter()

        X = self._to_input_array(features, n_rows=1)
        ort_out = self._run_session(X)
        score = float(ort_out[0])
        label = int(score >= self.threshold)

        latency_ms = (time.perf_counter() - start) * 1000
        self.latency.record(latency_ms)

        return PredictionResult(
            applicant_id=applicant_id,
            score=score,
            label=label,
            latency_ms=latency_ms,
            model_version=self.model_version,
            timestamp=_utc_now(),
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        *,
        id_col: str | None = None,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> list[PredictionResult]:
        """Score a DataFrame in chunks.

        Args:
            df:         Feature DataFrame. Columns must match feature_names if set.
            id_col:     Column to use as applicant_id. Defaults to DataFrame index.
            chunk_size: Number of rows per ONNX batch (tune for memory/throughput).

        Returns:
            List of PredictionResult, one per row, in input order.

        Raises:
            RuntimeError: If load() has not been called.
        """
        if self._session is None:
            raise RuntimeError("Call runner.load() before predict_batch()")

        start_total = time.perf_counter()
        results: list[PredictionResult] = []

        for chunk_start in range(0, len(df), chunk_size):
            chunk = df.iloc[chunk_start: chunk_start + chunk_size]
            X = self._to_input_array(chunk, n_rows=len(chunk))
            scores = self._run_session(X)

            for i, score in enumerate(scores):
                score_f = float(score)
                row_idx = chunk_start + i
                applicant_id = (
                    int(chunk[id_col].iloc[i]) if id_col and id_col in chunk.columns
                    else int(df.index[row_idx])
                )
                results.append(PredictionResult(
                    applicant_id=applicant_id,
                    score=score_f,
                    label=int(score_f >= self.threshold),
                    latency_ms=0.0,  # filled below as per-batch average
                    model_version=self.model_version,
                    timestamp=_utc_now(),
                ))

        total_ms = (time.perf_counter() - start_total) * 1000
        per_row_ms = total_ms / max(len(results), 1)
        for r in results:
            object.__setattr__(r, "latency_ms", per_row_ms) if hasattr(r, "__dataclass_fields__") else None

        log.info(
            "Batch inference: %d rows in %.1fms (%.3fms/row)",
            len(results), total_ms, per_row_ms,
        )
        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    def _to_input_array(
        self,
        features: dict | pd.Series | pd.DataFrame | np.ndarray,
        n_rows: int,
    ) -> np.ndarray:
        """Convert various input types to float32 numpy array."""
        if isinstance(features, dict):
            if self.feature_names:
                arr = np.array([[features[k] for k in self.feature_names]], dtype=np.float32)
            else:
                arr = np.array([[v for v in features.values()]], dtype=np.float32)
        elif isinstance(features, pd.Series):
            arr = features.to_numpy(dtype=np.float32).reshape(1, -1)
        elif isinstance(features, pd.DataFrame):
            if self.feature_names:
                arr = features[self.feature_names].to_numpy(dtype=np.float32)
            else:
                arr = features.to_numpy(dtype=np.float32)
        else:
            arr = np.asarray(features, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
        return arr

    def _run_session(self, X: np.ndarray) -> np.ndarray:
        """Run the ONNX session and extract positive-class probabilities."""
        ort_outputs = self._session.run(None, {self._input_name: X})

        # Prefer 2D [N, 2] probability output
        for out in ort_outputs:
            arr = np.array(out)
            if arr.ndim == 2 and arr.shape[1] == 2:
                return arr[:, 1]

        # Fallback: first 1D output
        return np.array(ort_outputs[0]).ravel()


def _utc_now() -> str:
    """ISO-8601 UTC timestamp string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
