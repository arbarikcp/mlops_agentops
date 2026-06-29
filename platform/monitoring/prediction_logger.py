"""Prediction logger: structured JSONL audit log with correlation IDs and replay support.

Day 51 — logs every prediction to a JSONL file with full feature snapshot,
correlation ID, model version, and timing. Supports:
  - Structured log entries with audit-ready schema
  - Correlation IDs (request-scoped) + prediction IDs (outcome-join key)
  - JSONL format (append-only, one JSON object per line)
  - read_log() for feedback loop consumption and replay

Classes:
  PredictionLogEntry — one log record (immutable after creation)
  PredictionLogger   — writes / reads the JSONL log

See: docs/phase7/day51_prediction_logging.md
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── PredictionLogEntry ────────────────────────────────────────────────────────

@dataclass
class PredictionLogEntry:
    """One prediction log record with full audit schema.

    Attributes:
        prediction_id:  UUID unique to this prediction (used to join outcomes).
        correlation_id: Request-scoped ID shared across microservices.
        entity_key:     Customer / entity identifier.
        model_version:  Model version or MLflow run ID.
        score:          Raw probability output (0–1).
        decision:       "approve" / "review" / "decline".
        features:       Feature snapshot captured at inference time.
        prediction_ts:  UTC timestamp of prediction.
        latency_ms:     End-to-end inference latency in milliseconds.
        environment:    "prod" / "staging" / "shadow".
    """

    prediction_id: str
    correlation_id: str
    entity_key: str
    model_version: str
    score: float
    decision: str
    features: dict[str, Any] = field(default_factory=dict)
    prediction_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0
    environment: str = "prod"

    def __post_init__(self) -> None:
        if not self.prediction_id:
            raise ValueError("prediction_id cannot be empty")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0,1], got {self.score}")
        if self.decision not in {"approve", "review", "decline"}:
            raise ValueError(f"decision must be approve/review/decline, got '{self.decision}'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "correlation_id": self.correlation_id,
            "entity_key": self.entity_key,
            "model_version": self.model_version,
            "score": self.score,
            "decision": self.decision,
            "features": self.features,
            "prediction_ts": self.prediction_ts.isoformat(),
            "latency_ms": self.latency_ms,
            "environment": self.environment,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PredictionLogEntry":
        ts = d.get("prediction_ts", "")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.now(timezone.utc)
        return cls(
            prediction_id=d["prediction_id"],
            correlation_id=d.get("correlation_id", ""),
            entity_key=d["entity_key"],
            model_version=d.get("model_version", ""),
            score=float(d["score"]),
            decision=d["decision"],
            features=d.get("features", {}),
            prediction_ts=ts,
            latency_ms=float(d.get("latency_ms", 0.0)),
            environment=d.get("environment", "prod"),
        )


# ── PredictionLogger ──────────────────────────────────────────────────────────

class PredictionLogger:
    """Append-only JSONL prediction logger with read and replay support.

    Args:
        log_path:      Path to the JSONL log file. Created on first write.
        model_version: Model version / run_id to embed in every entry.
        environment:   "prod" / "staging" / "shadow".
        buffer_size:   Number of entries to buffer before flushing (0 = flush every write).
    """

    def __init__(
        self,
        log_path: str,
        model_version: str = "unknown",
        environment: str = "prod",
        buffer_size: int = 0,
    ) -> None:
        if not log_path:
            raise ValueError("log_path cannot be empty")
        self.log_path = Path(log_path)
        self.model_version = model_version
        self.environment = environment
        self.buffer_size = buffer_size
        self._buffer: list[PredictionLogEntry] = []
        self._total_written: int = 0

    def log(
        self,
        entity_key: str,
        score: float,
        decision: str,
        features: dict[str, Any] | None = None,
        latency_ms: float = 0.0,
        correlation_id: str | None = None,
        prediction_id: str | None = None,
    ) -> PredictionLogEntry:
        """Create and log one prediction entry.

        Args:
            entity_key:     Customer identifier.
            score:          Model output probability.
            decision:       approve / review / decline.
            features:       Feature dict at inference time (captured for replay).
            latency_ms:     Inference latency.
            correlation_id: Request-scoped trace ID (generated if not provided).
            prediction_id:  Override auto-generated UUID (for testing).

        Returns:
            The written PredictionLogEntry.
        """
        entry = PredictionLogEntry(
            prediction_id=prediction_id or str(uuid.uuid4()),
            correlation_id=correlation_id or str(uuid.uuid4()),
            entity_key=entity_key,
            model_version=self.model_version,
            score=score,
            decision=decision,
            features=features or {},
            latency_ms=latency_ms,
            environment=self.environment,
        )
        self._buffer.append(entry)
        if self.buffer_size == 0 or len(self._buffer) >= self.buffer_size:
            self.flush()
        return entry

    def flush(self) -> None:
        """Write all buffered entries to the JSONL log file."""
        if not self._buffer:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(entry.to_json() + "\n")
        self._total_written += len(self._buffer)
        self._buffer.clear()

    def read_log(self, n_last: int | None = None) -> list[PredictionLogEntry]:
        """Read entries from the JSONL log.

        Args:
            n_last: Return only the last N entries. If None, return all.

        Returns:
            List of PredictionLogEntry objects, oldest first.
        """
        self.flush()  # ensure buffer is persisted
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").strip().splitlines()
        entries: list[PredictionLogEntry] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(PredictionLogEntry.from_dict(d))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        if n_last is not None:
            return entries[-n_last:]
        return entries

    def total_written(self) -> int:
        """Total entries written (including current session)."""
        return self._total_written + len(self._buffer)

    def pending(self) -> int:
        """Entries in buffer not yet flushed."""
        return len(self._buffer)
