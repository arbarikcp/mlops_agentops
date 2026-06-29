"""Streaming feature ingestion: PushSource, PushEvent, OnDemandTransform, StreamProcessor.

Implements Feast-style push source and on-demand feature view concepts in pure Python.
No Kafka or Flink required — events are buffered in-memory and flushed to the online store.

See: docs/phase6/day42_streaming_features.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

log = logging.getLogger(__name__)


# ── Push Event ────────────────────────────────────────────────────────────────

@dataclass
class PushEvent:
    """A single feature update event from a streaming source.

    Attributes:
        entity_key:  Entity identifier (e.g. "customer_id:C1").
        features:    Dict of feature_name → value.
        timestamp:   When the event occurred (UTC).
        source_name: Which push source emitted this event.
    """

    entity_key: str
    features: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_name: str = ""

    def __post_init__(self) -> None:
        if not self.entity_key:
            raise ValueError("PushEvent.entity_key must not be empty")
        if not isinstance(self.features, dict):
            raise TypeError("PushEvent.features must be a dict")


# ── Push Schema ───────────────────────────────────────────────────────────────

@dataclass
class PushSchema:
    """Schema validation for push events.

    Attributes:
        required_fields: Field names that must appear in every event.
        feature_types:   Expected Python type name per field ("float", "int", "bool", "str").
    """

    required_fields: list[str] = field(default_factory=list)
    feature_types: dict[str, str] = field(default_factory=dict)

    _TYPE_MAP: dict[str, type] = field(default_factory=lambda: {
        "float": (float, int),  # int is acceptable for float fields
        "int": int,
        "bool": bool,
        "str": str,
    }, init=False, repr=False)

    def validate(self, event: PushEvent) -> list[str]:
        """Return a list of validation error strings (empty if valid)."""
        errors: list[str] = []
        for req in self.required_fields:
            if req not in event.features:
                errors.append(f"Missing required field: '{req}'")

        type_map: dict[str, Any] = {
            "float": (float, int),
            "int": (int,),
            "bool": (bool,),
            "str": (str,),
        }
        for feat_name, expected_type in self.feature_types.items():
            value = event.features.get(feat_name)
            if value is None:
                continue  # nullable — schema allows None
            allowed = type_map.get(expected_type, (object,))
            if not isinstance(value, allowed):
                errors.append(
                    f"Field '{feat_name}' expected {expected_type}, got {type(value).__name__}"
                )
        return errors


# ── Push Source ───────────────────────────────────────────────────────────────

class PushSource:
    """Accepts streaming feature events and flushes them to the online store.

    Events are buffered in-memory. Call `flush()` to write all buffered events
    to the online store (typically triggered by a periodic job or request-time hook).

    Args:
        name:           Identifier for this push source.
        schema:         Validation schema; None means no validation.
        max_buffer_size: Drop events when buffer exceeds this size (0 = unlimited).
    """

    def __init__(
        self,
        name: str,
        schema: PushSchema | None = None,
        max_buffer_size: int = 0,
    ) -> None:
        if not name:
            raise ValueError("PushSource.name must not be empty")
        self.name = name
        self.schema = schema
        self.max_buffer_size = max_buffer_size
        self._buffer: list[PushEvent] = []
        self._dropped: int = 0
        self._total_pushed: int = 0

    def push(self, event: PushEvent) -> list[str]:
        """Add an event to the buffer after schema validation.

        Args:
            event: PushEvent to buffer.

        Returns:
            List of validation errors (empty if accepted).
        """
        errors: list[str] = []
        if self.schema:
            errors = self.schema.validate(event)
        if errors:
            self._dropped += 1
            log.warning("PushSource '%s' dropped event for '%s': %s", self.name, event.entity_key, errors)
            return errors

        if self.max_buffer_size > 0 and len(self._buffer) >= self.max_buffer_size:
            self._dropped += 1
            log.warning("PushSource '%s' buffer full — dropping event for '%s'", self.name, event.entity_key)
            return ["Buffer full"]

        event.source_name = self.name
        self._buffer.append(event)
        self._total_pushed += 1
        return []

    def flush(self, online_store: Any, ttl_seconds: int = 3600) -> int:
        """Write all buffered events to the online store and clear the buffer.

        For each entity key, uses the **most recent** event (by timestamp)
        to avoid overwriting newer data with older events.

        Args:
            online_store: InMemoryOnlineStore (or compatible) to write to.
            ttl_seconds:  TTL for each written entry.

        Returns:
            Number of distinct entity keys written.
        """
        if not self._buffer:
            return 0

        # Deduplicate: keep most recent event per entity key
        latest: dict[str, PushEvent] = {}
        for event in self._buffer:
            existing = latest.get(event.entity_key)
            if existing is None or event.timestamp > existing.timestamp:
                latest[event.entity_key] = event

        for entity_key, event in latest.items():
            online_store.put(entity_key, event.features, ttl_seconds=ttl_seconds)

        flushed = len(latest)
        self._buffer.clear()
        log.info("PushSource '%s' flushed %d entities", self.name, flushed)
        return flushed

    def pending_count(self) -> int:
        """Number of events currently in the buffer."""
        return len(self._buffer)

    def dropped_count(self) -> int:
        """Total number of events dropped (validation failure or buffer full)."""
        return self._dropped

    def total_pushed(self) -> int:
        """Total events successfully accepted (not dropped)."""
        return self._total_pushed


# ── On-Demand Transform ────────────────────────────────────────────────────────

@dataclass
class OnDemandTransform:
    """A feature transformation applied at request time (not pre-materialised).

    Computes derived features from already-retrieved online features + request payload.
    No I/O occurs; all computation is in-memory.

    Attributes:
        name:            Identifier for this transform.
        fn:              Callable(features: dict) → dict of new feature values.
        input_features:  Which feature names this transform reads.
        output_features: Which feature names this transform produces.
    """

    name: str
    fn: Callable[[dict[str, Any]], dict[str, Any]]
    input_features: list[str] = field(default_factory=list)
    output_features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("OnDemandTransform.name must not be empty")
        if not callable(self.fn):
            raise TypeError("OnDemandTransform.fn must be callable")

    def transform(self, feature_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply the transform to a feature dictionary.

        Missing input features are passed as None — the fn must handle them.

        Args:
            feature_dict: All available feature values for one entity.

        Returns:
            Dict containing only the newly computed output features.
        """
        inputs = {k: feature_dict.get(k) for k in self.input_features}
        return self.fn(inputs)


# ── Stream Processor ──────────────────────────────────────────────────────────

class StreamProcessor:
    """Manages multiple push sources and flushes them to the online store.

    Provides a unified interface for processing batches of raw events:
    1. Route each event to the appropriate push source by source_name.
    2. Apply optional transforms.
    3. Flush all buffered events on demand.

    Args:
        push_sources: Mapping of source_name → PushSource.
    """

    def __init__(self, push_sources: dict[str, PushSource] | None = None) -> None:
        self.push_sources: dict[str, PushSource] = push_sources or {}

    def register(self, source: PushSource) -> None:
        """Add a push source to this processor."""
        self.push_sources[source.name] = source

    def process(self, events: list[PushEvent]) -> list[str]:
        """Route events to their respective push sources.

        Events whose source_name is not registered are silently skipped
        (logged as warnings).

        Args:
            events: List of PushEvents to process.

        Returns:
            List of all validation error strings across all events.
        """
        all_errors: list[str] = []
        for event in events:
            source = self.push_sources.get(event.source_name)
            if source is None:
                log.warning("No push source registered for '%s' — skipping", event.source_name)
                continue
            errors = source.push(event)
            all_errors.extend(errors)
        return all_errors

    def flush_all(self, online_store: Any, ttl_seconds: int = 3600) -> int:
        """Flush all push sources to the online store.

        Returns:
            Total number of distinct entity keys written across all sources.
        """
        total = 0
        for source in self.push_sources.values():
            total += source.flush(online_store, ttl_seconds=ttl_seconds)
        return total

    def pending_count(self) -> int:
        """Total buffered events across all push sources."""
        return sum(s.pending_count() for s in self.push_sources.values())


# ── Credit Risk Streaming Definitions ─────────────────────────────────────────

REAL_TIME_TX_SCHEMA = PushSchema(
    required_fields=["tx_count_last_hour", "last_tx_amount"],
    feature_types={
        "tx_count_last_hour": "int",
        "last_tx_amount": "float",
        "session_duration_s": "float",
    },
)

REAL_TIME_TX_SOURCE = PushSource(
    name="real_time_tx",
    schema=REAL_TIME_TX_SCHEMA,
    max_buffer_size=10_000,
)

COMPOSITE_RISK_TRANSFORM = OnDemandTransform(
    name="composite_risk",
    fn=lambda f: {
        "composite_risk_score": (
            (f.get("util_rate") or 0.0) * 0.7
            + (1.0 - (f.get("pay_ratio") or 0.0)) * 0.3
        ),
        "high_risk_flag": int((f.get("util_rate") or 0.0) > 0.8),
    },
    input_features=["util_rate", "pay_ratio"],
    output_features=["composite_risk_score", "high_risk_flag"],
)
