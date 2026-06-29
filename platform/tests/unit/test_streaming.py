"""Tests for features/streaming.py — PushEvent, PushSchema, PushSource, OnDemandTransform, StreamProcessor."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from features.feature_store import InMemoryOnlineStore
from features.streaming import (
    COMPOSITE_RISK_TRANSFORM,
    REAL_TIME_TX_SCHEMA,
    REAL_TIME_TX_SOURCE,
    OnDemandTransform,
    PushEvent,
    PushSchema,
    PushSource,
    StreamProcessor,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── PushEvent ──────────────────────────────────────────────────────────────────

class TestPushEvent:
    def test_basic(self) -> None:
        e = PushEvent("c1", {"pay_ratio": 0.25})
        assert e.entity_key == "c1"
        assert e.features["pay_ratio"] == 0.25

    def test_empty_entity_key_raises(self) -> None:
        with pytest.raises(ValueError, match="entity_key"):
            PushEvent("", {"pay_ratio": 0.25})

    def test_non_dict_features_raises(self) -> None:
        with pytest.raises(TypeError, match="dict"):
            PushEvent("c1", [1, 2, 3])  # type: ignore[arg-type]

    def test_default_timestamp_is_utc(self) -> None:
        e = PushEvent("c1", {})
        assert e.timestamp.tzinfo is not None


# ── PushSchema ────────────────────────────────────────────────────────────────

class TestPushSchema:
    def test_valid_event_no_errors(self) -> None:
        schema = PushSchema(
            required_fields=["pay_ratio"],
            feature_types={"pay_ratio": "float"},
        )
        event = PushEvent("c1", {"pay_ratio": 0.25})
        assert schema.validate(event) == []

    def test_missing_required_field(self) -> None:
        schema = PushSchema(required_fields=["pay_ratio"])
        event = PushEvent("c1", {"util_rate": 0.5})
        errors = schema.validate(event)
        assert any("pay_ratio" in e for e in errors)

    def test_wrong_type_detected(self) -> None:
        schema = PushSchema(feature_types={"count": "int"})
        event = PushEvent("c1", {"count": "three"})
        errors = schema.validate(event)
        assert any("count" in e for e in errors)

    def test_none_value_allowed_even_with_type_constraint(self) -> None:
        schema = PushSchema(feature_types={"pay_ratio": "float"})
        event = PushEvent("c1", {"pay_ratio": None})
        assert schema.validate(event) == []

    def test_int_acceptable_for_float_field(self) -> None:
        schema = PushSchema(feature_types={"amount": "float"})
        event = PushEvent("c1", {"amount": 100})  # int, but acceptable for float
        assert schema.validate(event) == []


# ── PushSource ────────────────────────────────────────────────────────────────

class TestPushSource:
    def test_push_accepted(self) -> None:
        src = PushSource("tx_src")
        errors = src.push(PushEvent("c1", {"pay_ratio": 0.25}))
        assert errors == []
        assert src.pending_count() == 1

    def test_push_rejected_by_schema(self) -> None:
        schema = PushSchema(required_fields=["pay_ratio"])
        src = PushSource("tx_src", schema=schema)
        errors = src.push(PushEvent("c1", {"util_rate": 0.5}))  # missing pay_ratio
        assert len(errors) > 0
        assert src.pending_count() == 0
        assert src.dropped_count() == 1

    def test_flush_writes_to_online_store(self) -> None:
        src = PushSource("tx_src")
        online = InMemoryOnlineStore()
        src.push(PushEvent("c1", {"pay_ratio": 0.25}))
        count = src.flush(online)
        assert count == 1
        result = online.get("c1", ["pay_ratio"])
        assert result["pay_ratio"] == pytest.approx(0.25)

    def test_flush_clears_buffer(self) -> None:
        src = PushSource("tx_src")
        online = InMemoryOnlineStore()
        src.push(PushEvent("c1", {"pay_ratio": 0.25}))
        src.flush(online)
        assert src.pending_count() == 0

    def test_flush_deduplicates_by_entity_key(self) -> None:
        src = PushSource("tx_src")
        online = InMemoryOnlineStore()
        t1 = _now()
        t2 = t1 + timedelta(seconds=10)
        src.push(PushEvent("c1", {"pay_ratio": 0.10}, timestamp=t1))
        src.push(PushEvent("c1", {"pay_ratio": 0.25}, timestamp=t2))  # newer
        count = src.flush(online)
        assert count == 1  # deduplicated to one entity
        result = online.get("c1", ["pay_ratio"])
        assert result["pay_ratio"] == pytest.approx(0.25)  # most recent wins

    def test_flush_empty_buffer_returns_zero(self) -> None:
        src = PushSource("tx_src")
        online = InMemoryOnlineStore()
        assert src.flush(online) == 0

    def test_max_buffer_size_drops_excess(self) -> None:
        src = PushSource("tx_src", max_buffer_size=2)
        for i in range(5):
            src.push(PushEvent(f"c{i}", {"pay_ratio": 0.1}))
        assert src.pending_count() == 2
        assert src.dropped_count() == 3

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            PushSource("")

    def test_total_pushed_counts_accepted(self) -> None:
        src = PushSource("tx_src")
        src.push(PushEvent("c1", {}))
        src.push(PushEvent("c2", {}))
        assert src.total_pushed() == 2


# ── OnDemandTransform ─────────────────────────────────────────────────────────

class TestOnDemandTransform:
    def test_basic_transform(self) -> None:
        t = OnDemandTransform(
            name="ratio",
            fn=lambda f: {"double": (f.get("x") or 0) * 2},
            input_features=["x"],
            output_features=["double"],
        )
        result = t.transform({"x": 5.0})
        assert result["double"] == pytest.approx(10.0)

    def test_missing_input_passed_as_none(self) -> None:
        t = OnDemandTransform(
            name="safe",
            fn=lambda f: {"val": 0 if f.get("x") is None else f["x"]},
            input_features=["x"],
            output_features=["val"],
        )
        result = t.transform({})
        assert result["val"] == 0

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            OnDemandTransform("", fn=lambda f: {})

    def test_non_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            OnDemandTransform("t", fn="not_a_function")  # type: ignore[arg-type]

    def test_composite_risk_transform(self) -> None:
        result = COMPOSITE_RISK_TRANSFORM.transform({"util_rate": 0.9, "pay_ratio": 0.1})
        assert "composite_risk_score" in result
        assert result["high_risk_flag"] == 1

    def test_high_risk_flag_false_below_threshold(self) -> None:
        result = COMPOSITE_RISK_TRANSFORM.transform({"util_rate": 0.5, "pay_ratio": 0.8})
        assert result["high_risk_flag"] == 0


# ── StreamProcessor ───────────────────────────────────────────────────────────

class TestStreamProcessor:
    def test_register_and_process(self) -> None:
        src = PushSource("src_a")
        proc = StreamProcessor()
        proc.register(src)
        events = [PushEvent("c1", {"pay_ratio": 0.25}, source_name="src_a")]
        proc.process(events)
        assert src.pending_count() == 1

    def test_unknown_source_name_skipped(self) -> None:
        proc = StreamProcessor()
        events = [PushEvent("c1", {}, source_name="unknown")]
        errors = proc.process(events)
        assert errors == []  # no crash, just skipped

    def test_flush_all_writes_to_online_store(self) -> None:
        src = PushSource("src_a")
        proc = StreamProcessor({"src_a": src})
        online = InMemoryOnlineStore()
        src.push(PushEvent("c1", {"pay_ratio": 0.25}))
        total = proc.flush_all(online)
        assert total == 1

    def test_pending_count_across_sources(self) -> None:
        src1 = PushSource("a")
        src2 = PushSource("b")
        proc = StreamProcessor({"a": src1, "b": src2})
        src1.push(PushEvent("c1", {}))
        src1.push(PushEvent("c2", {}))
        src2.push(PushEvent("c3", {}))
        assert proc.pending_count() == 3

    def test_real_time_tx_schema(self) -> None:
        event = PushEvent("c1", {"tx_count_last_hour": 3, "last_tx_amount": 150.0})
        errors = REAL_TIME_TX_SCHEMA.validate(event)
        assert errors == []

    def test_real_time_tx_source_exists(self) -> None:
        assert REAL_TIME_TX_SOURCE.name == "real_time_tx"
        assert REAL_TIME_TX_SOURCE.max_buffer_size == 10_000
