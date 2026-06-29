"""Tests for monitoring/prediction_logger.py — PredictionLogEntry, PredictionLogger."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from monitoring.prediction_logger import PredictionLogEntry, PredictionLogger


def _entry(**kwargs) -> PredictionLogEntry:
    defaults = dict(
        prediction_id="pred-001",
        correlation_id="corr-001",
        entity_key="c123",
        model_version="v1.0",
        score=0.7,
        decision="approve",
    )
    defaults.update(kwargs)
    return PredictionLogEntry(**defaults)


# ── PredictionLogEntry ─────────────────────────────────────────────────────────

class TestPredictionLogEntry:
    def test_basic(self) -> None:
        e = _entry()
        assert e.score == 0.7
        assert e.environment == "prod"

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="prediction_id"):
            _entry(prediction_id="")

    def test_invalid_score_raises(self) -> None:
        with pytest.raises(ValueError, match="score"):
            _entry(score=1.5)

    def test_invalid_decision_raises(self) -> None:
        with pytest.raises(ValueError, match="decision"):
            _entry(decision="maybe")

    def test_to_dict_keys(self) -> None:
        d = _entry().to_dict()
        for key in ["prediction_id", "correlation_id", "entity_key", "score",
                    "decision", "features", "prediction_ts", "latency_ms"]:
            assert key in d

    def test_to_json_valid(self) -> None:
        j = _entry().to_json()
        parsed = json.loads(j)
        assert parsed["score"] == 0.7

    def test_round_trip_from_dict(self) -> None:
        orig = _entry(features={"pay_ratio": 0.5})
        restored = PredictionLogEntry.from_dict(orig.to_dict())
        assert restored.prediction_id == orig.prediction_id
        assert restored.features["pay_ratio"] == 0.5

    def test_from_dict_invalid_ts_fallback(self) -> None:
        d = _entry().to_dict()
        d["prediction_ts"] = "not-a-date"
        restored = PredictionLogEntry.from_dict(d)
        assert isinstance(restored.prediction_ts, datetime)


# ── PredictionLogger ───────────────────────────────────────────────────────────

class TestPredictionLogger:
    def test_empty_log_path_raises(self) -> None:
        with pytest.raises(ValueError, match="log_path"):
            PredictionLogger("")

    def test_log_creates_file(self, tmp_path: Path) -> None:
        log = tmp_path / "preds.jsonl"
        logger = PredictionLogger(str(log), model_version="v1")
        logger.log("c1", 0.6, "approve")
        assert log.exists()

    def test_log_returns_entry(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "p.jsonl"), model_version="v1")
        e = logger.log("c1", 0.6, "approve")
        assert e.entity_key == "c1"
        assert e.model_version == "v1"

    def test_prediction_id_auto_generated(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "p.jsonl"))
        e1 = logger.log("c1", 0.6, "approve")
        e2 = logger.log("c2", 0.4, "decline")
        assert e1.prediction_id != e2.prediction_id

    def test_correlation_id_carried(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "p.jsonl"))
        e = logger.log("c1", 0.6, "approve", correlation_id="req-abc")
        assert e.correlation_id == "req-abc"

    def test_features_captured(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "p.jsonl"))
        e = logger.log("c1", 0.6, "approve", features={"pay_ratio": 0.5, "util_rate": 0.3})
        assert e.features["pay_ratio"] == 0.5

    def test_read_log_returns_entries(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "p.jsonl"))
        logger.log("c1", 0.6, "approve")
        logger.log("c2", 0.3, "decline")
        entries = logger.read_log()
        assert len(entries) == 2

    def test_read_log_n_last(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "p.jsonl"))
        for i in range(5):
            logger.log(f"c{i}", 0.5, "review")
        entries = logger.read_log(n_last=2)
        assert len(entries) == 2

    def test_read_log_empty_file_returns_empty(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "missing.jsonl"))
        assert logger.read_log() == []

    def test_total_written_counter(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "p.jsonl"))
        logger.log("c1", 0.6, "approve")
        logger.log("c2", 0.4, "decline")
        assert logger.total_written() == 2

    def test_buffer_delays_flush(self, tmp_path: Path) -> None:
        log_path = tmp_path / "p.jsonl"
        logger = PredictionLogger(str(log_path), buffer_size=3)
        logger.log("c1", 0.6, "approve")
        logger.log("c2", 0.4, "decline")
        # Only 2 logged, buffer_size=3 → not yet flushed
        assert logger.pending() == 2
        assert not log_path.exists()

    def test_buffer_flushes_at_capacity(self, tmp_path: Path) -> None:
        log_path = tmp_path / "p.jsonl"
        logger = PredictionLogger(str(log_path), buffer_size=2)
        logger.log("c1", 0.6, "approve")
        logger.log("c2", 0.4, "decline")
        # 2nd write hits capacity → flush
        assert log_path.exists()
        assert logger.pending() == 0

    def test_environment_embedded(self, tmp_path: Path) -> None:
        logger = PredictionLogger(str(tmp_path / "p.jsonl"), environment="shadow")
        e = logger.log("c1", 0.5, "review")
        assert e.environment == "shadow"

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "logs" / "model_v1" / "preds.jsonl"
        logger = PredictionLogger(str(nested))
        logger.log("c1", 0.5, "review")
        assert nested.exists()
