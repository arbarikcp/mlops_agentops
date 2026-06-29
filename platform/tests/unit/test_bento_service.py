"""Tests for serving/bento_service.py — BentoML abstractions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from serving.bento_service import (
    AdaptiveBatcher,
    BatchDispatchResult,
    BentoPackager,
    BentoServiceConfig,
    RunnerConfig,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _dummy_predict(df: pd.DataFrame) -> np.ndarray:
    """Predict function that returns row_index/10 as probability."""
    return np.arange(len(df), dtype=float) / 10.0


@pytest.fixture
def config() -> RunnerConfig:
    return RunnerConfig(max_batch_size=4, max_latency_ms=50)


@pytest.fixture
def batcher(config) -> AdaptiveBatcher:
    return AdaptiveBatcher(config=config, predict_fn=_dummy_predict)


def _df(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"feat": range(n)})


# ── RunnerConfig ──────────────────────────────────────────────────────────────

class TestRunnerConfig:
    def test_defaults(self) -> None:
        cfg = RunnerConfig()
        assert cfg.max_batch_size == 128
        assert cfg.max_latency_ms == 10
        assert cfg.n_workers == 1

    def test_custom_values(self) -> None:
        cfg = RunnerConfig(max_batch_size=512, max_latency_ms=20)
        assert cfg.max_batch_size == 512

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size"):
            RunnerConfig(max_batch_size=0)

    def test_invalid_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="max_latency_ms"):
            RunnerConfig(max_latency_ms=-1)

    def test_invalid_workers_raises(self) -> None:
        with pytest.raises(ValueError, match="n_workers"):
            RunnerConfig(n_workers=0)


# ── AdaptiveBatcher.flush_now ─────────────────────────────────────────────────

class TestAdaptiveBatcherFlushNow:
    def test_returns_batch_dispatch_result(self, batcher) -> None:
        result = batcher.flush_now(_df(3))
        assert isinstance(result, BatchDispatchResult)

    def test_n_rows_correct(self, batcher) -> None:
        result = batcher.flush_now(_df(5))
        assert result.n_rows == 5

    def test_scores_length_matches_rows(self, batcher) -> None:
        result = batcher.flush_now(_df(4))
        assert len(result.scores) == 4

    def test_batch_id_increments(self, batcher) -> None:
        result1 = batcher.flush_now(_df(2))
        result2 = batcher.flush_now(_df(2))
        assert result2.batch_id == result1.batch_id + 1

    def test_latency_ms_non_negative(self, batcher) -> None:
        result = batcher.flush_now(_df(3))
        assert result.latency_ms >= 0.0

    def test_queue_empty_after_flush(self, batcher) -> None:
        batcher.flush_now(_df(3))
        assert batcher.queue_size == 0

    def test_raises_on_empty_queue(self, batcher) -> None:
        with pytest.raises(RuntimeError, match="empty"):
            batcher._dispatch()


# ── AdaptiveBatcher.submit ────────────────────────────────────────────────────

class TestAdaptiveBatcherSubmit:
    def test_does_not_dispatch_when_below_threshold(self, batcher) -> None:
        # max_batch_size=4, submitting 2 rows should NOT dispatch
        dispatched = batcher.submit(_df(2))
        assert dispatched is False
        assert batcher.queue_size == 2

    def test_dispatches_when_batch_full(self, batcher) -> None:
        # max_batch_size=4, submit 4 rows → dispatch
        dispatched = batcher.submit(_df(4))
        assert dispatched is True
        assert batcher.queue_size == 0

    def test_accumulates_across_submits(self, batcher) -> None:
        batcher.submit(_df(1))
        batcher.submit(_df(2))
        assert batcher.queue_size == 3

    def test_n_batches_dispatched_increments(self, batcher) -> None:
        batcher.submit(_df(4))  # triggers dispatch
        assert batcher.n_batches_dispatched == 1

    def test_multiple_dispatches(self, batcher) -> None:
        for _ in range(3):
            batcher.flush_now(_df(2))
        assert batcher.n_batches_dispatched == 3


# ── AdaptiveBatcher.stats ─────────────────────────────────────────────────────

class TestAdaptiveBatcherStats:
    def test_zero_batches(self, batcher) -> None:
        stats = batcher.stats()
        assert stats["n_batches"] == 0
        assert stats["total_rows"] == 0

    def test_stats_after_dispatch(self, batcher) -> None:
        batcher.flush_now(_df(5))
        stats = batcher.stats()
        assert stats["n_batches"] == 1
        assert stats["total_rows"] == 5
        assert stats["mean_batch_latency_ms"] >= 0.0

    def test_total_rows_accumulates(self, batcher) -> None:
        batcher.flush_now(_df(3))
        batcher.flush_now(_df(7))
        assert batcher.stats()["total_rows"] == 10

    def test_p99_latency_present(self, batcher) -> None:
        for _ in range(10):
            batcher.flush_now(_df(2))
        stats = batcher.stats()
        assert "p99_batch_latency_ms" in stats
        assert stats["p99_batch_latency_ms"] >= stats["mean_batch_latency_ms"]


# ── BentoPackager ─────────────────────────────────────────────────────────────

class TestBentoPackager:
    @pytest.fixture
    def svc_config(self, tmp_path) -> BentoServiceConfig:
        return BentoServiceConfig(
            name="credit-risk-api",
            version="1.0.0",
            model_path=tmp_path / "model.onnx",
            labels={"team": "ml", "env": "staging"},
        )

    def test_creates_bento_yaml(self, tmp_path, svc_config) -> None:
        packager = BentoPackager()
        out_dir = tmp_path / "bento"
        bento_path = packager.package(svc_config, out_dir)
        assert bento_path.exists()
        assert bento_path.name == "bento.yaml"

    def test_bento_yaml_has_name(self, tmp_path, svc_config) -> None:
        import yaml
        packager = BentoPackager()
        bento_path = packager.package(svc_config, tmp_path / "bento")
        data = yaml.safe_load(bento_path.read_text())
        assert data["name"] == "credit-risk-api"

    def test_bento_yaml_has_version(self, tmp_path, svc_config) -> None:
        import yaml
        packager = BentoPackager()
        bento_path = packager.package(svc_config, tmp_path / "bento")
        data = yaml.safe_load(bento_path.read_text())
        assert data["version"] == "1.0.0"

    def test_runner_json_created(self, tmp_path, svc_config) -> None:
        packager = BentoPackager()
        out_dir = tmp_path / "bento"
        packager.package(svc_config, out_dir)
        runner_json = out_dir / "runner.json"
        assert runner_json.exists()
        data = json.loads(runner_json.read_text())
        assert "max_batch_size" in data

    def test_labels_in_bento_yaml(self, tmp_path, svc_config) -> None:
        import yaml
        packager = BentoPackager()
        bento_path = packager.package(svc_config, tmp_path / "bento")
        data = yaml.safe_load(bento_path.read_text())
        assert data["labels"]["team"] == "ml"

    def test_returns_path_to_bento_yaml(self, tmp_path, svc_config) -> None:
        packager = BentoPackager()
        result = packager.package(svc_config, tmp_path / "bento")
        assert isinstance(result, Path)
        assert result.suffix == ".yaml"
