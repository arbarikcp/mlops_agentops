"""Tests for serving/inference.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from serving.inference import (
    LatencyTracker,
    ModelRunner,
    PredictionResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session():
    """ORT InferenceSession mock returning probabilities for N rows."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="float_input", shape=[None, 10])]

    def _run(output_names, feed_dict):
        x = list(feed_dict.values())[0]
        n = x.shape[0]
        rng = np.random.default_rng(42)
        proba = rng.uniform(0.2, 0.8, n).astype(np.float32)
        return [np.column_stack([1 - proba, proba])]

    session.run.side_effect = _run
    return session


@pytest.fixture
def runner(mock_session, tmp_path) -> ModelRunner:
    """A ModelRunner with a mocked ONNX session already loaded."""
    p = tmp_path / "model.onnx"
    p.write_bytes(b"\x00")
    r = ModelRunner(model_path=p, model_version="v1", threshold=0.5)
    r._session = mock_session
    r._input_name = "float_input"
    return r


@pytest.fixture
def sample_features() -> dict[str, float]:
    return {f"feat_{i}": float(i) for i in range(10)}


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.uniform(0, 1, (50, 10)), columns=[f"feat_{i}" for i in range(10)])


# ── LatencyTracker ────────────────────────────────────────────────────────────

class TestLatencyTracker:
    def test_empty_percentile_returns_zero(self) -> None:
        tracker = LatencyTracker()
        assert tracker.percentile(99) == 0.0

    def test_record_and_retrieve(self) -> None:
        tracker = LatencyTracker()
        tracker.record(10.0)
        tracker.record(20.0)
        assert tracker.percentile(50) == pytest.approx(15.0, rel=0.1)

    def test_summary_keys(self) -> None:
        tracker = LatencyTracker()
        for v in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]:
            tracker.record(float(v))
        summary = tracker.summary()
        assert set(summary.keys()) == {"n", "mean", "p50", "p90", "p95", "p99"}

    def test_n_samples_count(self) -> None:
        tracker = LatencyTracker()
        for i in range(5):
            tracker.record(float(i))
        assert tracker.n_samples == 5

    def test_reset_clears_samples(self) -> None:
        tracker = LatencyTracker()
        tracker.record(10.0)
        tracker.reset()
        assert tracker.n_samples == 0
        assert tracker.percentile(50) == 0.0

    def test_max_samples_evicts_oldest(self) -> None:
        tracker = LatencyTracker(max_samples=3)
        for i in range(5):
            tracker.record(float(i))
        assert tracker.n_samples == 3

    def test_p99_higher_than_p50(self) -> None:
        tracker = LatencyTracker()
        rng = np.random.default_rng(0)
        for v in rng.uniform(1, 100, 500):
            tracker.record(float(v))
        assert tracker.percentile(99) >= tracker.percentile(50)


# ── ModelRunner.load ──────────────────────────────────────────────────────────

class TestModelRunnerLoad:
    def test_raises_without_onnxruntime(self, tmp_path) -> None:
        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")
        r = ModelRunner(model_path=p)
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime"):
                r.load()

    def test_raises_file_not_found(self, tmp_path) -> None:
        r = ModelRunner(model_path=tmp_path / "nonexistent.onnx")
        mock_ort = MagicMock()
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            with pytest.raises(FileNotFoundError):
                r.load()

    def test_load_sets_session(self, tmp_path) -> None:
        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")
        r = ModelRunner(model_path=p)

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="float_input")]
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            r.load()

        assert r._session is mock_session


# ── ModelRunner.predict_single ────────────────────────────────────────────────

class TestPredictSingle:
    def test_returns_prediction_result(self, runner, sample_features) -> None:
        result = runner.predict_single(sample_features)
        assert isinstance(result, PredictionResult)

    def test_score_in_range(self, runner, sample_features) -> None:
        result = runner.predict_single(sample_features)
        assert 0.0 <= result.score <= 1.0

    def test_label_is_binary(self, runner, sample_features) -> None:
        result = runner.predict_single(sample_features)
        assert result.label in (0, 1)

    def test_model_version_in_result(self, runner, sample_features) -> None:
        result = runner.predict_single(sample_features)
        assert result.model_version == "v1"

    def test_applicant_id_passed_through(self, runner, sample_features) -> None:
        result = runner.predict_single(sample_features, applicant_id=9999)
        assert result.applicant_id == 9999

    def test_latency_recorded(self, runner, sample_features) -> None:
        runner.predict_single(sample_features)
        assert runner.latency.n_samples == 1

    def test_raises_if_not_loaded(self, tmp_path) -> None:
        r = ModelRunner(model_path=tmp_path / "model.onnx")
        with pytest.raises(RuntimeError, match="load"):
            r.predict_single({"a": 1.0})

    def test_accepts_series_input(self, runner) -> None:
        s = pd.Series({f"feat_{i}": float(i) for i in range(10)})
        result = runner.predict_single(s)
        assert isinstance(result, PredictionResult)

    def test_accepts_ndarray_input(self, runner) -> None:
        arr = np.zeros(10, dtype=np.float32)
        result = runner.predict_single(arr)
        assert isinstance(result, PredictionResult)

    def test_threshold_controls_label(self, runner) -> None:
        # With threshold=0.0 everything should be label=1
        runner.threshold = 0.0
        result = runner.predict_single({f"feat_{i}": 0.1 for i in range(10)})
        assert result.label == 1

        # With threshold=1.0 everything should be label=0
        runner.threshold = 1.0
        result = runner.predict_single({f"feat_{i}": 0.9 for i in range(10)})
        assert result.label == 0


# ── ModelRunner.predict_batch ─────────────────────────────────────────────────

class TestPredictBatch:
    def test_returns_list(self, runner, sample_df) -> None:
        results = runner.predict_batch(sample_df)
        assert isinstance(results, list)

    def test_result_count_matches_rows(self, runner, sample_df) -> None:
        results = runner.predict_batch(sample_df)
        assert len(results) == len(sample_df)

    def test_all_results_are_prediction_result(self, runner, sample_df) -> None:
        results = runner.predict_batch(sample_df)
        assert all(isinstance(r, PredictionResult) for r in results)

    def test_raises_if_not_loaded(self, tmp_path) -> None:
        r = ModelRunner(model_path=tmp_path / "model.onnx")
        with pytest.raises(RuntimeError, match="load"):
            r.predict_batch(pd.DataFrame({"a": [1, 2]}))

    def test_chunking_produces_same_count(self, runner, sample_df) -> None:
        r1 = runner.predict_batch(sample_df, chunk_size=10)
        r2 = runner.predict_batch(sample_df, chunk_size=50)
        assert len(r1) == len(r2)

    def test_id_col_used_as_applicant_id(self, runner) -> None:
        df = pd.DataFrame({
            "id": [101, 202, 303],
            **{f"feat_{i}": [0.1, 0.2, 0.3] for i in range(10)},
        })
        results = runner.predict_batch(df, id_col="id")
        ids = [r.applicant_id for r in results]
        assert ids == [101, 202, 303]
