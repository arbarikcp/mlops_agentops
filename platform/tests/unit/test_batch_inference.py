"""Tests for serving/batch_inference.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from serving.batch_inference import (
    BatchInferenceJob,
    BatchJobManifest,
    BatchJobResult,
    ManifestStore,
    plan_backfill,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def manifest_store(tmp_path) -> ManifestStore:
    return ManifestStore(root_dir=tmp_path / "manifests")


@pytest.fixture
def features_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.uniform(0, 1, (50, 5)), columns=[f"feat_{i}" for i in range(5)])


@pytest.fixture
def predict_fn():
    """Returns uniform probabilities based on row index."""
    def _predict(df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), 0.65)
    return _predict


@pytest.fixture
def job(manifest_store) -> BatchInferenceJob:
    return BatchInferenceJob(
        job_id="test_job_2024-01",
        model_version="v1.0",
        manifest_store=manifest_store,
        data_partition="2024-01",
    )


# ── ManifestStore ─────────────────────────────────────────────────────────────

class TestManifestStore:
    def _make_manifest(self, job_id: str = "job_1") -> BatchJobManifest:
        return BatchJobManifest(
            job_id=job_id,
            model_version="v1",
            data_partition="2024-01",
            n_rows_scored=100,
            output_path="/tmp/scores.csv",
            output_checksum="abc123",
            status="completed",
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:01:00Z",
        )

    def test_write_creates_file(self, manifest_store) -> None:
        m = self._make_manifest()
        manifest_store.write(m)
        assert (manifest_store.root_dir / "job_1.json").exists()

    def test_read_returns_manifest(self, manifest_store) -> None:
        m = self._make_manifest()
        manifest_store.write(m)
        loaded = manifest_store.read("job_1")
        assert loaded is not None
        assert loaded.job_id == "job_1"
        assert loaded.n_rows_scored == 100

    def test_read_returns_none_if_missing(self, manifest_store) -> None:
        assert manifest_store.read("nonexistent") is None

    def test_is_complete_true_for_completed(self, manifest_store) -> None:
        manifest_store.write(self._make_manifest("job_x"))
        assert manifest_store.is_complete("job_x") is True

    def test_is_complete_false_if_missing(self, manifest_store) -> None:
        assert manifest_store.is_complete("no_such_job") is False

    def test_is_complete_false_for_failed(self, manifest_store) -> None:
        m = self._make_manifest("failed_job")
        m = BatchJobManifest(**{**m.to_dict(), "status": "failed"})
        manifest_store.write(m)
        assert manifest_store.is_complete("failed_job") is False

    def test_list_all_returns_all(self, manifest_store) -> None:
        manifest_store.write(self._make_manifest("j1"))
        manifest_store.write(self._make_manifest("j2"))
        manifests = manifest_store.list_all()
        assert len(manifests) == 2

    def test_roundtrip_preserves_fields(self, manifest_store) -> None:
        m = self._make_manifest()
        manifest_store.write(m)
        loaded = manifest_store.read("job_1")
        assert loaded.output_checksum == "abc123"
        assert loaded.model_version == "v1"


# ── BatchInferenceJob ─────────────────────────────────────────────────────────

class TestBatchInferenceJob:
    def test_run_returns_batch_job_result(self, job, features_df, predict_fn, tmp_path) -> None:
        result = job.run(features_df, tmp_path / "out.csv", predict_fn)
        assert isinstance(result, BatchJobResult)

    def test_run_writes_output_file(self, job, features_df, predict_fn, tmp_path) -> None:
        out = tmp_path / "scores" / "out.csv"
        job.run(features_df, out, predict_fn)
        assert out.exists()

    def test_run_scores_all_rows(self, job, features_df, predict_fn, tmp_path) -> None:
        result = job.run(features_df, tmp_path / "out.csv", predict_fn)
        assert result.n_rows == len(features_df)

    def test_output_has_score_column(self, job, features_df, predict_fn, tmp_path) -> None:
        out = tmp_path / "out.csv"
        job.run(features_df, out, predict_fn)
        df_out = pd.read_csv(out)
        assert "score" in df_out.columns

    def test_output_has_label_column(self, job, features_df, predict_fn, tmp_path) -> None:
        out = tmp_path / "out.csv"
        job.run(features_df, out, predict_fn)
        df_out = pd.read_csv(out)
        assert "label" in df_out.columns

    def test_manifest_written_after_run(self, job, features_df, predict_fn, tmp_path, manifest_store) -> None:
        job.run(features_df, tmp_path / "out.csv", predict_fn)
        assert manifest_store.is_complete(job.job_id)

    def test_idempotency_skips_on_second_run(self, job, features_df, predict_fn, tmp_path) -> None:
        out = tmp_path / "out.csv"
        job.run(features_df, out, predict_fn)
        result2 = job.run(features_df, out, predict_fn)
        assert result2.skipped is True
        assert result2.n_rows == 0

    def test_force_reruns_despite_manifest(self, job, features_df, predict_fn, tmp_path) -> None:
        out = tmp_path / "out.csv"
        job.run(features_df, out, predict_fn)
        result2 = job.run(features_df, out, predict_fn, force=True)
        assert result2.skipped is False
        assert result2.n_rows == len(features_df)

    def test_chunked_scoring_same_as_single(self, manifest_store, features_df, predict_fn, tmp_path) -> None:
        job_small = BatchInferenceJob(
            job_id="chunk_10", model_version="v1",
            manifest_store=manifest_store, chunk_size=10,
        )
        job_large = BatchInferenceJob(
            job_id="chunk_100", model_version="v1",
            manifest_store=manifest_store, chunk_size=100,
        )
        r1 = job_small.run(features_df, tmp_path / "s1.csv", predict_fn)
        r2 = job_large.run(features_df, tmp_path / "s2.csv", predict_fn)
        assert r1.n_rows == r2.n_rows

    def test_manifest_checksum_matches_file(self, job, features_df, predict_fn, tmp_path) -> None:
        from serving.batch_inference import _checksum_file
        out = tmp_path / "out.csv"
        result = job.run(features_df, out, predict_fn)
        actual_checksum = _checksum_file(out)
        assert result.manifest.output_checksum == actual_checksum


# ── plan_backfill ─────────────────────────────────────────────────────────────

class TestPlanBackfill:
    def test_all_pending_for_new_model(self, manifest_store) -> None:
        partitions = ["2024-01", "2024-02", "2024-03"]
        plan = plan_backfill(partitions, "v2.0", manifest_store)
        assert all(v == "pending" for v in plan.values())

    def test_completed_partitions_detected(self, manifest_store, tmp_path) -> None:
        # Mark 2024-01 as complete
        job = BatchInferenceJob(
            job_id="backfill_v2.0_2024-01", model_version="v2.0",
            manifest_store=manifest_store,
        )
        features_df = pd.DataFrame({"x": [1, 2, 3]})
        predict_fn = lambda df: np.full(len(df), 0.5)
        job.run(features_df, tmp_path / "out.csv", predict_fn)

        plan = plan_backfill(["2024-01", "2024-02"], "v2.0", manifest_store)
        assert plan["2024-01"] == "completed"
        assert plan["2024-02"] == "pending"

    def test_plan_returns_all_partitions(self, manifest_store) -> None:
        partitions = ["p1", "p2", "p3"]
        plan = plan_backfill(partitions, "v1", manifest_store)
        assert set(plan.keys()) == set(partitions)
