"""Tests for features/materialization.py — MaterializationInterval, MaterializationJob, IncrementalMaterializer."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from features.feature_store import (
    DataSource,
    DataSourceType,
    FeatureRegistry,
    InMemoryOnlineStore,
    OfflineStore,
)
from features.materialization import (
    IncrementalMaterializer,
    MaterializationInterval,
    MaterializationJob,
    MaterializationStatus,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _make_parquet(tmp_path: Path) -> tuple[Path, DataSource]:
    df = pd.DataFrame({
        "entity_id": ["c1", "c2", "c1"],
        "event_timestamp": [
            "2023-01-01T00:00:00+00:00",
            "2023-01-01T00:00:00+00:00",
            "2023-06-01T00:00:00+00:00",
        ],
        "pay_ratio": [0.10, 0.50, 0.25],
        "util_rate": [0.30, 0.70, 0.40],
    })
    p = tmp_path / "features.parquet"
    df.to_parquet(p, index=False)
    ds = DataSource("feats", DataSourceType.PARQUET, path=str(p))
    return p, ds


# ── MaterializationInterval ────────────────────────────────────────────────────

class TestMaterializationInterval:
    def test_basic(self) -> None:
        iv = MaterializationInterval(_dt("2023-01-01"), _dt("2023-01-02"), "view_a")
        assert iv.duration_hours() == pytest.approx(24.0)

    def test_start_after_end_raises(self) -> None:
        with pytest.raises(ValueError, match="before end"):
            MaterializationInterval(_dt("2023-01-02"), _dt("2023-01-01"), "view_a")

    def test_equal_start_end_raises(self) -> None:
        with pytest.raises(ValueError, match="before end"):
            MaterializationInterval(_dt("2023-01-01"), _dt("2023-01-01"), "view_a")

    def test_frozen(self) -> None:
        iv = MaterializationInterval(_dt("2023-01-01"), _dt("2023-01-02"), "view_a")
        with pytest.raises(Exception):
            iv.start_dt = _dt("2023-02-01")  # type: ignore[misc]


# ── MaterializationJob ────────────────────────────────────────────────────────

class TestMaterializationJob:
    def _job(self, status=MaterializationStatus.DONE) -> MaterializationJob:
        iv = MaterializationInterval(_dt("2023-01-01"), _dt("2023-01-02"), "v1")
        return MaterializationJob("v1", [iv], status=status, rows_written=10)

    def test_succeeded(self) -> None:
        assert self._job(MaterializationStatus.DONE).succeeded()
        assert not self._job(MaterializationStatus.FAILED).succeeded()

    def test_total_hours(self) -> None:
        iv1 = MaterializationInterval(_dt("2023-01-01"), _dt("2023-01-02"), "v1")
        iv2 = MaterializationInterval(_dt("2023-01-02"), _dt("2023-01-04"), "v1")
        job = MaterializationJob("v1", [iv1, iv2])
        assert job.total_hours() == pytest.approx(72.0)

    def test_default_status_pending(self) -> None:
        job = MaterializationJob("v1", [])
        assert job.status == MaterializationStatus.PENDING


# ── IncrementalMaterializer — planning ────────────────────────────────────────

class TestMaterializerPlan:
    def test_plan_single_interval(self) -> None:
        mat = IncrementalMaterializer(interval_hours=24)
        intervals = mat.plan("v1", _dt("2023-01-01"), _dt("2023-01-02"))
        assert len(intervals) == 1
        assert intervals[0].duration_hours() == pytest.approx(24.0)

    def test_plan_splits_into_multiple_intervals(self) -> None:
        mat = IncrementalMaterializer(interval_hours=24)
        intervals = mat.plan("v1", _dt("2023-01-01"), _dt("2023-01-04"))
        assert len(intervals) == 3

    def test_plan_returns_empty_when_since_equals_until(self) -> None:
        mat = IncrementalMaterializer(interval_hours=24)
        intervals = mat.plan("v1", _dt("2023-01-01"), _dt("2023-01-01"))
        assert intervals == []

    def test_plan_returns_empty_when_since_after_until(self) -> None:
        mat = IncrementalMaterializer(interval_hours=24)
        intervals = mat.plan("v1", _dt("2023-01-02"), _dt("2023-01-01"))
        assert intervals == []

    def test_plan_intervals_are_contiguous(self) -> None:
        mat = IncrementalMaterializer(interval_hours=24)
        intervals = mat.plan("v1", _dt("2023-01-01"), _dt("2023-01-05"))
        for i in range(len(intervals) - 1):
            assert intervals[i].end_dt == intervals[i + 1].start_dt

    def test_invalid_interval_hours_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_hours"):
            IncrementalMaterializer(interval_hours=0)

    def test_last_interval_capped_at_until(self) -> None:
        mat = IncrementalMaterializer(interval_hours=24)
        intervals = mat.plan("v1", _dt("2023-01-01"), _dt("2023-01-03T12:00:00"))
        assert intervals[-1].end_dt == _dt("2023-01-03T12:00:00")


# ── IncrementalMaterializer — run ─────────────────────────────────────────────

class TestMaterializerRun:
    def test_run_writes_entities(self, tmp_path: Path) -> None:
        _, ds = _make_parquet(tmp_path)
        offline = OfflineStore(str(tmp_path))
        online = InMemoryOnlineStore()
        registry = FeatureRegistry()
        mat = IncrementalMaterializer()

        job = mat.run("v1", ["pay_ratio"], offline, online, ds, registry,
                      since_dt=_dt("2022-01-01"), until_dt=_dt("2023-12-31"))

        assert job.succeeded()
        assert job.rows_written == 2  # c1, c2 unique entities

    def test_run_online_store_updated(self, tmp_path: Path) -> None:
        _, ds = _make_parquet(tmp_path)
        offline = OfflineStore(str(tmp_path))
        online = InMemoryOnlineStore()
        registry = FeatureRegistry()
        mat = IncrementalMaterializer()

        mat.run("v1", ["pay_ratio"], offline, online, ds, registry,
                since_dt=_dt("2022-01-01"), until_dt=_dt("2023-12-31"))

        result = online.get("c1", ["pay_ratio"])
        assert result["pay_ratio"] is not None

    def test_run_records_in_registry(self, tmp_path: Path) -> None:
        _, ds = _make_parquet(tmp_path)
        offline = OfflineStore(str(tmp_path))
        online = InMemoryOnlineStore()
        registry = FeatureRegistry()
        mat = IncrementalMaterializer()

        mat.run("v1", ["pay_ratio"], offline, online, ds, registry,
                since_dt=_dt("2022-01-01"), until_dt=_dt("2023-12-31"))

        assert registry.last_materialized_at("v1") is not None

    def test_run_uses_registry_last_materialized(self, tmp_path: Path) -> None:
        _, ds = _make_parquet(tmp_path)
        offline = OfflineStore(str(tmp_path))
        online = InMemoryOnlineStore()
        registry = FeatureRegistry()
        # Seed registry with a recent end_dt that is after all feature data
        registry.record_materialization("v1", _dt("2023-01-01"), _dt("2024-01-01"), 0)
        mat = IncrementalMaterializer()

        # since_dt not provided → uses registry → no new data → 0 rows
        job = mat.run("v1", ["pay_ratio"], offline, online, ds, registry)
        assert job.rows_written == 0  # all data before registry's last end

    def test_run_empty_source_returns_done(self, tmp_path: Path) -> None:
        (tmp_path / "empty.parquet").write_bytes(
            pd.DataFrame(columns=["entity_id", "event_timestamp", "pay_ratio"]).to_parquet()
        )
        ds = DataSource("empty", DataSourceType.PARQUET, path=str(tmp_path / "empty.parquet"))
        offline = OfflineStore(str(tmp_path))
        online = InMemoryOnlineStore()
        registry = FeatureRegistry()
        mat = IncrementalMaterializer()

        job = mat.run("v1", ["pay_ratio"], offline, online, ds, registry,
                      since_dt=_dt("2022-01-01"), until_dt=_dt("2023-12-31"))
        assert job.status == MaterializationStatus.DONE
        assert job.rows_written == 0


# ── IncrementalMaterializer — backfill ───────────────────────────────────────

class TestMaterializerBackfill:
    def test_backfill_returns_one_job_per_interval(self, tmp_path: Path) -> None:
        _, ds = _make_parquet(tmp_path)
        offline = OfflineStore(str(tmp_path))
        online = InMemoryOnlineStore()
        registry = FeatureRegistry()
        mat = IncrementalMaterializer(interval_hours=24)

        jobs = mat.backfill("v1", ["pay_ratio"], offline, online, ds, registry,
                            start_dt=_dt("2023-01-01"), end_dt=_dt("2023-01-03"))
        assert len(jobs) == 2

    def test_backfill_all_jobs_succeed(self, tmp_path: Path) -> None:
        _, ds = _make_parquet(tmp_path)
        offline = OfflineStore(str(tmp_path))
        online = InMemoryOnlineStore()
        registry = FeatureRegistry()
        mat = IncrementalMaterializer(interval_hours=24)

        jobs = mat.backfill("v1", ["pay_ratio"], offline, online, ds, registry,
                            start_dt=_dt("2023-01-01"), end_dt=_dt("2023-01-03"))
        assert all(j.succeeded() for j in jobs)

    def test_backfill_idempotent(self, tmp_path: Path) -> None:
        _, ds = _make_parquet(tmp_path)
        offline = OfflineStore(str(tmp_path))
        online1 = InMemoryOnlineStore()
        online2 = InMemoryOnlineStore()
        registry = FeatureRegistry()
        mat = IncrementalMaterializer()

        mat.backfill("v1", ["pay_ratio"], offline, online1, ds, registry,
                     _dt("2022-01-01"), _dt("2023-12-31"))
        mat.backfill("v1", ["pay_ratio"], offline, online2, ds, registry,
                     _dt("2022-01-01"), _dt("2023-12-31"))

        r1 = online1.get("c1", ["pay_ratio"])
        r2 = online2.get("c1", ["pay_ratio"])
        assert r1["pay_ratio"] == r2["pay_ratio"]
