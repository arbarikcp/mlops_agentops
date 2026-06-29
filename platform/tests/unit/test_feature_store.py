"""Tests for features/feature_store.py — DataSource, OfflineStore, OnlineStore, FeatureStore."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from features.feature_store import (
    DataSource,
    DataSourceType,
    FeatureRegistry,
    FeatureStore,
    FeatureStoreConfig,
    InMemoryOnlineStore,
    MaterializationRecord,
    OfflineStore,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_feature_parquet(tmp_path: Path) -> Path:
    """Write a small feature Parquet file for tests."""
    df = pd.DataFrame({
        "entity_id": ["c1", "c1", "c2", "c2"],
        "event_timestamp": [
            "2023-01-01T00:00:00+00:00",
            "2023-06-01T00:00:00+00:00",
            "2023-01-01T00:00:00+00:00",
            "2023-09-01T00:00:00+00:00",
        ],
        "pay_ratio": [0.10, 0.25, 0.50, 0.80],
        "util_rate": [0.30, 0.40, 0.70, 0.90],
    })
    path = tmp_path / "features.parquet"
    df.to_parquet(path, index=False)
    return path


# ── DataSource ─────────────────────────────────────────────────────────────────

class TestDataSource:
    def test_basic_creation(self) -> None:
        ds = DataSource("test", DataSourceType.PARQUET, path="/tmp/f.parquet")
        assert ds.name == "test"
        assert ds.source_type == DataSourceType.PARQUET

    def test_validate_parquet_requires_path(self) -> None:
        ds = DataSource("x", DataSourceType.PARQUET, path="")
        with pytest.raises(ValueError, match="requires a path"):
            ds.validate()

    def test_validate_push_no_path_required(self) -> None:
        ds = DataSource("stream", DataSourceType.PUSH)
        ds.validate()  # should not raise

    def test_default_fields(self) -> None:
        ds = DataSource("x", DataSourceType.CSV, path="f.csv")
        assert ds.timestamp_field == "event_timestamp"
        assert ds.entity_field == "entity_id"


# ── OfflineStore ───────────────────────────────────────────────────────────────

class TestOfflineStore:
    def test_read_parquet(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = OfflineStore(str(tmp_path))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))
        df = store.read(ds)
        assert len(df) == 4
        assert "pay_ratio" in df.columns

    def test_read_returns_empty_if_missing(self, tmp_path: Path) -> None:
        store = OfflineStore(str(tmp_path))
        ds = DataSource("missing", DataSourceType.PARQUET, path=str(tmp_path / "no.parquet"))
        df = store.read(ds)
        assert df.empty

    def test_read_filters_by_start(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = OfflineStore(str(tmp_path))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))
        start = datetime(2023, 5, 1, tzinfo=timezone.utc)
        df = store.read(ds, start_dt=start)
        # only rows after May 1
        assert len(df) == 2

    def test_read_filters_by_end(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = OfflineStore(str(tmp_path))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))
        end = datetime(2023, 3, 1, tzinfo=timezone.utc)
        df = store.read(ds, end_dt=end)
        assert len(df) == 2  # both Jan rows

    def test_pit_join_uses_most_recent_before_event(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = OfflineStore(str(tmp_path))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))

        entity_df = pd.DataFrame({
            "entity_id": ["c1"],
            "event_timestamp": ["2023-03-01T00:00:00+00:00"],
        })
        result = store.get_historical_features(entity_df, ds, ["pay_ratio", "util_rate"])
        # c1 has Jan feature (0.10), Jun feature is after March → should get Jan
        assert result.iloc[0]["pay_ratio"] == pytest.approx(0.10)

    def test_pit_join_gets_latest_available(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = OfflineStore(str(tmp_path))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))

        entity_df = pd.DataFrame({
            "entity_id": ["c1"],
            "event_timestamp": ["2023-12-01T00:00:00+00:00"],
        })
        result = store.get_historical_features(entity_df, ds, ["pay_ratio"])
        # Jun snapshot is available by Dec
        assert result.iloc[0]["pay_ratio"] == pytest.approx(0.25)

    def test_pit_join_returns_none_when_no_data_before_event(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = OfflineStore(str(tmp_path))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))

        entity_df = pd.DataFrame({
            "entity_id": ["c1"],
            "event_timestamp": ["2022-01-01T00:00:00+00:00"],  # before all data
        })
        result = store.get_historical_features(entity_df, ds, ["pay_ratio"])
        assert result.iloc[0]["pay_ratio"] is None


# ── InMemoryOnlineStore ────────────────────────────────────────────────────────

class TestInMemoryOnlineStore:
    def test_put_and_get(self) -> None:
        store = InMemoryOnlineStore()
        store.put("c1", {"pay_ratio": 0.25, "util_rate": 0.40})
        result = store.get("c1", ["pay_ratio", "util_rate"])
        assert result["pay_ratio"] == pytest.approx(0.25)
        assert result["util_rate"] == pytest.approx(0.40)

    def test_missing_key_returns_none(self) -> None:
        store = InMemoryOnlineStore()
        result = store.get("unknown", ["pay_ratio"])
        assert result["pay_ratio"] is None

    def test_missing_feature_returns_none(self) -> None:
        store = InMemoryOnlineStore()
        store.put("c1", {"pay_ratio": 0.25})
        result = store.get("c1", ["pay_ratio", "util_rate"])
        assert result["util_rate"] is None

    def test_ttl_expiry(self) -> None:
        store = InMemoryOnlineStore()
        store.put("c1", {"pay_ratio": 0.25}, ttl_seconds=1)
        time.sleep(1.1)
        result = store.get("c1", ["pay_ratio"])
        assert result["pay_ratio"] is None

    def test_delete(self) -> None:
        store = InMemoryOnlineStore()
        store.put("c1", {"pay_ratio": 0.25})
        store.delete("c1")
        result = store.get("c1", ["pay_ratio"])
        assert result["pay_ratio"] is None

    def test_len(self) -> None:
        store = InMemoryOnlineStore()
        store.put("c1", {"f": 1})
        store.put("c2", {"f": 2})
        assert len(store) == 2


# ── FeatureRegistry ────────────────────────────────────────────────────────────

class TestFeatureRegistry:
    def test_register_and_get(self) -> None:
        reg = FeatureRegistry()
        reg.register("credit_features", {"entity": "customer_id", "features": ["pay_ratio"]})
        meta = reg.get("credit_features")
        assert meta["entity"] == "customer_id"

    def test_get_unknown_raises(self) -> None:
        reg = FeatureRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_list_views(self) -> None:
        reg = FeatureRegistry()
        reg.register("view_a", {})
        reg.register("view_b", {})
        assert set(reg.list_views()) == {"view_a", "view_b"}

    def test_record_and_last_materialized(self) -> None:
        reg = FeatureRegistry()
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 6, 1, tzinfo=timezone.utc)
        reg.record_materialization("credit_features", start, end, row_count=500)
        last = reg.last_materialized_at("credit_features")
        assert last is not None
        assert last.year == 2023

    def test_last_materialized_none_if_never_run(self) -> None:
        reg = FeatureRegistry()
        assert reg.last_materialized_at("credit_features") is None

    def test_persist_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        reg = FeatureRegistry(str(path))
        reg.register("v1", {"features": ["x"]})
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 6, 1, tzinfo=timezone.utc)
        reg.record_materialization("v1", start, end, 100)

        reg2 = FeatureRegistry(str(path))
        assert "v1" in reg2.list_views()
        assert reg2.last_materialized_at("v1") is not None


# ── FeatureStoreConfig ─────────────────────────────────────────────────────────

class TestFeatureStoreConfig:
    def test_defaults(self) -> None:
        cfg = FeatureStoreConfig()
        assert cfg.project == "credit_risk"
        assert cfg.online_ttl_seconds == 86400

    def test_empty_project_raises(self) -> None:
        with pytest.raises(ValueError, match="project"):
            FeatureStoreConfig(project="")

    def test_negative_ttl_raises(self) -> None:
        with pytest.raises(ValueError, match="online_ttl_seconds"):
            FeatureStoreConfig(online_ttl_seconds=-1)


# ── FeatureStore ───────────────────────────────────────────────────────────────

class TestFeatureStore:
    def test_project_name(self) -> None:
        store = FeatureStore(FeatureStoreConfig(project="my_project"))
        assert store.project_name() == "my_project"

    def test_register_source(self, tmp_path: Path) -> None:
        store = FeatureStore(FeatureStoreConfig(offline_store_path=str(tmp_path)))
        ds = DataSource("s1", DataSourceType.PARQUET, path=str(tmp_path / "f.parquet"))
        store.register_source(ds)
        meta = store.registry.get("source:s1")
        assert meta["name"] == "s1"

    def test_get_online_features_empty_store(self) -> None:
        store = FeatureStore()
        results = store.get_online_features([{"entity_id": "c1"}], ["pay_ratio"])
        assert results[0]["pay_ratio"] is None

    def test_materialize_and_serve(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        cfg = FeatureStoreConfig(offline_store_path=str(tmp_path))
        store = FeatureStore(cfg)
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))

        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 12, 31, tzinfo=timezone.utc)
        count = store.materialize(ds, ["pay_ratio", "util_rate"], start, end)
        assert count == 2  # 2 unique entities

        results = store.get_online_features([{"entity_id": "c1"}], ["pay_ratio"])
        assert results[0]["pay_ratio"] is not None

    def test_materialize_records_in_registry(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = FeatureStore(FeatureStoreConfig(offline_store_path=str(tmp_path)))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 12, 31, tzinfo=timezone.utc)
        store.materialize(ds, ["pay_ratio"], start, end, feature_view_name="credit_view")
        last = store.registry.last_materialized_at("credit_view")
        assert last is not None

    def test_historical_features_no_leakage(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = FeatureStore(FeatureStoreConfig(offline_store_path=str(tmp_path)))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))

        entity_df = pd.DataFrame({
            "entity_id": ["c1"],
            "event_timestamp": ["2023-03-01T00:00:00+00:00"],
        })
        result = store.get_historical_features(entity_df, ds, ["pay_ratio"])
        # Jan value (0.10) not Jun (0.25) — no future leakage
        assert result.iloc[0]["pay_ratio"] == pytest.approx(0.10)

    def test_get_online_multiple_entities(self, tmp_path: Path) -> None:
        parquet_path = _make_feature_parquet(tmp_path)
        store = FeatureStore(FeatureStoreConfig(offline_store_path=str(tmp_path)))
        ds = DataSource("feats", DataSourceType.PARQUET, path=str(parquet_path))
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = datetime(2023, 12, 31, tzinfo=timezone.utc)
        store.materialize(ds, ["pay_ratio"], start, end)

        results = store.get_online_features(
            [{"entity_id": "c1"}, {"entity_id": "c2"}], ["pay_ratio"]
        )
        assert len(results) == 2
        assert all(r["pay_ratio"] is not None for r in results)
