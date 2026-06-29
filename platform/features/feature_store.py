"""Feature Store core: DataSource, OfflineStore, OnlineStore, FeatureRegistry, FeatureStore.

Implements Feast-style feature store concepts without requiring the Feast library.
Uses Parquet files as the offline store and an in-memory dict (Redis-compatible
interface) as the online store. Suitable for local dev and unit tests.

See: docs/phase6/day39_feast_architecture.md
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd


# ── Enums ─────────────────────────────────────────────────────────────────────

class DataSourceType(str, Enum):
    PARQUET = "parquet"
    CSV = "csv"
    PUSH = "push"       # streaming push events
    REQUEST = "request" # request-time features (no materialization)


# ── DataSource ────────────────────────────────────────────────────────────────

@dataclass
class DataSource:
    """Describes where feature data lives.

    Attributes:
        name:            Unique identifier for this source.
        source_type:     How to read the data.
        path:            File path or URI (for PARQUET / CSV sources).
        timestamp_field: Column name carrying the event timestamp.
        entity_field:    Column name carrying the entity join key.
    """

    name: str
    source_type: DataSourceType
    path: str = ""
    timestamp_field: str = "event_timestamp"
    entity_field: str = "entity_id"

    def validate(self) -> None:
        if self.source_type in (DataSourceType.PARQUET, DataSourceType.CSV) and not self.path:
            raise ValueError(f"DataSource '{self.name}' requires a path for {self.source_type}")


# ── Offline Store ─────────────────────────────────────────────────────────────

class OfflineStore:
    """Reads feature data from Parquet files for training.

    Handles point-in-time correctness: for each entity row,
    retrieves the most-recent feature snapshot at-or-before event_timestamp.
    """

    def __init__(self, root_path: str) -> None:
        self.root = Path(root_path)

    def read(
        self,
        source: DataSource,
        start_dt: datetime | None = None,
        end_dt: datetime | None = None,
    ) -> pd.DataFrame:
        """Read all rows from source, optionally filtered by timestamp range."""
        if source.source_type == DataSourceType.PARQUET:
            path = Path(source.path) if Path(source.path).is_absolute() else self.root / source.path
            if not path.exists():
                return pd.DataFrame()
            df = pd.read_parquet(path)
        elif source.source_type == DataSourceType.CSV:
            path = Path(source.path) if Path(source.path).is_absolute() else self.root / source.path
            if not path.exists():
                return pd.DataFrame()
            df = pd.read_csv(path)
        else:
            return pd.DataFrame()

        if source.timestamp_field in df.columns:
            df[source.timestamp_field] = pd.to_datetime(df[source.timestamp_field], utc=True)
            if start_dt:
                start = pd.Timestamp(start_dt).tz_localize("UTC") if start_dt.tzinfo is None else pd.Timestamp(start_dt)
                df = df[df[source.timestamp_field] >= start]
            if end_dt:
                end = pd.Timestamp(end_dt).tz_localize("UTC") if end_dt.tzinfo is None else pd.Timestamp(end_dt)
                df = df[df[source.timestamp_field] <= end]
        return df.reset_index(drop=True)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        source: DataSource,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        """Point-in-time join: for each entity row, retrieve features as-of event_timestamp.

        Args:
            entity_df:       DataFrame with entity_field + timestamp_field columns.
            source:          DataSource to join against.
            feature_columns: Feature columns to retrieve from the source.

        Returns:
            entity_df with feature columns appended (no future leakage).
        """
        feature_df = self.read(source)
        if feature_df.empty:
            for col in feature_columns:
                entity_df = entity_df.copy()
                entity_df[col] = None
            return entity_df

        ts_col = source.timestamp_field
        ent_col = source.entity_field

        if ts_col not in feature_df.columns or ent_col not in feature_df.columns:
            for col in feature_columns:
                entity_df = entity_df.copy()
                entity_df[col] = None
            return entity_df

        entity_ts = source.timestamp_field
        if entity_ts not in entity_df.columns:
            entity_ts = "event_timestamp"

        entity_df = entity_df.copy()
        entity_df[entity_ts] = pd.to_datetime(entity_df[entity_ts], utc=True)
        feature_df[ts_col] = pd.to_datetime(feature_df[ts_col], utc=True)

        results: list[dict[str, Any]] = []
        for _, row in entity_df.iterrows():
            eid = row[ent_col]
            evt = row[entity_ts]

            # as-of: all feature rows for this entity at or before event timestamp
            mask = (feature_df[ent_col] == eid) & (feature_df[ts_col] <= evt)
            candidates = feature_df[mask]

            merged = row.to_dict()
            if candidates.empty:
                for col in feature_columns:
                    merged[col] = None
            else:
                latest = candidates.sort_values(ts_col).iloc[-1]
                for col in feature_columns:
                    merged[col] = latest.get(col, None)
            results.append(merged)

        return pd.DataFrame(results)


# ── Online Store ──────────────────────────────────────────────────────────────

class InMemoryOnlineStore:
    """Redis-compatible online store backed by an in-memory dict.

    In production, swap for a Redis-backed implementation with the same interface.
    """

    def __init__(self) -> None:
        # {entity_key: {feature_name: (value, expire_at_epoch)}}
        self._store: dict[str, dict[str, tuple[Any, float | None]]] = {}

    def put(
        self,
        entity_key: str,
        features: dict[str, Any],
        ttl_seconds: int | None = None,
    ) -> None:
        """Write feature values for an entity.

        Args:
            entity_key:  String key (e.g. "customer_id:1001").
            features:    Dict of feature_name → value.
            ttl_seconds: Seconds until expiry; None means no expiry.
        """
        expire_at = time.time() + ttl_seconds if ttl_seconds else None
        if entity_key not in self._store:
            self._store[entity_key] = {}
        for name, value in features.items():
            self._store[entity_key][name] = (value, expire_at)

    def get(self, entity_key: str, feature_names: list[str]) -> dict[str, Any | None]:
        """Retrieve feature values for an entity.

        Returns None for missing or expired features.
        """
        result: dict[str, Any | None] = {}
        entity_data = self._store.get(entity_key, {})
        now = time.time()
        for name in feature_names:
            if name not in entity_data:
                result[name] = None
            else:
                value, expire_at = entity_data[name]
                if expire_at is not None and now > expire_at:
                    result[name] = None  # expired
                else:
                    result[name] = value
        return result

    def delete(self, entity_key: str) -> None:
        """Remove all feature values for an entity."""
        self._store.pop(entity_key, None)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)


# ── Feature Registry ──────────────────────────────────────────────────────────

@dataclass
class MaterializationRecord:
    """Records one completed materialization run for a feature view."""

    feature_view_name: str
    start_dt: str
    end_dt: str
    row_count: int
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FeatureRegistry:
    """In-process feature registry backed by a JSON file.

    Stores feature view metadata and materialization history.
    In production, back this with Postgres or a managed registry.
    """

    def __init__(self, registry_path: str = "") -> None:
        self._path = Path(registry_path) if registry_path else None
        # {name: dict} feature view metadata store
        self._views: dict[str, dict[str, Any]] = {}
        self._materialization_history: list[MaterializationRecord] = []
        if self._path and self._path.exists():
            self._load()

    # -- registration --

    def register(self, name: str, metadata: dict[str, Any]) -> None:
        """Register or update a feature view definition."""
        if name in self._views and self._views[name] != metadata:
            # allow updates but track version bump
            metadata = {**metadata, "updated_at": datetime.now(timezone.utc).isoformat()}
        self._views[name] = metadata
        self._persist()

    def get(self, name: str) -> dict[str, Any]:
        """Retrieve a feature view definition by name.

        Raises:
            KeyError: if the feature view is not registered.
        """
        if name not in self._views:
            raise KeyError(f"Feature view '{name}' not found in registry")
        return self._views[name]

    def list_views(self) -> list[str]:
        return list(self._views.keys())

    # -- materialization history --

    def record_materialization(
        self,
        feature_view_name: str,
        start_dt: datetime,
        end_dt: datetime,
        row_count: int,
    ) -> None:
        record = MaterializationRecord(
            feature_view_name=feature_view_name,
            start_dt=start_dt.isoformat(),
            end_dt=end_dt.isoformat(),
            row_count=row_count,
        )
        self._materialization_history.append(record)
        self._persist()

    def last_materialized_at(self, feature_view_name: str) -> datetime | None:
        """Return the end_dt of the most recent successful materialization."""
        relevant = [r for r in self._materialization_history if r.feature_view_name == feature_view_name]
        if not relevant:
            return None
        latest = max(relevant, key=lambda r: r.end_dt)
        return datetime.fromisoformat(latest.end_dt)

    def materialization_history(self, feature_view_name: str) -> list[MaterializationRecord]:
        return [r for r in self._materialization_history if r.feature_view_name == feature_view_name]

    # -- persistence --

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "views": self._views,
            "history": [
                {
                    "feature_view_name": r.feature_view_name,
                    "start_dt": r.start_dt,
                    "end_dt": r.end_dt,
                    "row_count": r.row_count,
                    "completed_at": r.completed_at,
                }
                for r in self._materialization_history
            ],
        }
        self._path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self._views = data.get("views", {})
        self._materialization_history = [
            MaterializationRecord(**r) for r in data.get("history", [])
        ]


# ── Feature Store Config ───────────────────────────────────────────────────────

@dataclass
class FeatureStoreConfig:
    """Top-level configuration for the feature store.

    Attributes:
        project:          Project name (namespaces all features).
        offline_store_path: Root path for offline Parquet files.
        registry_path:    Path to the registry JSON file.
        online_ttl_seconds: Default TTL for online store entries.
    """

    project: str = "credit_risk"
    offline_store_path: str = "data/features/offline"
    registry_path: str = ""
    online_ttl_seconds: int = 86400  # 24 hours

    def __post_init__(self) -> None:
        if not self.project:
            raise ValueError("FeatureStoreConfig.project must not be empty")
        if self.online_ttl_seconds <= 0:
            raise ValueError("online_ttl_seconds must be positive")


# ── Feature Store ─────────────────────────────────────────────────────────────

class FeatureStore:
    """Main entry point — coordinates offline store, online store, and registry.

    Args:
        config: FeatureStoreConfig with paths and project settings.
    """

    def __init__(self, config: FeatureStoreConfig | None = None) -> None:
        self.config = config or FeatureStoreConfig()
        self.offline_store = OfflineStore(self.config.offline_store_path)
        self.online_store = InMemoryOnlineStore()
        self.registry = FeatureRegistry(self.config.registry_path)

    # -- registration --

    def register_source(self, source: DataSource) -> None:
        """Validate and register a data source."""
        source.validate()
        self.registry.register(f"source:{source.name}", {
            "name": source.name,
            "source_type": source.source_type.value,
            "path": source.path,
            "timestamp_field": source.timestamp_field,
            "entity_field": source.entity_field,
        })

    # -- training path --

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        source: DataSource,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        """Return a training DataFrame with features joined via PIT semantics."""
        return self.offline_store.get_historical_features(entity_df, source, feature_columns)

    # -- serving path --

    def get_online_features(
        self,
        entity_rows: list[dict[str, Any]],
        feature_names: list[str],
        entity_key_field: str = "entity_id",
    ) -> list[dict[str, Any]]:
        """Retrieve latest feature values from the online store for a list of entities.

        Args:
            entity_rows:      List of dicts each containing the entity key field.
            feature_names:    Features to retrieve.
            entity_key_field: Name of the key field in each entity row.

        Returns:
            List of dicts — one per entity row — with requested feature values merged in.
        """
        results = []
        for row in entity_rows:
            key = str(row.get(entity_key_field, ""))
            features = self.online_store.get(key, feature_names)
            results.append({**row, **features})
        return results

    # -- materialization --

    def materialize(
        self,
        source: DataSource,
        feature_columns: list[str],
        start_dt: datetime,
        end_dt: datetime,
        entity_key_field: str = "entity_id",
        feature_view_name: str = "",
    ) -> int:
        """Read features from offline store and write to online store.

        Args:
            source:           DataSource to materialise from.
            feature_columns:  Which columns to write to online store.
            start_dt:         Inclusive start of the time window.
            end_dt:           Inclusive end of the time window.
            entity_key_field: Entity join key column.
            feature_view_name: Name for registry tracking.

        Returns:
            Number of entities written to the online store.
        """
        df = self.offline_store.read(source, start_dt, end_dt)
        if df.empty:
            return 0

        ts_col = source.timestamp_field
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
            # keep only the latest row per entity
            df = df.sort_values(ts_col).groupby(entity_key_field).last().reset_index()

        count = 0
        for _, row in df.iterrows():
            entity_key = str(row[entity_key_field])
            features = {col: row[col] for col in feature_columns if col in row}
            self.online_store.put(entity_key, features, ttl_seconds=self.config.online_ttl_seconds)
            count += 1

        name = feature_view_name or source.name
        self.registry.record_materialization(name, start_dt, end_dt, count)
        return count

    def project_name(self) -> str:
        return self.config.project
