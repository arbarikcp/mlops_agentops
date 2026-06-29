"""Materialization: copy features from offline store to online store for low-latency serving.

Classes:
  MaterializationStatus    — PENDING / RUNNING / DONE / FAILED
  MaterializationInterval  — bounded time window for one materialization batch
  MaterializationJob       — result of running one materialization
  IncrementalMaterializer  — plans and executes incremental + backfill jobs

See: docs/phase6/day41_materialization.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


# ── Status ─────────────────────────────────────────────────────────────────────

class MaterializationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ── Interval ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MaterializationInterval:
    """A bounded time window for one materialization batch.

    Attributes:
        start_dt:          Inclusive start (UTC).
        end_dt:            Exclusive end (UTC).
        feature_view_name: Which feature view this interval belongs to.
    """

    start_dt: datetime
    end_dt: datetime
    feature_view_name: str

    def __post_init__(self) -> None:
        if self.start_dt >= self.end_dt:
            raise ValueError(
                f"MaterializationInterval start ({self.start_dt}) must be before end ({self.end_dt})"
            )

    def duration_hours(self) -> float:
        return (self.end_dt - self.start_dt).total_seconds() / 3600


# ── Job ───────────────────────────────────────────────────────────────────────

@dataclass
class MaterializationJob:
    """Result of executing a materialization run.

    Attributes:
        feature_view_name: Which feature view was materialised.
        intervals:         The time intervals covered.
        status:            Final execution status.
        rows_written:      How many entity rows were written to online store.
        error:             Error message if status is FAILED.
        completed_at:      Wall-clock completion time (UTC).
    """

    feature_view_name: str
    intervals: list[MaterializationInterval]
    status: MaterializationStatus = MaterializationStatus.PENDING
    rows_written: int = 0
    error: str = ""
    completed_at: datetime | None = None

    def succeeded(self) -> bool:
        return self.status == MaterializationStatus.DONE

    def total_hours(self) -> float:
        return sum(i.duration_hours() for i in self.intervals)


# ── Materializer ───────────────────────────────────────────────────────────────

class IncrementalMaterializer:
    """Plans and executes incremental and backfill materialization jobs.

    Splits large time windows into daily intervals for parallelism and
    partial-failure recovery. Each interval write is idempotent.

    Args:
        interval_hours: Size of each batch interval (default: 24h = daily).
        entity_key_field: Column used as the entity key in feature DataFrames.
    """

    def __init__(
        self,
        interval_hours: int = 24,
        entity_key_field: str = "entity_id",
    ) -> None:
        if interval_hours <= 0:
            raise ValueError("interval_hours must be positive")
        self.interval_hours = interval_hours
        self.entity_key_field = entity_key_field

    # -- planning --

    def plan(
        self,
        feature_view_name: str,
        since_dt: datetime,
        until_dt: datetime | None = None,
    ) -> list[MaterializationInterval]:
        """Divide [since_dt, until_dt) into equal-sized intervals.

        Args:
            feature_view_name: Name to attach to each interval.
            since_dt:          Inclusive start (last materialized end).
            until_dt:          Exclusive end (defaults to now UTC).

        Returns:
            Ordered list of non-overlapping MaterializationIntervals.
        """
        if until_dt is None:
            until_dt = datetime.now(timezone.utc)

        if since_dt >= until_dt:
            return []

        intervals: list[MaterializationInterval] = []
        current = since_dt
        step = timedelta(hours=self.interval_hours)

        while current < until_dt:
            end = min(current + step, until_dt)
            intervals.append(MaterializationInterval(
                start_dt=current,
                end_dt=end,
                feature_view_name=feature_view_name,
            ))
            current = end

        return intervals

    # -- execution --

    def run(
        self,
        feature_view_name: str,
        feature_columns: list[str],
        offline_store: Any,            # OfflineStore
        online_store: Any,             # InMemoryOnlineStore
        data_source: Any,              # DataSource
        registry: Any,                 # FeatureRegistry
        since_dt: datetime | None = None,
        until_dt: datetime | None = None,
        ttl_seconds: int = 86400,
    ) -> MaterializationJob:
        """Execute incremental materialization for a feature view.

        Reads from last_materialized_at (or since_dt) to until_dt (or now),
        writes to online_store, and records completion in registry.

        Args:
            feature_view_name: Name used for registry tracking.
            feature_columns:   Feature columns to copy to online store.
            offline_store:     OfflineStore instance to read from.
            online_store:      InMemoryOnlineStore instance to write to.
            data_source:       DataSource describing the offline data.
            registry:          FeatureRegistry for tracking history.
            since_dt:          Override for start time (ignores registry).
            until_dt:          Override for end time (defaults to now).
            ttl_seconds:       TTL for online store entries.

        Returns:
            MaterializationJob with execution result.
        """
        import pandas as pd

        if until_dt is None:
            until_dt = datetime.now(timezone.utc)

        # Use last_materialized_at from registry if since_dt not provided
        start_dt = since_dt
        if start_dt is None:
            last = registry.last_materialized_at(feature_view_name)
            start_dt = last if last else datetime(2020, 1, 1, tzinfo=timezone.utc)

        intervals = self.plan(feature_view_name, start_dt, until_dt)
        job = MaterializationJob(
            feature_view_name=feature_view_name,
            intervals=intervals,
            status=MaterializationStatus.RUNNING,
        )

        try:
            df = offline_store.read(data_source, start_dt, until_dt)
            if df.empty:
                job.status = MaterializationStatus.DONE
                job.completed_at = datetime.now(timezone.utc)
                return job

            ts_col = data_source.timestamp_field
            if ts_col in df.columns:
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
                # keep most-recent row per entity
                df = df.sort_values(ts_col).groupby(self.entity_key_field).last().reset_index()

            count = 0
            for _, row in df.iterrows():
                entity_key = str(row[self.entity_key_field])
                features = {col: row[col] for col in feature_columns if col in row}
                online_store.put(entity_key, features, ttl_seconds=ttl_seconds)
                count += 1

            registry.record_materialization(feature_view_name, start_dt, until_dt, count)
            job.rows_written = count
            job.status = MaterializationStatus.DONE
        except Exception as exc:
            job.status = MaterializationStatus.FAILED
            job.error = str(exc)
        finally:
            job.completed_at = datetime.now(timezone.utc)

        return job

    def backfill(
        self,
        feature_view_name: str,
        feature_columns: list[str],
        offline_store: Any,
        online_store: Any,
        data_source: Any,
        registry: Any,
        start_dt: datetime,
        end_dt: datetime,
        ttl_seconds: int = 86400,
    ) -> list[MaterializationJob]:
        """Run materialization across a historical range, one interval at a time.

        Useful for fixing gaps or re-materialising after a source change.
        Each interval is idempotent — safe to re-run.

        Returns:
            One MaterializationJob per interval.
        """
        intervals = self.plan(feature_view_name, start_dt, end_dt)
        jobs: list[MaterializationJob] = []
        for interval in intervals:
            job = self.run(
                feature_view_name=feature_view_name,
                feature_columns=feature_columns,
                offline_store=offline_store,
                online_store=online_store,
                data_source=data_source,
                registry=registry,
                since_dt=interval.start_dt,
                until_dt=interval.end_dt,
                ttl_seconds=ttl_seconds,
            )
            jobs.append(job)
        return jobs
