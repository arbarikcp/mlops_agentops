"""Feature Views, Entities, Features, FeatureService, and PointInTimeJoin.

Implements Feast-style feature view concepts in pure Python/pandas.
No external feature-store library required.

See: docs/phase6/day40_feature_views.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ── Entity ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entity:
    """The primary key domain of a set of features.

    Attributes:
        name:       Human-readable entity name (e.g. "customer").
        join_key:   Column name used for joins (e.g. "customer_id").
        description: What this entity represents.
        value_type: Python type name for the join key ("str", "int").
    """

    name: str
    join_key: str
    description: str = ""
    value_type: str = "str"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Entity.name must not be empty")
        if not self.join_key:
            raise ValueError("Entity.join_key must not be empty")


# ── Feature ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Feature:
    """A single named feature with its type and constraints.

    Attributes:
        name:        Column name in the feature DataFrame.
        dtype:       Python/pandas type ("float", "int", "bool", "str").
        description: What this feature measures.
        nullable:    Whether None values are allowed at serving time.
    """

    name: str
    dtype: str = "float"
    description: str = ""
    nullable: bool = True

    def __post_init__(self) -> None:
        allowed = {"float", "int", "bool", "str", "timestamp"}
        if self.dtype not in allowed:
            raise ValueError(f"Feature '{self.name}' has unsupported dtype '{self.dtype}'. Allowed: {allowed}")


# ── FeatureView ────────────────────────────────────────────────────────────────

@dataclass
class FeatureView:
    """Groups a set of features sharing an entity and a data source.

    Attributes:
        name:     Unique identifier for this feature view.
        entities: List of Entity objects this view belongs to.
        features: List of Feature objects included in this view.
        source:   DataSource reference (name, not object, to avoid circular imports).
        ttl_days: How many days a materialised value is considered fresh (0 = no TTL).
        tags:     Arbitrary key-value metadata (owner, team, domain).
    """

    name: str
    entities: list[Entity]
    features: list[Feature]
    source: str = ""
    ttl_days: int = 1
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureView.name must not be empty")
        if not self.entities:
            raise ValueError(f"FeatureView '{self.name}' must have at least one entity")
        if not self.features:
            raise ValueError(f"FeatureView '{self.name}' must have at least one feature")
        if self.ttl_days < 0:
            raise ValueError(f"FeatureView '{self.name}' ttl_days must be >= 0")
        # Ensure feature names are unique within the view
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError(f"FeatureView '{self.name}' has duplicate feature names")

    def feature_names(self) -> list[str]:
        """Return a list of all feature column names."""
        return [f.name for f in self.features]

    def join_keys(self) -> list[str]:
        """Return the join key(s) for all entities."""
        return [e.join_key for e in self.entities]

    def schema(self) -> dict[str, str]:
        """Return a dict of feature_name → dtype."""
        return {f.name: f.dtype for f in self.features}

    def ttl_seconds(self) -> int:
        """Convert ttl_days to seconds."""
        return self.ttl_days * 86400


# ── FeatureService ─────────────────────────────────────────────────────────────

@dataclass
class FeatureService:
    """Groups multiple feature views for one model or serving use-case.

    Attributes:
        name:          Service identifier (e.g. "credit_risk_v1").
        feature_views: Feature views included in this service.
        tags:          Arbitrary key-value metadata.
    """

    name: str
    feature_views: list[FeatureView]
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FeatureService.name must not be empty")
        if not self.feature_views:
            raise ValueError(f"FeatureService '{self.name}' must include at least one feature view")

    def all_feature_names(self) -> list[str]:
        """Return all feature names across all feature views (no duplicates)."""
        seen: set[str] = set()
        result: list[str] = []
        for fv in self.feature_views:
            for name in fv.feature_names():
                if name not in seen:
                    seen.add(name)
                    result.append(name)
        return result

    def entities(self) -> list[Entity]:
        """Return all unique entities across all feature views."""
        seen: set[str] = set()
        result: list[Entity] = []
        for fv in self.feature_views:
            for ent in fv.entities:
                if ent.join_key not in seen:
                    seen.add(ent.join_key)
                    result.append(ent)
        return result

    def join_keys(self) -> list[str]:
        return [e.join_key for e in self.entities()]


# ── PointInTimeJoin ────────────────────────────────────────────────────────────

class PointInTimeJoin:
    """Performs a point-in-time correct join between entity rows and feature history.

    The join ensures that for each entity row at time T, only feature values
    from at-or-before T are used — no future data leaks into training.

    Args:
        entity_timestamp_col:  Column in entity_df with the event timestamp.
        feature_timestamp_col: Column in feature_df with the feature snapshot timestamp.
    """

    def __init__(
        self,
        entity_timestamp_col: str = "event_timestamp",
        feature_timestamp_col: str = "event_timestamp",
    ) -> None:
        self.entity_ts = entity_timestamp_col
        self.feature_ts = feature_timestamp_col

    def join(
        self,
        entity_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        join_key: str,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """Join entity_df with feature_df using point-in-time semantics.

        Args:
            entity_df:    DataFrame with join_key + entity_timestamp_col columns.
            feature_df:   DataFrame with join_key + feature_timestamp_col + feature_cols.
            join_key:     Column name used for entity matching.
            feature_cols: Which columns from feature_df to include in the result.

        Returns:
            entity_df with feature_cols appended; None where no historical data exists.
        """
        if entity_df.empty:
            for col in feature_cols:
                entity_df = entity_df.copy()
                entity_df[col] = None
            return entity_df

        # Normalise timestamps to UTC-aware
        entity_df = entity_df.copy()
        entity_df[self.entity_ts] = pd.to_datetime(entity_df[self.entity_ts], utc=True)

        feature_df = feature_df.copy()
        feature_df[self.feature_ts] = pd.to_datetime(feature_df[self.feature_ts], utc=True)

        rows: list[dict[str, Any]] = []
        for _, erow in entity_df.iterrows():
            eid = erow[join_key]
            ets = erow[self.entity_ts]

            # as-of filter: same entity, feature snapshot at or before event time
            mask = (feature_df[join_key] == eid) & (feature_df[self.feature_ts] <= ets)
            candidates = feature_df[mask]

            merged: dict[str, Any] = erow.to_dict()
            if candidates.empty:
                for col in feature_cols:
                    merged[col] = None
            else:
                # most recent snapshot before event timestamp
                latest = candidates.sort_values(self.feature_ts).iloc[-1]
                for col in feature_cols:
                    merged[col] = latest.get(col, None)
            rows.append(merged)

        return pd.DataFrame(rows).reset_index(drop=True)


# ── Credit Risk Feature Views (canonical definitions) ─────────────────────────

CUSTOMER_ENTITY = Entity(
    name="customer",
    join_key="customer_id",
    description="Credit card account holder",
    value_type="str",
)

PAYMENT_FEATURES = FeatureView(
    name="payment_features",
    entities=[CUSTOMER_ENTITY],
    features=[
        Feature("pay_ratio", "float", "Payment amount / bill amount (last cycle)"),
        Feature("avg_pay_ratio_6m", "float", "Average pay ratio over 6 months"),
        Feature("num_late_payments", "int", "Count of late payments in 6 months"),
        Feature("max_consecutive_late", "int", "Longest streak of consecutive late months"),
    ],
    ttl_days=7,
    tags={"team": "credit", "domain": "payment"},
)

BALANCE_FEATURES = FeatureView(
    name="balance_features",
    entities=[CUSTOMER_ENTITY],
    features=[
        Feature("util_rate", "float", "Credit utilisation rate (bill / limit)"),
        Feature("avg_util_6m", "float", "Average utilisation over 6 months"),
        Feature("limit_bal", "float", "Credit limit (USD)"),
        Feature("max_bill_ratio", "float", "Max bill amount / limit over 6 months"),
    ],
    ttl_days=7,
    tags={"team": "credit", "domain": "balance"},
)

CREDIT_RISK_SERVICE = FeatureService(
    name="credit_risk_v1",
    feature_views=[PAYMENT_FEATURES, BALANCE_FEATURES],
    tags={"model": "credit_risk_lgbm", "version": "1"},
)
