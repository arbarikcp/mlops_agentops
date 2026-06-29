"""Tests for features/feature_views.py — Entity, Feature, FeatureView, FeatureService, PointInTimeJoin."""
from __future__ import annotations

import pandas as pd
import pytest

from features.feature_views import (
    BALANCE_FEATURES,
    CREDIT_RISK_SERVICE,
    CUSTOMER_ENTITY,
    PAYMENT_FEATURES,
    Entity,
    Feature,
    FeatureService,
    FeatureView,
    PointInTimeJoin,
)


# ── Entity ─────────────────────────────────────────────────────────────────────

class TestEntity:
    def test_basic(self) -> None:
        e = Entity("customer", "customer_id")
        assert e.join_key == "customer_id"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            Entity("", "customer_id")

    def test_empty_join_key_raises(self) -> None:
        with pytest.raises(ValueError, match="join_key"):
            Entity("customer", "")

    def test_frozen(self) -> None:
        e = Entity("customer", "customer_id")
        with pytest.raises(Exception):
            e.name = "other"  # type: ignore[misc]


# ── Feature ────────────────────────────────────────────────────────────────────

class TestFeature:
    def test_default_dtype(self) -> None:
        f = Feature("pay_ratio")
        assert f.dtype == "float"
        assert f.nullable is True

    def test_invalid_dtype_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported dtype"):
            Feature("x", dtype="complex128")

    def test_all_allowed_dtypes(self) -> None:
        for dtype in ("float", "int", "bool", "str", "timestamp"):
            Feature("x", dtype=dtype)  # should not raise


# ── FeatureView ────────────────────────────────────────────────────────────────

class TestFeatureView:
    def _fv(self, **kw) -> FeatureView:
        defaults = dict(
            name="v1",
            entities=[CUSTOMER_ENTITY],
            features=[Feature("f1"), Feature("f2")],
            ttl_days=1,
        )
        return FeatureView(**{**defaults, **kw})

    def test_feature_names(self) -> None:
        fv = self._fv()
        assert fv.feature_names() == ["f1", "f2"]

    def test_join_keys(self) -> None:
        fv = self._fv()
        assert fv.join_keys() == ["customer_id"]

    def test_schema(self) -> None:
        fv = self._fv()
        assert fv.schema() == {"f1": "float", "f2": "float"}

    def test_ttl_seconds(self) -> None:
        fv = self._fv(ttl_days=2)
        assert fv.ttl_seconds() == 2 * 86400

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            self._fv(name="")

    def test_no_entities_raises(self) -> None:
        with pytest.raises(ValueError, match="entity"):
            self._fv(entities=[])

    def test_no_features_raises(self) -> None:
        with pytest.raises(ValueError, match="feature"):
            self._fv(features=[])

    def test_negative_ttl_raises(self) -> None:
        with pytest.raises(ValueError, match="ttl_days"):
            self._fv(ttl_days=-1)

    def test_duplicate_feature_names_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            self._fv(features=[Feature("f1"), Feature("f1")])

    def test_tags(self) -> None:
        fv = self._fv(tags={"team": "credit"})
        assert fv.tags["team"] == "credit"


# ── FeatureService ─────────────────────────────────────────────────────────────

class TestFeatureService:
    def test_all_feature_names_no_duplicates(self) -> None:
        fv1 = FeatureView("v1", [CUSTOMER_ENTITY], [Feature("f1"), Feature("f2")])
        fv2 = FeatureView("v2", [CUSTOMER_ENTITY], [Feature("f2"), Feature("f3")])  # f2 shared
        svc = FeatureService("svc", [fv1, fv2])
        names = svc.all_feature_names()
        assert len(names) == len(set(names))  # no dupes
        assert "f1" in names and "f3" in names

    def test_entities_deduplication(self) -> None:
        fv1 = FeatureView("v1", [CUSTOMER_ENTITY], [Feature("f1")])
        fv2 = FeatureView("v2", [CUSTOMER_ENTITY], [Feature("f2")])
        svc = FeatureService("svc", [fv1, fv2])
        assert len(svc.entities()) == 1  # same entity

    def test_join_keys(self) -> None:
        svc = CREDIT_RISK_SERVICE
        assert "customer_id" in svc.join_keys()

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            FeatureService("", [PAYMENT_FEATURES])

    def test_no_views_raises(self) -> None:
        with pytest.raises(ValueError, match="feature view"):
            FeatureService("svc", [])


# ── Canonical Definitions ──────────────────────────────────────────────────────

class TestCanonicalDefinitions:
    def test_customer_entity(self) -> None:
        assert CUSTOMER_ENTITY.join_key == "customer_id"

    def test_payment_features(self) -> None:
        assert "pay_ratio" in PAYMENT_FEATURES.feature_names()
        assert PAYMENT_FEATURES.ttl_days == 7

    def test_balance_features(self) -> None:
        assert "util_rate" in BALANCE_FEATURES.feature_names()

    def test_credit_risk_service_has_all_views(self) -> None:
        views = [fv.name for fv in CREDIT_RISK_SERVICE.feature_views]
        assert "payment_features" in views
        assert "balance_features" in views

    def test_service_feature_count(self) -> None:
        # payment (4) + balance (4) = 8 unique features
        assert len(CREDIT_RISK_SERVICE.all_feature_names()) == 8


# ── PointInTimeJoin ────────────────────────────────────────────────────────────

def _entity_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "event_timestamp": [
            "2023-03-01T00:00:00+00:00",
            "2023-09-01T00:00:00+00:00",
            "2023-06-01T00:00:00+00:00",
        ],
        "label": [1, 0, 1],
    })


def _feature_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id": ["C1", "C1", "C2"],
        "event_timestamp": [
            "2023-01-01T00:00:00+00:00",
            "2023-06-01T00:00:00+00:00",
            "2023-01-01T00:00:00+00:00",
        ],
        "pay_ratio": [0.10, 0.25, 0.50],
        "util_rate": [0.30, 0.45, 0.70],
    })


class TestPointInTimeJoin:
    def test_returns_correct_features(self) -> None:
        pit = PointInTimeJoin()
        result = pit.join(_entity_df(), _feature_df(), "customer_id", ["pay_ratio"])
        # C1 at March → only Jan snap available → 0.10
        c1_mar = result[
            (result["customer_id"] == "C1") &
            (result["event_timestamp"].astype(str).str.startswith("2023-03"))
        ]
        assert c1_mar.iloc[0]["pay_ratio"] == pytest.approx(0.10)

    def test_latest_snapshot_used_when_multiple_available(self) -> None:
        pit = PointInTimeJoin()
        result = pit.join(_entity_df(), _feature_df(), "customer_id", ["pay_ratio"])
        # C1 at September → both Jan and Jun available → Jun (0.25) is most recent
        c1_sep = result[
            (result["customer_id"] == "C1") &
            (result["event_timestamp"].astype(str).str.startswith("2023-09"))
        ]
        assert c1_sep.iloc[0]["pay_ratio"] == pytest.approx(0.25)

    def test_no_future_leakage(self) -> None:
        pit = PointInTimeJoin()
        result = pit.join(_entity_df(), _feature_df(), "customer_id", ["pay_ratio"])
        # C1 at March must NOT get June features
        c1_mar = result[
            (result["customer_id"] == "C1") &
            (result["event_timestamp"].astype(str).str.startswith("2023-03"))
        ]
        assert c1_mar.iloc[0]["pay_ratio"] != pytest.approx(0.25)

    def test_none_when_no_historical_data(self) -> None:
        pit = PointInTimeJoin()
        entity_df = pd.DataFrame({
            "customer_id": ["C_NEW"],
            "event_timestamp": ["2023-06-01T00:00:00+00:00"],
        })
        result = pit.join(entity_df, _feature_df(), "customer_id", ["pay_ratio"])
        assert result.iloc[0]["pay_ratio"] is None

    def test_preserves_entity_row_count(self) -> None:
        pit = PointInTimeJoin()
        result = pit.join(_entity_df(), _feature_df(), "customer_id", ["pay_ratio"])
        assert len(result) == len(_entity_df())

    def test_all_feature_cols_added(self) -> None:
        pit = PointInTimeJoin()
        result = pit.join(_entity_df(), _feature_df(), "customer_id", ["pay_ratio", "util_rate"])
        assert "pay_ratio" in result.columns
        assert "util_rate" in result.columns

    def test_empty_entity_df(self) -> None:
        pit = PointInTimeJoin()
        empty = pd.DataFrame(columns=["customer_id", "event_timestamp"])
        result = pit.join(empty, _feature_df(), "customer_id", ["pay_ratio"])
        assert result.empty or "pay_ratio" in result.columns

    def test_custom_timestamp_columns(self) -> None:
        pit = PointInTimeJoin(
            entity_timestamp_col="application_date",
            feature_timestamp_col="snapshot_date",
        )
        entity_df = pd.DataFrame({
            "customer_id": ["C1"],
            "application_date": ["2023-03-01T00:00:00+00:00"],
        })
        feature_df = pd.DataFrame({
            "customer_id": ["C1"],
            "snapshot_date": ["2023-01-01T00:00:00+00:00"],
            "pay_ratio": [0.10],
        })
        result = pit.join(entity_df, feature_df, "customer_id", ["pay_ratio"])
        assert result.iloc[0]["pay_ratio"] == pytest.approx(0.10)
