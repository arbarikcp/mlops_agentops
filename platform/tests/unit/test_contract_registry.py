"""Tests for data/contracts/contract_registry.py."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data.contracts.contract_registry import (
    ContractMetadata,
    ContractRegistry,
    ContractViolationError,
    DataFreshnessError,
    default_registry,
)
from data.contracts.feature_schema import feature_schema


# ── Helpers ───────────────────────────────────────────────────────────────────

N = 80


def _make_valid_feature_df(n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "ID": range(1, n + 1),
        "LIMIT_BAL": rng.choice([50_000.0, 100_000.0], n),
        "SEX": rng.choice([1, 2], n),
        "EDUCATION": rng.choice([1, 2, 3, 4], n),
        "MARRIAGE": rng.choice([1, 2, 3], n),
        "AGE": rng.integers(20, 70, n).astype(int),
        "PAY_0": rng.integers(-2, 2, n),
        "PAY_2": rng.integers(-2, 2, n),
        "PAY_3": rng.integers(-2, 2, n),
        "PAY_4": rng.integers(-2, 2, n),
        "PAY_5": rng.integers(-2, 2, n),
        "PAY_6": rng.integers(-2, 2, n),
        **{f"BILL_AMT{i}": rng.uniform(0, 50_000, n) for i in range(1, 7)},
        **{f"PAY_AMT{i}": rng.uniform(0, 10_000, n) for i in range(1, 7)},
        "DEFAULT_PAYMENT_NEXT_MONTH": rng.choice([0, 1], n),
        "utilization_rate": rng.uniform(0.0, 0.9, n),
        "payment_ratio": rng.uniform(0.0, 2.0, n),
        "max_delay": rng.uniform(-1.0, 2.0, n),
        "avg_delay": rng.uniform(-1.0, 1.5, n),
        "consecutive_delays": rng.integers(0, 4, n).astype(float),
        "bill_trend": rng.uniform(-3.0, 3.0, n),
        "total_payment_ratio": rng.uniform(0.0, 2.0, n),
    })


def _make_minimal_contract(mode: str = "strict") -> ContractMetadata:
    return ContractMetadata(
        name="test_contract",
        version="1.0",
        owner="test@example.com",
        description="Test contract",
        enforcement_mode=mode,
        schema=feature_schema,
    )


# ── ContractMetadata tests ────────────────────────────────────────────────────

class TestContractMetadata:
    def test_valid_construction(self) -> None:
        meta = _make_minimal_contract()
        assert meta.name == "test_contract"
        assert meta.version == "1.0"
        assert meta.enforcement_mode == "strict"

    def test_full_name_property(self) -> None:
        meta = _make_minimal_contract()
        assert meta.full_name == "test_contract@1.0"

    def test_invalid_enforcement_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="enforcement_mode must be"):
            ContractMetadata(
                name="bad",
                version="1.0",
                owner="x@example.com",
                description="",
                enforcement_mode="invalid_mode",
                schema=feature_schema,
            )

    def test_frozen_immutable(self) -> None:
        meta = _make_minimal_contract()
        with pytest.raises((AttributeError, TypeError)):
            meta.owner = "other@example.com"  # type: ignore[misc]


# ── ContractRegistry tests ────────────────────────────────────────────────────

class TestContractRegistry:
    def test_register_and_get(self) -> None:
        registry = ContractRegistry()
        meta = _make_minimal_contract()
        registry.register(meta)
        assert registry.get("test_contract") is meta

    def test_get_unknown_raises_key_error(self) -> None:
        registry = ContractRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_contracts(self) -> None:
        registry = ContractRegistry()
        registry.register(_make_minimal_contract())
        assert "test_contract" in registry.list_contracts()

    def test_duplicate_same_version_ok(self) -> None:
        registry = ContractRegistry()
        meta = _make_minimal_contract()
        registry.register(meta)
        registry.register(meta)  # same object, same version — should not raise

    def test_duplicate_different_version_raises(self) -> None:
        registry = ContractRegistry()
        registry.register(_make_minimal_contract())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ContractMetadata(
                name="test_contract",
                version="2.0",  # different version
                owner="x",
                description="",
                enforcement_mode="strict",
                schema=feature_schema,
            ))

    def test_validate_valid_df_passes(self) -> None:
        registry = ContractRegistry()
        registry.register(_make_minimal_contract())
        df = _make_valid_feature_df()
        result = registry.validate("test_contract", df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == N

    def test_validate_strict_bad_df_raises(self) -> None:
        registry = ContractRegistry()
        registry.register(_make_minimal_contract("strict"))
        bad_df = _make_valid_feature_df()
        bad_df.loc[0, "EDUCATION"] = 0  # invalid
        with pytest.raises(ContractViolationError):
            registry.validate("test_contract", bad_df)

    def test_validate_warn_mode_does_not_raise(self) -> None:
        registry = ContractRegistry()
        registry.register(_make_minimal_contract("warn"))
        bad_df = _make_valid_feature_df()
        bad_df.loc[0, "EDUCATION"] = 0
        # Should return without raising
        result = registry.validate("test_contract", bad_df)
        assert result is not None

    def test_validate_log_only_mode_does_not_raise(self) -> None:
        registry = ContractRegistry()
        registry.register(_make_minimal_contract("log_only"))
        bad_df = _make_valid_feature_df()
        bad_df.loc[0, "EDUCATION"] = 0
        result = registry.validate("test_contract", bad_df)
        assert result is not None

    def test_override_mode_strict_to_warn(self) -> None:
        registry = ContractRegistry()
        registry.register(_make_minimal_contract("strict"))
        bad_df = _make_valid_feature_df()
        bad_df.loc[0, "EDUCATION"] = 0
        # Override to warn — should not raise even though contract is strict
        result = registry.validate("test_contract", bad_df, override_mode="warn")
        assert result is not None


# ── Freshness tests ───────────────────────────────────────────────────────────

class TestFreshnessChecks:
    def _fresh_contract(self, mode: str = "strict") -> ContractMetadata:
        return ContractMetadata(
            name="fresh_test",
            version="1.0",
            owner="x",
            description="",
            enforcement_mode=mode,
            schema=feature_schema,
            max_age_hours=12.0,
        )

    def test_fresh_data_passes(self) -> None:
        registry = ContractRegistry()
        registry.register(self._fresh_contract())
        df = _make_valid_feature_df()
        recent = datetime.now(timezone.utc) - timedelta(hours=1)
        result = registry.validate("fresh_test", df, created_at=recent)
        assert isinstance(result, pd.DataFrame)

    def test_stale_data_strict_raises(self) -> None:
        registry = ContractRegistry()
        registry.register(self._fresh_contract("strict"))
        df = _make_valid_feature_df()
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        with pytest.raises(DataFreshnessError):
            registry.validate("fresh_test", df, created_at=old)

    def test_stale_data_warn_does_not_raise(self) -> None:
        registry = ContractRegistry()
        registry.register(self._fresh_contract("warn"))
        df = _make_valid_feature_df()
        old = datetime.now(timezone.utc) - timedelta(hours=25)
        result = registry.validate("fresh_test", df, created_at=old)
        assert result is not None

    def test_no_created_at_skips_freshness(self) -> None:
        registry = ContractRegistry()
        registry.register(self._fresh_contract("strict"))
        df = _make_valid_feature_df()
        # No created_at → freshness check is skipped entirely
        result = registry.validate("fresh_test", df, created_at=None)
        assert isinstance(result, pd.DataFrame)


# ── Default registry tests ────────────────────────────────────────────────────

class TestDefaultRegistry:
    def test_credit_raw_v1_registered(self) -> None:
        assert "credit_raw_v1" in default_registry.list_contracts()

    def test_credit_feature_v1_registered(self) -> None:
        assert "credit_feature_v1" in default_registry.list_contracts()

    def test_feature_contract_validates_valid_df(self) -> None:
        df = _make_valid_feature_df()
        result = default_registry.validate("credit_feature_v1", df)
        assert len(result) == N
