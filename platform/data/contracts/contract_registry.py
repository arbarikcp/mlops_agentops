"""Data contract registry — ownership, versioning, and enforcement modes.

The registry is the organisational layer on top of Pandera schemas.
Pandera enforces technical rules (types, ranges, nullability).
The registry enforces organisational rules:
  - who owns each contract
  - what version it is
  - whether violations stop the pipeline (strict) or emit a warning (warn)
  - how fresh the data must be

Pattern:
    registry = ContractRegistry()
    registry.register(CONTRACT_CREDIT_FEATURE_V1)
    registry.validate("credit_feature_v1", df, created_at=datetime.now(UTC))

Enforcement modes:
    strict   — raises ContractViolationError (stops the pipeline)
    warn     — logs WARNING and returns (pipeline continues)
    log_only — logs INFO only (silent in CI, for dev/exploration)

See: docs/phase3/day19_data_contracts.md for theory and invariants.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import pandera as pa

log = logging.getLogger(__name__)

EnforcementMode = str  # "strict" | "warn" | "log_only"

_VALID_MODES = {"strict", "warn", "log_only"}


class ContractViolationError(RuntimeError):
    """Raised when strict-mode contract validation fails."""


class DataFreshnessError(ContractViolationError):
    """Raised when data exceeds the max_age_hours freshness threshold."""


@dataclass(frozen=True)
class ContractMetadata:
    """Immutable contract descriptor — one instance per versioned contract.

    Attributes:
        name:             Unique contract identifier (e.g. "credit_feature_v1").
        version:          Semantic version string (e.g. "1.0").
        owner:            Responsible team / email.
        description:      What this contract covers and why it exists.
        enforcement_mode: "strict" | "warn" | "log_only".
        max_age_hours:    Freshness threshold. None = no freshness check.
        schema:           The Pandera DataFrameSchema that enforces technical rules.
        column_owners:    Optional per-column ownership map {col_name: owner}.
    """

    name: str
    version: str
    owner: str
    description: str
    enforcement_mode: EnforcementMode
    schema: Any  # pa.DataFrameSchema
    max_age_hours: float | None = None
    column_owners: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.enforcement_mode not in _VALID_MODES:
            raise ValueError(
                f"enforcement_mode must be one of {_VALID_MODES}, got {self.enforcement_mode!r}"
            )

    @property
    def full_name(self) -> str:
        return f"{self.name}@{self.version}"


class ContractRegistry:
    """Registry of versioned data contracts.

    Thread-safe for reads; not designed for concurrent writes.
    """

    def __init__(self) -> None:
        self._contracts: dict[str, ContractMetadata] = {}

    def register(self, contract: ContractMetadata) -> None:
        """Add a contract to the registry. Raises ValueError on name collision."""
        if contract.name in self._contracts:
            existing = self._contracts[contract.name]
            if existing.version != contract.version:
                raise ValueError(
                    f"Contract {contract.name!r} already registered at version "
                    f"{existing.version}. Register under a new name to version up."
                )
        self._contracts[contract.name] = contract
        log.info(
            "Registered contract %r [%s] owner=%s mode=%s",
            contract.name,
            contract.version,
            contract.owner,
            contract.enforcement_mode,
        )

    def get(self, name: str) -> ContractMetadata:
        if name not in self._contracts:
            raise KeyError(f"No contract registered with name {name!r}. Available: {list(self._contracts)}")
        return self._contracts[name]

    def list_contracts(self) -> list[str]:
        return sorted(self._contracts)

    def validate(
        self,
        name: str,
        df: pd.DataFrame,
        *,
        created_at: datetime | None = None,
        override_mode: EnforcementMode | None = None,
    ) -> pd.DataFrame:
        """Validate df against the named contract.

        Args:
            name:          Contract name (registered key).
            df:            DataFrame to validate.
            created_at:    When the data was produced (UTC). Used for freshness check.
            override_mode: If set, temporarily overrides the contract's enforcement_mode.

        Returns:
            The validated DataFrame (possibly with coerced dtypes).

        Raises:
            KeyError:               Contract name not registered.
            DataFreshnessError:     Data exceeds max_age_hours (strict mode only).
            ContractViolationError: Schema validation failed (strict mode only).
        """
        contract = self.get(name)
        mode = override_mode if override_mode is not None else contract.enforcement_mode

        # ── Freshness check ──────────────────────────────────────────────────
        if contract.max_age_hours is not None and created_at is not None:
            now = datetime.now(timezone.utc)
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            age_hours = (now - created_at).total_seconds() / 3600
            if age_hours > contract.max_age_hours:
                msg = (
                    f"Contract {name!r}: data is {age_hours:.1f}h old, "
                    f"max allowed is {contract.max_age_hours}h"
                )
                if mode == "strict":
                    raise DataFreshnessError(msg)
                else:
                    log.warning("Freshness check FAILED: %s", msg)

        # ── Schema validation ────────────────────────────────────────────────
        try:
            validated = contract.schema.validate(df, lazy=True)
            log.info(
                "Contract %r [%s] PASSED — %d rows validated",
                contract.name,
                contract.version,
                len(df),
            )
            return validated

        except pa.errors.SchemaErrors as exc:
            n_failures = len(exc.failure_cases)
            msg = (
                f"Contract {name!r} [{contract.version}] FAILED — "
                f"{n_failures} violation(s). Owner: {contract.owner}. "
                f"First failure: {exc.failure_cases.head(1).to_dict(orient='records')}"
            )
            if mode == "strict":
                raise ContractViolationError(msg) from exc
            elif mode == "warn":
                log.warning("Contract violation (warn mode — pipeline continues): %s", msg)
                return df
            else:  # log_only
                log.info("Contract violation (log_only): %s", msg)
                return df


# ── Built-in contracts ───────────────────────────────────────────────────────

def _build_default_registry() -> ContractRegistry:
    """Build the project-default registry with known contracts pre-registered."""
    from data.contracts.raw_schema import raw_schema
    from data.contracts.feature_schema import feature_schema

    registry = ContractRegistry()

    registry.register(ContractMetadata(
        name="credit_raw_v1",
        version="1.0",
        owner="data-engineering@example.com",
        description=(
            "Raw UCI Credit Card Default dataset. "
            "EDUCATION and MARRIAGE still contain undocumented values (0/5/6). "
            "Run clean_raw_data() before passing to feature_schema."
        ),
        enforcement_mode="strict",
        max_age_hours=26.0,
        schema=raw_schema,
        column_owners={
            "LIMIT_BAL": "credit-risk-team@example.com",
            "DEFAULT_PAYMENT_NEXT_MONTH": "data-labelling@example.com",
        },
    ))

    registry.register(ContractMetadata(
        name="credit_feature_v1",
        version="1.0",
        owner="ml-platform@example.com",
        description=(
            "Engineered feature dataset: raw columns (cleaned) + 7 derived features. "
            "Produced by training.features.engineer_features(). "
            "EDUCATION/MARRIAGE undocumented values already remapped."
        ),
        enforcement_mode="strict",
        max_age_hours=48.0,
        schema=feature_schema,
        column_owners={
            "utilization_rate": "ml-platform@example.com",
            "max_delay": "ml-platform@example.com",
        },
    ))

    return registry


# Module-level default registry — import and use directly
default_registry: ContractRegistry = _build_default_registry()
