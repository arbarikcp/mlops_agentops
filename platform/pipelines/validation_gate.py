"""Data validation gate — schema, domain, and statistical checks wired into pipeline.

Implements Pandera-style schema validation and Great Expectations-style
statistical checks as a pipeline gate step:
  - SchemaCheck:         per-column type, null, range, set constraints
  - StatisticalCheck:    row count, null rate, class balance, PSI
  - DataValidationGate:  runs all checks; raises or returns report
  - ValidationGateReport: typed output with full failure list
  - credit_risk_gate():  pre-built gate for credit-risk feature schema

See: docs/phase5/day34_validation_gate.md
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Exception ─────────────────────────────────────────────────────────────────

class ValidationGateFailure(RuntimeError):
    """Raised by DataValidationGate when validation fails.

    Attributes:
        failures:  List of all failure messages collected in lazy mode.
        report:    The full ValidationGateReport (may be None on hard error).
    """

    def __init__(self, message: str, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures = failures or []


# ── Schema Check ──────────────────────────────────────────────────────────────

@dataclass
class SchemaCheck:
    """Per-column structural constraint (Pandera-style).

    Attributes:
        column:          Column name to check.
        dtype:           Expected pandas dtype prefix (e.g. "float", "int", "object").
        nullable:        If False, column must have no NaN values.
        min_val:         Inclusive minimum for numeric columns.
        max_val:         Inclusive maximum for numeric columns.
        allowed_values:  If set, all values must be in this set.
        required:        If True, column must exist in the DataFrame.
    """

    column: str
    dtype: str = "float"
    nullable: bool = True
    min_val: float | None = None
    max_val: float | None = None
    allowed_values: list[Any] | None = None
    required: bool = True

    def check(self, df: pd.DataFrame) -> list[str]:
        """Return list of failure messages (empty = passed)."""
        failures: list[str] = []

        if self.column not in df.columns:
            if self.required:
                failures.append(f"Column '{self.column}' is missing")
            return failures

        col = df[self.column]

        # dtype check
        if not str(col.dtype).startswith(self.dtype):
            try:
                col.astype(self.dtype)
            except (ValueError, TypeError):
                failures.append(
                    f"Column '{self.column}': expected dtype '{self.dtype}', got '{col.dtype}'"
                )

        # null check
        if not self.nullable:
            n_null = int(col.isna().sum())
            if n_null > 0:
                failures.append(f"Column '{self.column}': {n_null} null values (not nullable)")

        # range checks
        non_null = col.dropna()
        if self.min_val is not None:
            n_below = int((non_null < self.min_val).sum())
            if n_below > 0:
                failures.append(
                    f"Column '{self.column}': {n_below} values below min {self.min_val}"
                )

        if self.max_val is not None:
            n_above = int((non_null > self.max_val).sum())
            if n_above > 0:
                failures.append(
                    f"Column '{self.column}': {n_above} values above max {self.max_val}"
                )

        # allowed values
        if self.allowed_values is not None:
            invalid = non_null[~non_null.isin(self.allowed_values)]
            if len(invalid) > 0:
                failures.append(
                    f"Column '{self.column}': {len(invalid)} values not in {self.allowed_values}"
                )

        return failures


# ── Statistical Check ─────────────────────────────────────────────────────────

@dataclass
class StatisticalCheck:
    """DataFrame-level statistical constraint (Great Expectations-style).

    Supported check_types:
        - "min_rows":       df must have at least `threshold` rows
        - "null_rate":      null rate in `column` must be <= `threshold`
        - "positive_rate":  positive rate in binary `column` must be in
                            [threshold_low, threshold_high] (uses extra field)
        - "mean_range":     column mean must be in [min_val, max_val]
        - "std_positive":   column std must be > 0 (non-constant)
        - "no_inf":         column must have no Inf values
    """

    check_type: str
    column: str | None = None
    threshold: float = 0.0
    threshold_high: float = 1.0
    min_val: float | None = None
    max_val: float | None = None
    description: str = ""

    def check(self, df: pd.DataFrame) -> list[str]:
        """Return list of failure messages (empty = passed)."""
        failures: list[str] = []

        if self.check_type == "min_rows":
            if len(df) < self.threshold:
                failures.append(
                    f"Row count {len(df)} < minimum {int(self.threshold)}"
                )

        elif self.check_type == "null_rate":
            if self.column is None or self.column not in df.columns:
                return failures
            rate = float(df[self.column].isna().mean())
            if rate > self.threshold:
                failures.append(
                    f"Null rate for '{self.column}': {rate:.1%} > threshold {self.threshold:.1%}"
                )

        elif self.check_type == "positive_rate":
            if self.column is None or self.column not in df.columns:
                return failures
            rate = float(df[self.column].mean())
            if not (self.threshold <= rate <= self.threshold_high):
                failures.append(
                    f"Positive rate for '{self.column}': {rate:.1%} not in "
                    f"[{self.threshold:.1%}, {self.threshold_high:.1%}]"
                )

        elif self.check_type == "mean_range":
            if self.column is None or self.column not in df.columns:
                return failures
            mean = float(df[self.column].mean())
            if self.min_val is not None and mean < self.min_val:
                failures.append(
                    f"Mean of '{self.column}': {mean:.2f} < {self.min_val}"
                )
            if self.max_val is not None and mean > self.max_val:
                failures.append(
                    f"Mean of '{self.column}': {mean:.2f} > {self.max_val}"
                )

        elif self.check_type == "std_positive":
            if self.column is None or self.column not in df.columns:
                return failures
            std = float(df[self.column].std())
            if std <= 0:
                failures.append(
                    f"Column '{self.column}' has zero standard deviation (constant column)"
                )

        elif self.check_type == "no_inf":
            if self.column is None or self.column not in df.columns:
                return failures
            n_inf = int(np.isinf(df[self.column].replace([None], np.nan)).sum())
            if n_inf > 0:
                failures.append(
                    f"Column '{self.column}': {n_inf} Inf values"
                )

        else:
            log.warning("Unknown check_type: %r", self.check_type)

        return failures


# ── Validation Gate Report ────────────────────────────────────────────────────

@dataclass
class ValidationGateReport:
    """Result of running a DataValidationGate.

    Attributes:
        passed:               True if all schema and statistical checks passed.
        n_rows_checked:       Number of rows in the validated DataFrame.
        schema_failures:      Failure messages from schema checks.
        statistical_failures: Failure messages from statistical checks.
        warnings:             Non-blocking notices.
        duration_s:           Wall-clock seconds the gate took.
    """

    passed: bool
    n_rows_checked: int
    schema_failures: list[str] = field(default_factory=list)
    statistical_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def all_failures(self) -> list[str]:
        return self.schema_failures + self.statistical_failures

    @property
    def n_failures(self) -> int:
        return len(self.all_failures)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_rows_checked": self.n_rows_checked,
            "n_failures": self.n_failures,
            "schema_failures": self.schema_failures,
            "statistical_failures": self.statistical_failures,
            "warnings": self.warnings,
            "duration_s": self.duration_s,
        }


# ── Data Validation Gate ──────────────────────────────────────────────────────

class DataValidationGate:
    """Pipeline gate that applies schema and statistical checks to a DataFrame.

    Runs all checks in lazy mode (collects all failures before deciding).
    If `raise_on_failure=True`, raises ValidationGateFailure with the full
    failure list if any check fails.

    Args:
        schema_checks:       List of SchemaCheck objects.
        statistical_checks:  List of StatisticalCheck objects.
        raise_on_failure:    If True, raise on any failure; if False, return report.
        gate_name:           Human-readable name for logging.
    """

    def __init__(
        self,
        schema_checks: list[SchemaCheck] | None = None,
        statistical_checks: list[StatisticalCheck] | None = None,
        *,
        raise_on_failure: bool = True,
        gate_name: str = "data_validation_gate",
    ) -> None:
        self.schema_checks = schema_checks or []
        self.statistical_checks = statistical_checks or []
        self.raise_on_failure = raise_on_failure
        self.gate_name = gate_name

    def validate(self, df: pd.DataFrame) -> ValidationGateReport:
        """Run all checks against `df`.

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationGateReport with results.

        Raises:
            ValidationGateFailure: If any check fails and raise_on_failure=True.
        """
        start = time.monotonic()
        schema_failures: list[str] = []
        statistical_failures: list[str] = []

        for check in self.schema_checks:
            schema_failures.extend(check.check(df))

        for check in self.statistical_checks:
            statistical_failures.extend(check.check(df))

        duration = time.monotonic() - start
        passed = len(schema_failures) == 0 and len(statistical_failures) == 0

        report = ValidationGateReport(
            passed=passed,
            n_rows_checked=len(df),
            schema_failures=schema_failures,
            statistical_failures=statistical_failures,
            duration_s=duration,
        )

        if passed:
            log.info("%s: PASSED (%d rows, %.3fs)", self.gate_name, len(df), duration)
        else:
            all_failures = report.all_failures
            log.warning(
                "%s: FAILED (%d failures in %.3fs): %s",
                self.gate_name, len(all_failures), duration, all_failures[:3],
            )
            if self.raise_on_failure:
                raise ValidationGateFailure(
                    f"{self.gate_name} failed with {len(all_failures)} violations",
                    failures=all_failures,
                )

        return report

    def add_schema_check(self, check: SchemaCheck) -> "DataValidationGate":
        self.schema_checks.append(check)
        return self

    def add_statistical_check(self, check: StatisticalCheck) -> "DataValidationGate":
        self.statistical_checks.append(check)
        return self


# ── Pre-built gate for credit-risk feature data ───────────────────────────────

def credit_risk_gate(*, raise_on_failure: bool = True) -> DataValidationGate:
    """Pre-built validation gate for the credit-risk feature dataset.

    Schema checks:
        LIMIT_BAL: float, non-null, > 0
        AGE: int-compatible, 18–100
        default.payment.next.month: int, 0 or 1

    Statistical checks:
        Row count >= 100
        Label null rate <= 5%
        Positive rate in [1%, 70%]
        LIMIT_BAL mean in [5000, 1_000_000]
    """
    return DataValidationGate(
        schema_checks=[
            SchemaCheck("LIMIT_BAL", dtype="float", nullable=False, min_val=0),
            SchemaCheck("AGE", dtype="int", nullable=False, min_val=18, max_val=100),
            SchemaCheck(
                "default.payment.next.month",
                dtype="int",
                nullable=False,
                allowed_values=[0, 1],
            ),
        ],
        statistical_checks=[
            StatisticalCheck("min_rows", threshold=100),
            StatisticalCheck(
                "null_rate",
                column="default.payment.next.month",
                threshold=0.05,
            ),
            StatisticalCheck(
                "positive_rate",
                column="default.payment.next.month",
                threshold=0.01,
                threshold_high=0.70,
            ),
            StatisticalCheck(
                "mean_range",
                column="LIMIT_BAL",
                min_val=5_000,
                max_val=1_000_000,
            ),
        ],
        raise_on_failure=raise_on_failure,
        gate_name="credit_risk_data_gate",
    )
