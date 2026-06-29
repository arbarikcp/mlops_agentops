"""Tests for pipelines/validation_gate.py — data validation gate."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines.validation_gate import (
    DataValidationGate,
    SchemaCheck,
    StatisticalCheck,
    ValidationGateFailure,
    ValidationGateReport,
    credit_risk_gate,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "LIMIT_BAL": rng.uniform(10_000, 500_000, n).astype(float),
        "AGE": rng.integers(20, 70, n),
        "BILL_AMT1": rng.uniform(0, 200_000, n).astype(float),
        "PAY_AMT1": rng.uniform(0, 50_000, n).astype(float),
        "default.payment.next.month": rng.integers(0, 2, n),
        "EDUCATION": rng.integers(1, 5, n),
    })


# ── SchemaCheck ────────────────────────────────────────────────────────────────

class TestSchemaCheck:
    def test_valid_column_no_failures(self) -> None:
        df = make_df()
        check = SchemaCheck("LIMIT_BAL", dtype="float", nullable=False, min_val=0)
        assert check.check(df) == []

    def test_missing_required_column(self) -> None:
        df = pd.DataFrame({"A": [1, 2, 3]})
        check = SchemaCheck("B", required=True)
        failures = check.check(df)
        assert any("missing" in f for f in failures)

    def test_optional_missing_column_no_failure(self) -> None:
        df = pd.DataFrame({"A": [1, 2, 3]})
        check = SchemaCheck("B", required=False)
        assert check.check(df) == []

    def test_null_check_failure(self) -> None:
        df = pd.DataFrame({"X": [1.0, None, 3.0]})
        check = SchemaCheck("X", nullable=False)
        failures = check.check(df)
        assert any("null" in f for f in failures)

    def test_min_val_violation(self) -> None:
        df = pd.DataFrame({"X": [-1.0, 2.0, 3.0]})
        check = SchemaCheck("X", min_val=0.0)
        failures = check.check(df)
        assert any("below min" in f for f in failures)

    def test_max_val_violation(self) -> None:
        df = pd.DataFrame({"X": [1.0, 2.0, 999.0]})
        check = SchemaCheck("X", max_val=100.0)
        failures = check.check(df)
        assert any("above max" in f for f in failures)

    def test_allowed_values_violation(self) -> None:
        df = pd.DataFrame({"Y": [0, 1, 5]})
        check = SchemaCheck("Y", allowed_values=[0, 1])
        failures = check.check(df)
        assert any("not in" in f for f in failures)

    def test_allowed_values_all_valid(self) -> None:
        df = pd.DataFrame({"Y": [0, 1, 0, 1]})
        check = SchemaCheck("Y", allowed_values=[0, 1])
        assert check.check(df) == []

    def test_multiple_violations_collected(self) -> None:
        df = pd.DataFrame({"X": [-5.0, None, 200.0]})
        check = SchemaCheck("X", nullable=False, min_val=0.0, max_val=100.0)
        failures = check.check(df)
        assert len(failures) >= 2


# ── StatisticalCheck ───────────────────────────────────────────────────────────

class TestStatisticalCheck:
    def test_min_rows_passes(self) -> None:
        df = make_df(200)
        check = StatisticalCheck("min_rows", threshold=100)
        assert check.check(df) == []

    def test_min_rows_fails(self) -> None:
        df = make_df(50)
        check = StatisticalCheck("min_rows", threshold=100)
        failures = check.check(df)
        assert len(failures) == 1
        assert "Row count" in failures[0]

    def test_null_rate_passes(self) -> None:
        df = pd.DataFrame({"Y": [0, 1, 0, 1, 0]})
        check = StatisticalCheck("null_rate", column="Y", threshold=0.10)
        assert check.check(df) == []

    def test_null_rate_fails(self) -> None:
        df = pd.DataFrame({"Y": [0, None, None, 1, None]})  # 60% null
        check = StatisticalCheck("null_rate", column="Y", threshold=0.10)
        failures = check.check(df)
        assert len(failures) == 1
        assert "Null rate" in failures[0]

    def test_positive_rate_in_range(self) -> None:
        df = pd.DataFrame({"Y": [0, 1, 0, 1, 0, 1]})  # 50%
        check = StatisticalCheck("positive_rate", column="Y", threshold=0.10, threshold_high=0.70)
        assert check.check(df) == []

    def test_positive_rate_out_of_range(self) -> None:
        df = pd.DataFrame({"Y": [0, 0, 0, 0, 0, 0]})  # 0% — below 1%
        check = StatisticalCheck("positive_rate", column="Y", threshold=0.01, threshold_high=0.70)
        failures = check.check(df)
        assert len(failures) == 1

    def test_mean_range_passes(self) -> None:
        df = pd.DataFrame({"X": [100.0, 200.0, 150.0]})
        check = StatisticalCheck("mean_range", column="X", min_val=50.0, max_val=300.0)
        assert check.check(df) == []

    def test_mean_range_fails_low(self) -> None:
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
        check = StatisticalCheck("mean_range", column="X", min_val=100.0)
        failures = check.check(df)
        assert any("<" in f for f in failures)

    def test_std_positive_passes(self) -> None:
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]})
        check = StatisticalCheck("std_positive", column="X")
        assert check.check(df) == []

    def test_std_positive_fails_on_constant(self) -> None:
        df = pd.DataFrame({"X": [5.0, 5.0, 5.0, 5.0]})
        check = StatisticalCheck("std_positive", column="X")
        failures = check.check(df)
        assert any("zero standard deviation" in f for f in failures)

    def test_no_inf_passes(self) -> None:
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
        check = StatisticalCheck("no_inf", column="X")
        assert check.check(df) == []

    def test_no_inf_fails(self) -> None:
        df = pd.DataFrame({"X": [1.0, float("inf"), 3.0]})
        check = StatisticalCheck("no_inf", column="X")
        failures = check.check(df)
        assert any("Inf" in f for f in failures)

    def test_unknown_check_type_no_failure(self) -> None:
        df = make_df()
        check = StatisticalCheck("unknown_check_type")
        assert check.check(df) == []


# ── ValidationGateReport ───────────────────────────────────────────────────────

class TestValidationGateReport:
    def test_all_failures_combines(self) -> None:
        r = ValidationGateReport(
            passed=False,
            n_rows_checked=100,
            schema_failures=["col A null"],
            statistical_failures=["row count too low"],
        )
        assert len(r.all_failures) == 2

    def test_n_failures_count(self) -> None:
        r = ValidationGateReport(
            passed=False, n_rows_checked=100,
            schema_failures=["f1", "f2"],
            statistical_failures=["f3"],
        )
        assert r.n_failures == 3

    def test_to_dict_keys(self) -> None:
        r = ValidationGateReport(passed=True, n_rows_checked=500)
        d = r.to_dict()
        assert "passed" in d
        assert "n_rows_checked" in d
        assert "n_failures" in d


# ── DataValidationGate ─────────────────────────────────────────────────────────

class TestDataValidationGate:
    def test_passes_on_good_data(self) -> None:
        gate = DataValidationGate(
            schema_checks=[SchemaCheck("X", dtype="float", nullable=False)],
            statistical_checks=[StatisticalCheck("min_rows", threshold=3)],
            raise_on_failure=True,
        )
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0, 4.0]})
        report = gate.validate(df)
        assert report.passed is True
        assert report.n_failures == 0

    def test_raises_on_failure(self) -> None:
        gate = DataValidationGate(
            schema_checks=[SchemaCheck("Y", required=True)],
            raise_on_failure=True,
        )
        df = pd.DataFrame({"X": [1.0]})
        with pytest.raises(ValidationGateFailure) as exc_info:
            gate.validate(df)
        assert len(exc_info.value.failures) > 0

    def test_no_raise_when_disabled(self) -> None:
        gate = DataValidationGate(
            schema_checks=[SchemaCheck("Z", required=True)],
            raise_on_failure=False,
        )
        df = pd.DataFrame({"X": [1.0]})
        report = gate.validate(df)
        assert report.passed is False

    def test_lazy_collects_all_failures(self) -> None:
        gate = DataValidationGate(
            schema_checks=[
                SchemaCheck("A", required=True),
                SchemaCheck("B", required=True),
                SchemaCheck("C", required=True),
            ],
            raise_on_failure=False,
        )
        df = pd.DataFrame({"X": [1.0]})  # A, B, C all missing
        report = gate.validate(df)
        assert report.n_failures == 3

    def test_add_schema_check_chaining(self) -> None:
        gate = DataValidationGate()
        returned = gate.add_schema_check(SchemaCheck("X"))
        assert returned is gate
        assert len(gate.schema_checks) == 1

    def test_add_statistical_check_chaining(self) -> None:
        gate = DataValidationGate()
        returned = gate.add_statistical_check(StatisticalCheck("min_rows", threshold=1))
        assert returned is gate
        assert len(gate.statistical_checks) == 1

    def test_n_rows_checked_in_report(self) -> None:
        gate = DataValidationGate(raise_on_failure=False)
        df = pd.DataFrame({"X": range(42)})
        report = gate.validate(df)
        assert report.n_rows_checked == 42

    def test_duration_positive(self) -> None:
        gate = DataValidationGate(raise_on_failure=False)
        df = make_df()
        report = gate.validate(df)
        assert report.duration_s >= 0


# ── credit_risk_gate ───────────────────────────────────────────────────────────

class TestCreditRiskGate:
    def test_passes_on_synthetic_data(self) -> None:
        df = make_df(500)
        # Fix AGE dtype to int
        df["AGE"] = df["AGE"].astype(int)
        gate = credit_risk_gate(raise_on_failure=False)
        report = gate.validate(df)
        assert report.passed is True

    def test_fails_on_too_few_rows(self) -> None:
        df = make_df(10)
        df["AGE"] = df["AGE"].astype(int)
        gate = credit_risk_gate(raise_on_failure=False)
        report = gate.validate(df)
        assert not report.passed
        assert any("Row count" in f for f in report.statistical_failures)

    def test_fails_on_negative_limit_bal(self) -> None:
        df = make_df(500)
        df["AGE"] = df["AGE"].astype(int)
        df.loc[0, "LIMIT_BAL"] = -1000.0  # violate min_val
        gate = credit_risk_gate(raise_on_failure=False)
        report = gate.validate(df)
        assert not report.passed

    def test_raises_on_failure_by_default(self) -> None:
        df = make_df(5)  # too few rows
        df["AGE"] = df["AGE"].astype(int)
        gate = credit_risk_gate(raise_on_failure=True)
        with pytest.raises(ValidationGateFailure):
            gate.validate(df)

    def test_failure_exception_carries_failures_list(self) -> None:
        df = make_df(5)
        df["AGE"] = df["AGE"].astype(int)
        gate = credit_risk_gate(raise_on_failure=True)
        with pytest.raises(ValidationGateFailure) as exc_info:
            gate.validate(df)
        assert len(exc_info.value.failures) > 0
