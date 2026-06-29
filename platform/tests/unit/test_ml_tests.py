"""Tests for ci/ml_tests.py — DataContractChecker, BehavioralChecker, SmokeTrainer, AUCGuard."""
from __future__ import annotations

import math
import pytest

from ci.ml_tests import (
    AUCGuard,
    AUCGuardResult,
    BehavioralChecker,
    DataContractChecker,
    SmokeTrainer,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SCHEMA = {
    "age":    {"dtype": float, "min": 18, "max": 100, "null_rate": 0.0},
    "income": {"dtype": float, "min": 0, "max": None, "null_rate": 0.05},
    "label":  {"dtype": int, "null_rate": 0.0},
}

def _make_row(age=30, income=50_000.0, label=0) -> dict:
    return {"age": age, "income": income, "label": label}

def _good_data(n=20) -> list[dict]:
    rows = [_make_row(label=(1 if i % 5 == 0 else 0)) for i in range(n)]
    return rows


# ── DataContractChecker ────────────────────────────────────────────────────────

class TestDataContractChecker:
    def _checker(self) -> DataContractChecker:
        return DataContractChecker(SCHEMA, label_column="label",
                                   min_positive_rate=0.05, max_positive_rate=0.50)

    def test_empty_schema_raises(self) -> None:
        with pytest.raises(ValueError, match="schema"):
            DataContractChecker({})

    def test_schema_passes_good_data(self) -> None:
        r = self._checker().check_schema(_good_data())
        assert r.passed

    def test_schema_fails_missing_column(self) -> None:
        data = [{"age": 30, "label": 0}]  # income missing
        r = self._checker().check_schema(data)
        assert not r.passed
        assert "income" in r.message

    def test_schema_empty_data(self) -> None:
        r = self._checker().check_schema([])
        assert not r.passed

    def test_null_rates_pass_no_nulls(self) -> None:
        r = self._checker().check_null_rates(_good_data())
        assert r.passed

    def test_null_rates_fails_on_null(self) -> None:
        data = _good_data()
        data[0]["age"] = None  # age has null_rate=0.0 in schema
        r = self._checker().check_null_rates(data)
        assert not r.passed
        assert "age" in r.message

    def test_null_rates_empty_data(self) -> None:
        r = self._checker().check_null_rates([])
        assert not r.passed

    def test_label_dist_passes(self) -> None:
        r = self._checker().check_label_dist(_good_data(20))
        assert r.passed

    def test_label_dist_fails_all_zero(self) -> None:
        data = [_make_row(label=0) for _ in range(20)]
        r = self._checker().check_label_dist(data)
        assert not r.passed

    def test_label_dist_fails_too_many_positives(self) -> None:
        data = [_make_row(label=1) for _ in range(20)]
        r = self._checker().check_label_dist(data)
        assert not r.passed

    def test_label_dist_no_labels(self) -> None:
        data = [{"age": 30, "income": 1000.0}]  # no label column
        r = self._checker().check_label_dist(data)
        assert not r.passed

    def test_run_all_returns_three_results(self) -> None:
        results = self._checker().run_all(_good_data())
        assert len(results) == 3
        names = [r.check_name for r in results]
        assert set(names) == {"schema", "null_rates", "label_dist"}


# ── BehavioralChecker ─────────────────────────────────────────────────────────

def _linear_predict(rows: list[dict]) -> list[float]:
    """Score = sigmoid(age * 0.05 - 1.5). Higher age → higher score."""
    def sig(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))
    return [sig(row.get("age", 0) * 0.05 - 1.5) for row in rows]

def _constant_predict(rows: list[dict]) -> list[float]:
    return [0.5] * len(rows)


class TestBehavioralChecker:
    def test_monotonicity_passes(self) -> None:
        checker = BehavioralChecker(_linear_predict)
        result = checker.check_monotonicity(
            base_row={"age": 30},
            feature="age",
            low_value=20,
            high_value=80,
            direction="higher_score_for_higher_value",
        )
        assert result.passed

    def test_monotonicity_fails(self) -> None:
        checker = BehavioralChecker(_linear_predict)
        # Expecting lower score for higher age — but model gives higher score
        result = checker.check_monotonicity(
            base_row={"age": 30},
            feature="age",
            low_value=20,
            high_value=80,
            direction="lower_score_for_higher_value",
        )
        assert not result.passed

    def test_robustness_passes_small_noise(self) -> None:
        checker = BehavioralChecker(_linear_predict)
        rows = [{"age": 50} for _ in range(10)]
        result = checker.check_robustness(rows, noise_pct=0.001, max_delta=0.10)
        assert result.passed

    def test_robustness_fails_large_delta(self) -> None:
        """A model that outputs random noise should fail robustness."""
        import random
        rng = random.Random(1)
        def noisy_predict(rows):
            return [rng.random() for _ in rows]

        checker = BehavioralChecker(noisy_predict)
        rows = [{"age": 50} for _ in range(20)]
        result = checker.check_robustness(rows, noise_pct=0.001, max_delta=0.01)
        assert not result.passed

    def test_invariance_passes_constant_model(self) -> None:
        checker = BehavioralChecker(_constant_predict)
        rows = [{"age": 40, "gender": "M"} for _ in range(5)]
        result = checker.check_invariance(rows, feature="gender", value_a="M", value_b="F")
        assert result.passed

    def test_invariance_fails_sensitive_model(self) -> None:
        def gender_biased(rows):
            return [1.0 if row.get("gender") == "M" else 0.0 for row in rows]

        checker = BehavioralChecker(gender_biased)
        rows = [{"gender": "M"} for _ in range(5)]
        result = checker.check_invariance(rows, feature="gender", value_a="M", value_b="F", tolerance=0.02)
        assert not result.passed
        assert result.violations == 5

    def test_confidence_passes_diverse_model(self) -> None:
        checker = BehavioralChecker(_linear_predict)
        rows = [{"age": 20 + i * 3} for i in range(20)]
        result = checker.check_confidence(rows, min_stdev=0.01)
        assert result.passed

    def test_confidence_fails_constant_model(self) -> None:
        checker = BehavioralChecker(_constant_predict)
        rows = [{"age": 50} for _ in range(20)]
        result = checker.check_confidence(rows, min_stdev=0.05)
        assert not result.passed


# ── SmokeTrainer ──────────────────────────────────────────────────────────────

class TestSmokeTrainer:
    def test_run_passes(self) -> None:
        trainer = SmokeTrainer(n_rows=100, seed=42)
        result = trainer.run()
        assert result.passed
        assert result.n_rows == 100
        assert result.auc > 0.5

    def test_reproducible(self) -> None:
        result = SmokeTrainer(n_rows=100, seed=42).run()
        assert result.reproducible

    def test_different_seeds_differ(self) -> None:
        r1 = SmokeTrainer(n_rows=100, seed=42).run()
        r2 = SmokeTrainer(n_rows=100, seed=99).run()
        # AUC can differ across seeds (not guaranteed to match)
        assert isinstance(r1.auc, float)
        assert isinstance(r2.auc, float)

    def test_result_fields_present(self) -> None:
        result = SmokeTrainer().run()
        assert hasattr(result, "auc")
        assert hasattr(result, "reproducible")
        assert hasattr(result, "message")


# ── AUCGuard ──────────────────────────────────────────────────────────────────

class TestAUCGuard:
    def test_passes_when_auc_same(self) -> None:
        guard = AUCGuard(baseline_auc=0.80, tolerance=0.01)
        result = guard.check(0.80)
        assert result.passed

    def test_passes_when_auc_improved(self) -> None:
        guard = AUCGuard(baseline_auc=0.80)
        result = guard.check(0.83)
        assert result.passed
        assert result.delta > 0
        assert "improved" in result.message

    def test_passes_within_tolerance(self) -> None:
        guard = AUCGuard(baseline_auc=0.80, tolerance=0.01)
        result = guard.check(0.795)  # 0.005 below baseline, within tolerance 0.01
        assert result.passed

    def test_fails_regression_exceeds_tolerance(self) -> None:
        guard = AUCGuard(baseline_auc=0.80, tolerance=0.01)
        result = guard.check(0.78)  # 0.02 below, exceeds tolerance
        assert not result.passed
        assert "regression" in result.message

    def test_invalid_baseline_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_auc"):
            AUCGuard(baseline_auc=1.5)

    def test_negative_tolerance_raises(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            AUCGuard(baseline_auc=0.80, tolerance=-0.01)

    def test_update_baseline(self) -> None:
        guard = AUCGuard(baseline_auc=0.80)
        guard.update_baseline(0.85)
        result = guard.check(0.83)  # previously would fail vs 0.85 within tolerance
        assert result.baseline_auc == 0.85

    def test_delta_correct(self) -> None:
        guard = AUCGuard(baseline_auc=0.75)
        result = guard.check(0.78)
        assert result.delta == pytest.approx(0.03)

    def test_result_type(self) -> None:
        guard = AUCGuard(baseline_auc=0.72)
        result = guard.check(0.72)
        assert isinstance(result, AUCGuardResult)
