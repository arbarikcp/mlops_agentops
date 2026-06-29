"""Tests for pipelines/model_gate.py — model validation gate."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification

from pipelines.model_gate import (
    ChampionRegistry,
    GateThresholds,
    ModelGate,
    ModelGateReport,
    ModelMetrics,
    compute_model_metrics,
)


# ── ModelMetrics ───────────────────────────────────────────────────────────────

class TestModelMetrics:
    def test_valid_metrics(self) -> None:
        m = ModelMetrics(auc=0.78, ece=0.02, slice_auc_gap=0.05, n_test=200)
        assert m.auc == 0.78

    def test_invalid_auc_raises(self) -> None:
        with pytest.raises(ValueError, match="auc"):
            ModelMetrics(auc=1.5)

    def test_invalid_negative_auc_raises(self) -> None:
        with pytest.raises(ValueError, match="auc"):
            ModelMetrics(auc=-0.1)

    def test_invalid_ece_raises(self) -> None:
        with pytest.raises(ValueError, match="ece"):
            ModelMetrics(auc=0.7, ece=-0.01)

    def test_invalid_slice_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="slice_auc_gap"):
            ModelMetrics(auc=0.7, slice_auc_gap=-0.01)

    def test_to_dict_has_all_keys(self) -> None:
        m = ModelMetrics(auc=0.78, model_version="v1")
        d = m.to_dict()
        assert "auc" in d
        assert "model_version" in d
        assert d["model_version"] == "v1"

    def test_default_status_is_candidate(self) -> None:
        m = ModelMetrics(auc=0.7)
        assert m.status == "candidate"


# ── GateThresholds ─────────────────────────────────────────────────────────────

class TestGateThresholds:
    def test_defaults(self) -> None:
        t = GateThresholds()
        assert t.min_auc == 0.75
        assert t.max_ece == 0.05
        assert t.max_slice_gap == 0.10
        assert t.champion_delta == 0.005

    def test_invalid_min_auc_raises(self) -> None:
        with pytest.raises(ValueError, match="min_auc"):
            GateThresholds(min_auc=0.0)

    def test_invalid_max_ece_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ece"):
            GateThresholds(max_ece=-0.01)

    def test_invalid_slice_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="max_slice_gap"):
            GateThresholds(max_slice_gap=-1)

    def test_invalid_delta_raises(self) -> None:
        with pytest.raises(ValueError, match="champion_delta"):
            GateThresholds(champion_delta=-0.01)

    def test_from_env_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv("GATE_MIN_AUC", raising=False)
        t = GateThresholds.from_env()
        assert t.min_auc == 0.75

    def test_from_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("GATE_MIN_AUC", "0.80")
        monkeypatch.setenv("GATE_MAX_SLICE_GAP", "0.05")
        t = GateThresholds.from_env()
        assert t.min_auc == 0.80
        assert t.max_slice_gap == 0.05


# ── ChampionRegistry ───────────────────────────────────────────────────────────

class TestChampionRegistry:
    def test_no_champion_initially(self) -> None:
        reg = ChampionRegistry()
        assert reg.get_champion() is None

    def test_promote_sets_champion(self) -> None:
        reg = ChampionRegistry()
        m = ModelMetrics(auc=0.78, model_version="v1")
        reg.promote(m)
        assert reg.get_champion().model_version == "v1"
        assert reg.get_champion().status == "champion"

    def test_promote_archives_previous(self) -> None:
        reg = ChampionRegistry()
        v1 = ModelMetrics(auc=0.78, model_version="v1")
        v2 = ModelMetrics(auc=0.80, model_version="v2")
        reg.promote(v1)
        reg.promote(v2)
        assert v1.status == "previous_stable"
        assert reg.get_champion().model_version == "v2"

    def test_reject_marks_status(self) -> None:
        reg = ChampionRegistry()
        m = ModelMetrics(auc=0.70, model_version="v1")
        reg.reject(m, reason="AUC too low")
        assert m.status == "rejected"
        assert "rejection_reason" in m.extra

    def test_rollback_restores_previous(self) -> None:
        reg = ChampionRegistry()
        v1 = ModelMetrics(auc=0.78, model_version="v1")
        v2 = ModelMetrics(auc=0.80, model_version="v2")
        reg.promote(v1)
        reg.promote(v2)
        restored = reg.rollback()
        assert restored.model_version == "v1"
        assert reg.get_champion().model_version == "v1"

    def test_rollback_returns_none_if_no_history(self) -> None:
        reg = ChampionRegistry()
        v1 = ModelMetrics(auc=0.78, model_version="v1")
        reg.promote(v1)
        result = reg.rollback()
        assert result is None

    def test_previous_stable_returns_second_to_last(self) -> None:
        reg = ChampionRegistry()
        v1 = ModelMetrics(auc=0.78, model_version="v1")
        v2 = ModelMetrics(auc=0.80, model_version="v2")
        reg.promote(v1)
        reg.promote(v2)
        prev = reg.previous_stable()
        assert prev.model_version == "v1"

    def test_history_accumulates(self) -> None:
        reg = ChampionRegistry()
        reg.promote(ModelMetrics(auc=0.78, model_version="v1"))
        reg.reject(ModelMetrics(auc=0.70, model_version="v2"), "AUC too low")
        reg.promote(ModelMetrics(auc=0.82, model_version="v3"))
        assert len(reg.history) == 3


# ── ModelGate ─────────────────────────────────────────────────────────────────

class TestModelGate:
    @pytest.fixture
    def gate(self) -> ModelGate:
        return ModelGate(
            thresholds=GateThresholds(min_auc=0.60, max_ece=0.10, max_slice_gap=0.20, champion_delta=0.01),
            registry=ChampionRegistry(),
        )

    def test_first_model_promoted_automatically(self, gate) -> None:
        m = ModelMetrics(auc=0.75, ece=0.03, model_version="v1")
        report = gate.evaluate(m)
        assert report.promoted is True
        assert report.champion_metrics is None
        assert "First model" in report.promotion_reason

    def test_challenger_better_than_champion_promoted(self, gate) -> None:
        gate.registry.promote(ModelMetrics(auc=0.75, model_version="v1"))
        challenger = ModelMetrics(auc=0.80, ece=0.03, model_version="v2")
        report = gate.evaluate(challenger)
        assert report.promoted is True
        assert gate.registry.get_champion().model_version == "v2"

    def test_challenger_not_better_enough_rejected(self, gate) -> None:
        gate.registry.promote(ModelMetrics(auc=0.78, model_version="v1"))
        challenger = ModelMetrics(auc=0.785, ece=0.03, model_version="v2")  # delta=0.005 < 0.01
        report = gate.evaluate(challenger)
        assert report.promoted is False
        assert report.passed is False

    def test_auc_below_threshold_rejected(self, gate) -> None:
        m = ModelMetrics(auc=0.50, model_version="v1")  # below min_auc=0.60
        report = gate.evaluate(m)
        assert report.promoted is False
        assert any("AUC" in f for f in report.gate_failures)

    def test_ece_above_threshold_rejected(self, gate) -> None:
        m = ModelMetrics(auc=0.80, ece=0.20, model_version="v1")  # above max_ece=0.10
        report = gate.evaluate(m)
        assert not report.promoted
        assert any("ECE" in f for f in report.gate_failures)

    def test_slice_gap_above_threshold_rejected(self, gate) -> None:
        m = ModelMetrics(auc=0.80, ece=0.03, slice_auc_gap=0.30, model_version="v1")
        report = gate.evaluate(m)
        assert not report.promoted
        assert any("slice" in f for f in report.gate_failures)

    def test_cost_gate_enforced(self) -> None:
        gate = ModelGate(
            thresholds=GateThresholds(min_auc=0.60, max_cost=1_000_000),
            registry=ChampionRegistry(),
        )
        m = ModelMetrics(auc=0.78, cost_at_threshold=2_000_000, model_version="v1")
        report = gate.evaluate(m)
        assert not report.promoted
        assert any("cost" in f for f in report.gate_failures)

    def test_report_to_dict(self, gate) -> None:
        m = ModelMetrics(auc=0.75, model_version="v1")
        report = gate.evaluate(m)
        d = report.to_dict()
        assert "passed" in d
        assert "promoted" in d
        assert "challenger" in d

    def test_hard_gate_failure_skips_champion_comparison(self, gate) -> None:
        gate.registry.promote(ModelMetrics(auc=0.50, model_version="v0"))
        challenger = ModelMetrics(auc=0.50, model_version="v1")  # below min_auc=0.60
        report = gate.evaluate(challenger)
        # Champion should remain v0 — no comparison was done
        assert gate.registry.get_champion().model_version == "v0"
        assert not report.promoted

    def test_duration_positive(self, gate) -> None:
        m = ModelMetrics(auc=0.75)
        report = gate.evaluate(m)
        assert report.duration_s >= 0


# ── compute_model_metrics ─────────────────────────────────────────────────────

class TestComputeModelMetrics:
    @pytest.fixture
    def fitted_model_and_data(self):
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=20, random_state=42)
        clf.fit(X_train, y_train)
        return clf, X_test, y_test

    def test_auc_in_range(self, fitted_model_and_data) -> None:
        clf, X_test, y_test = fitted_model_and_data
        metrics = compute_model_metrics(clf, X_test, y_test, model_version="v1")
        assert 0.5 <= metrics.auc <= 1.0

    def test_ece_non_negative(self, fitted_model_and_data) -> None:
        clf, X_test, y_test = fitted_model_and_data
        metrics = compute_model_metrics(clf, X_test, y_test)
        assert metrics.ece >= 0

    def test_brier_in_range(self, fitted_model_and_data) -> None:
        clf, X_test, y_test = fitted_model_and_data
        metrics = compute_model_metrics(clf, X_test, y_test)
        assert 0 <= metrics.brier <= 1

    def test_n_test_matches(self, fitted_model_and_data) -> None:
        clf, X_test, y_test = fitted_model_and_data
        metrics = compute_model_metrics(clf, X_test, y_test)
        assert metrics.n_test == len(y_test)

    def test_cost_positive(self, fitted_model_and_data) -> None:
        clf, X_test, y_test = fitted_model_and_data
        metrics = compute_model_metrics(clf, X_test, y_test, cost_fp=2000, cost_fn=8000)
        assert metrics.cost_at_threshold >= 0

    def test_slice_gap_with_column(self, fitted_model_and_data) -> None:
        clf, X_test, y_test = fitted_model_and_data
        rng = np.random.default_rng(0)
        groups = rng.integers(1, 4, size=len(y_test))
        metrics = compute_model_metrics(clf, X_test, y_test, slice_column=groups)
        assert metrics.slice_auc_gap >= 0

    def test_model_version_set(self, fitted_model_and_data) -> None:
        clf, X_test, y_test = fitted_model_and_data
        metrics = compute_model_metrics(clf, X_test, y_test, model_version="v-abc123")
        assert metrics.model_version == "v-abc123"
