"""Tests for infra/progressive_delivery.py — CanaryStep, RolloutStrategy, AnalysisTemplate, ArgoRollout."""
from __future__ import annotations

import pytest

from infra.progressive_delivery import (
    AnalysisMetric,
    AnalysisTemplate,
    ArgoRollout,
    CanaryStep,
    RolloutStrategy,
)


# ── CanaryStep ────────────────────────────────────────────────────────────────

class TestCanaryStep:
    def test_valid_step(self) -> None:
        s = CanaryStep(weight=10)
        assert s.weight == 10

    def test_weight_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            CanaryStep(weight=0)

    def test_weight_101_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            CanaryStep(weight=101)

    def test_negative_pause_raises(self) -> None:
        with pytest.raises(ValueError, match="pause_minutes"):
            CanaryStep(weight=10, pause_minutes=-1)

    def test_to_dict_weight_only(self) -> None:
        steps = CanaryStep(weight=10).to_dict()
        assert steps == [{"setWeight": 10}]

    def test_to_dict_with_pause(self) -> None:
        steps = CanaryStep(weight=10, pause_minutes=30).to_dict()
        assert {"pause": {"duration": "30m"}} in steps

    def test_to_dict_with_analysis(self) -> None:
        steps = CanaryStep(weight=10, analysis_template="ml-quality-gate").to_dict()
        assert any("analysis" in s for s in steps)

    def test_to_dict_full_step(self) -> None:
        steps = CanaryStep(weight=10, pause_minutes=30, analysis_template="ml-gate").to_dict()
        assert len(steps) == 3


# ── RolloutStrategy ───────────────────────────────────────────────────────────

class TestRolloutStrategy:
    def make_valid(self) -> RolloutStrategy:
        return RolloutStrategy(steps=[
            CanaryStep(weight=10, pause_minutes=30),
            CanaryStep(weight=50, pause_minutes=15),
            CanaryStep(weight=100),
        ])

    def test_valid_strategy(self) -> None:
        rs = self.make_valid()
        assert len(rs.steps) == 3

    def test_empty_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="steps"):
            RolloutStrategy(steps=[])

    def test_last_step_not_100_raises(self) -> None:
        with pytest.raises(ValueError, match="weight=100"):
            RolloutStrategy(steps=[CanaryStep(weight=50)])

    def test_negative_max_surge_raises(self) -> None:
        with pytest.raises(ValueError, match="max_surge"):
            RolloutStrategy(
                steps=[CanaryStep(weight=100)],
                max_surge=-1,
            )

    def test_to_dict_has_canary_key(self) -> None:
        d = self.make_valid().to_dict()
        assert "canary" in d

    def test_to_dict_has_steps(self) -> None:
        d = self.make_valid().to_dict()
        assert len(d["canary"]["steps"]) > 0

    def test_to_dict_has_metadata(self) -> None:
        d = self.make_valid().to_dict()
        assert "canaryMetadata" in d["canary"]
        assert "stableMetadata" in d["canary"]


# ── AnalysisMetric ────────────────────────────────────────────────────────────

class TestAnalysisMetric:
    def make(self, **kwargs) -> AnalysisMetric:
        defaults = dict(
            name="model-auc",
            prometheus_query='avg(ml_model_auc{variant="canary"}[10m])',
            success_condition="result >= 0.78",
        )
        defaults.update(kwargs)
        return AnalysisMetric(**defaults)

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            self.make(name="")

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="prometheus_query"):
            self.make(prometheus_query="")

    def test_empty_success_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="success_condition"):
            self.make(success_condition="")

    def test_negative_failure_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="failure_limit"):
            self.make(failure_limit=-1)

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="interval_m"):
            self.make(interval_m=0)

    def test_to_dict_structure(self) -> None:
        d = self.make().to_dict()
        assert d["name"] == "model-auc"
        assert "prometheus" in d["provider"]
        assert d["successCondition"] == "result >= 0.78"

    def test_interval_in_dict(self) -> None:
        d = self.make(interval_m=10).to_dict()
        assert d["interval"] == "10m"


# ── AnalysisTemplate ──────────────────────────────────────────────────────────

class TestAnalysisTemplate:
    def make(self, **kwargs) -> AnalysisTemplate:
        metric = AnalysisMetric(
            name="auc",
            prometheus_query="avg(ml_model_auc[5m])",
            success_condition="result >= 0.78",
        )
        defaults = dict(name="ml-gate", metrics=[metric])
        defaults.update(kwargs)
        return AnalysisTemplate(**defaults)

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            self.make(name="")

    def test_empty_metrics_raises(self) -> None:
        with pytest.raises(ValueError, match="metrics"):
            self.make(metrics=[])

    def test_to_manifest_kind(self) -> None:
        m = self.make().to_manifest()
        assert m["kind"] == "AnalysisTemplate"
        assert m["apiVersion"] == "argoproj.io/v1alpha1"

    def test_to_manifest_has_metrics(self) -> None:
        m = self.make().to_manifest()
        assert len(m["spec"]["metrics"]) == 1

    def test_ml_quality_gate_factory(self) -> None:
        at = AnalysisTemplate.ml_quality_gate()
        assert at.name == "ml-quality-gate"
        assert len(at.metrics) == 2

    def test_ml_quality_gate_auc_metric(self) -> None:
        at = AnalysisTemplate.ml_quality_gate(min_auc=0.80)
        auc_metric = next(m for m in at.metrics if m.name == "model-auc")
        assert "0.8" in auc_metric.success_condition

    def test_ml_quality_gate_psi_metric(self) -> None:
        at = AnalysisTemplate.ml_quality_gate(max_psi=0.15)
        psi_metric = next(m for m in at.metrics if m.name == "psi-check")
        assert "0.15" in psi_metric.success_condition


# ── ArgoRollout ───────────────────────────────────────────────────────────────

class TestArgoRollout:
    def make(self, **kwargs) -> ArgoRollout:
        defaults = dict(
            name="credit-risk-api",
            image="ghcr.io/arbarikcp/credit-risk-api:v2.1.0",
        )
        defaults.update(kwargs)
        return ArgoRollout(**defaults)

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            self.make(name="")

    def test_zero_replicas_raises(self) -> None:
        with pytest.raises(ValueError, match="replicas"):
            self.make(replicas=0)

    def test_to_manifest_kind(self) -> None:
        m = self.make().to_manifest()
        assert m["kind"] == "Rollout"
        assert m["apiVersion"] == "argoproj.io/v1alpha1"

    def test_to_manifest_strategy(self) -> None:
        m = self.make().to_manifest()
        assert "strategy" in m["spec"]
        assert "canary" in m["spec"]["strategy"]

    def test_to_manifest_selector(self) -> None:
        m = self.make().to_manifest()
        assert m["spec"]["selector"]["matchLabels"]["app"] == "credit-risk-api"

    def test_default_app_label_from_name(self) -> None:
        r = self.make()
        assert r.app_label == "credit-risk-api"

    def test_custom_app_label(self) -> None:
        r = self.make(app_label="credit-risk")
        assert r.app_label == "credit-risk"

    def test_replicas_in_manifest(self) -> None:
        m = self.make(replicas=5).to_manifest()
        assert m["spec"]["replicas"] == 5

    def test_image_in_container(self) -> None:
        m = self.make(image="my-image:v1").to_manifest()
        containers = m["spec"]["template"]["spec"]["containers"]
        assert containers[0]["image"] == "my-image:v1"

    def test_default_strategy_ends_at_100(self) -> None:
        r = self.make()
        assert r.strategy.steps[-1].weight == 100
