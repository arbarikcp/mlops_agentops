"""Tests for infra/ct_automation.py — CTTrigger, CTWorkflowSpec, CTRun."""
from __future__ import annotations

import pytest

from infra.ct_automation import (
    CTRun,
    CTTrigger,
    CTWorkflowSpec,
    CTWorkflowStep,
    TriggerType,
)


def make_trigger(**kwargs) -> CTTrigger:
    defaults = dict(trigger_type=TriggerType.DATA_DRIFT, threshold=0.2)
    defaults.update(kwargs)
    return CTTrigger(**defaults)


# ── TriggerType ───────────────────────────────────────────────────────────────

class TestTriggerType:
    def test_all_values(self) -> None:
        values = {t.value for t in TriggerType}
        assert "schedule" in values
        assert "data_drift" in values
        assert "manual" in values

    def test_is_str_enum(self) -> None:
        assert TriggerType.DATA_DRIFT == "data_drift"


# ── CTTrigger ─────────────────────────────────────────────────────────────────

class TestCTTrigger:
    def test_valid_drift_trigger(self) -> None:
        t = make_trigger()
        assert t.trigger_type == TriggerType.DATA_DRIFT

    def test_negative_cooldown_raises(self) -> None:
        with pytest.raises(ValueError, match="cooldown_hours"):
            make_trigger(cooldown_hours=-1)

    def test_zero_threshold_for_drift_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            make_trigger(trigger_type=TriggerType.DATA_DRIFT, threshold=0.0)

    def test_zero_threshold_ok_for_schedule(self) -> None:
        t = CTTrigger(trigger_type=TriggerType.SCHEDULE, threshold=0.0)
        assert t.trigger_type == TriggerType.SCHEDULE

    def test_should_trigger_drift_above_threshold(self) -> None:
        t = make_trigger(threshold=0.2)
        assert t.should_trigger(0.25) is True

    def test_should_trigger_drift_below_threshold(self) -> None:
        t = make_trigger(threshold=0.2)
        assert t.should_trigger(0.1) is False

    def test_should_trigger_at_threshold(self) -> None:
        t = make_trigger(threshold=0.2)
        assert t.should_trigger(0.2) is True

    def test_should_trigger_schedule_always_true(self) -> None:
        t = CTTrigger(trigger_type=TriggerType.SCHEDULE)
        assert t.should_trigger(0.0) is True

    def test_should_trigger_manual_always_true(self) -> None:
        t = CTTrigger(trigger_type=TriggerType.MANUAL)
        assert t.should_trigger(0.0) is True

    def test_to_dict_keys(self) -> None:
        d = make_trigger().to_dict()
        assert "trigger_type" in d
        assert "threshold" in d
        assert "cooldown_hours" in d

    def test_to_dict_type_is_string(self) -> None:
        d = make_trigger(trigger_type=TriggerType.LABEL_DRIFT, threshold=0.03).to_dict()
        assert d["trigger_type"] == "label_drift"

    def test_volume_spike_trigger(self) -> None:
        t = CTTrigger(trigger_type=TriggerType.VOLUME_SPIKE, threshold=10000.0)
        assert t.should_trigger(15000.0) is True
        assert t.should_trigger(5000.0) is False


# ── CTWorkflowStep ────────────────────────────────────────────────────────────

class TestCTWorkflowStep:
    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            CTWorkflowStep(name="", command="python foo.py")

    def test_empty_command_raises(self) -> None:
        with pytest.raises(ValueError, match="command"):
            CTWorkflowStep(name="step", command="")

    def test_to_task_dict_structure(self) -> None:
        step = CTWorkflowStep(name="train", command="python -m training.train")
        d = step.to_task_dict()
        assert d["name"] == "train"
        assert d["template"] == "run-step"
        assert d["arguments"]["parameters"][0]["value"] == "python -m training.train"

    def test_to_task_dict_no_deps_when_empty(self) -> None:
        step = CTWorkflowStep(name="train", command="python -m training.train")
        d = step.to_task_dict()
        assert "dependencies" not in d

    def test_to_task_dict_with_deps(self) -> None:
        step = CTWorkflowStep(name="train", command="cmd", dependencies=["validate"])
        d = step.to_task_dict()
        assert d["dependencies"] == ["validate"]


# ── CTWorkflowSpec ────────────────────────────────────────────────────────────

class TestCTWorkflowSpec:
    def make(self, **kwargs) -> CTWorkflowSpec:
        steps = [CTWorkflowStep("step1", "python foo.py")]
        defaults = dict(name="ct-retrain", steps=steps)
        defaults.update(kwargs)
        return CTWorkflowSpec(**defaults)

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            self.make(name="")

    def test_empty_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="steps"):
            self.make(steps=[])

    def test_to_manifest_kind(self) -> None:
        m = self.make().to_manifest()
        assert m["kind"] == "Workflow"
        assert m["apiVersion"] == "argoproj.io/v1alpha1"

    def test_to_manifest_entrypoint(self) -> None:
        m = self.make().to_manifest()
        assert m["spec"]["entrypoint"] == "retrain-pipeline"

    def test_to_manifest_templates(self) -> None:
        m = self.make().to_manifest()
        template_names = [t["name"] for t in m["spec"]["templates"]]
        assert "retrain-pipeline" in template_names
        assert "run-step" in template_names

    def test_to_manifest_service_account(self) -> None:
        m = self.make(service_account="custom-sa").to_manifest()
        assert m["spec"]["serviceAccountName"] == "custom-sa"

    def test_default_ct_pipeline_factory(self) -> None:
        spec = CTWorkflowSpec.default_ct_pipeline()
        assert spec.name == "ct-retrain"
        assert len(spec.steps) == 5

    def test_default_pipeline_step_names(self) -> None:
        spec = CTWorkflowSpec.default_ct_pipeline()
        names = [s.name for s in spec.steps]
        assert "data-validation" in names
        assert "train" in names
        assert "register-model" in names

    def test_default_pipeline_dependencies_set(self) -> None:
        spec = CTWorkflowSpec.default_ct_pipeline()
        train_step = next(s for s in spec.steps if s.name == "train")
        assert "feature-engineering" in train_step.dependencies

    def test_mlflow_uri_in_manifest_env(self) -> None:
        spec = self.make(mlflow_uri="http://mlflow:5000")
        m = spec.to_manifest()
        run_template = next(t for t in m["spec"]["templates"] if t["name"] == "run-step")
        env_names = [e["name"] for e in run_template["container"]["env"]]
        assert "MLFLOW_TRACKING_URI" in env_names


# ── CTRun ─────────────────────────────────────────────────────────────────────

class TestCTRun:
    def make(self, **kwargs) -> CTRun:
        defaults = dict(
            workflow_name="ct-retrain-001",
            trigger=make_trigger(),
            status="succeeded",
            auc_before=0.79,
            auc_after=0.82,
        )
        defaults.update(kwargs)
        return CTRun(**defaults)

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="workflow_name"):
            self.make(workflow_name="")

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError, match="status"):
            self.make(status="pending")

    def test_valid_statuses(self) -> None:
        for s in ("running", "succeeded", "failed"):
            r = self.make(status=s)
            assert r.status == s

    def test_improvement_positive(self) -> None:
        r = self.make(auc_before=0.79, auc_after=0.82)
        assert r.improvement() == pytest.approx(0.03)

    def test_improvement_negative(self) -> None:
        r = self.make(auc_before=0.82, auc_after=0.79)
        assert r.improvement() == pytest.approx(-0.03)

    def test_is_regression_true(self) -> None:
        r = self.make(auc_before=0.82, auc_after=0.79)
        assert r.is_regression(tolerance=0.01) is True

    def test_is_regression_false_within_tolerance(self) -> None:
        r = self.make(auc_before=0.82, auc_after=0.815)
        assert r.is_regression(tolerance=0.01) is False

    def test_is_regression_false_improvement(self) -> None:
        r = self.make(auc_before=0.79, auc_after=0.82)
        assert r.is_regression() is False

    def test_summary_contains_status(self) -> None:
        r = self.make(status="succeeded")
        assert "succeeded" in r.summary()

    def test_summary_contains_auc_values(self) -> None:
        r = self.make(auc_before=0.790, auc_after=0.820)
        s = r.summary()
        assert "0.790" in s
        assert "0.820" in s

    def test_summary_improvement_arrow(self) -> None:
        r_up = self.make(auc_before=0.79, auc_after=0.82)
        r_down = self.make(auc_before=0.82, auc_after=0.79)
        assert "▲" in r_up.summary()
        assert "▼" in r_down.summary()
