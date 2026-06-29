"""Continuous Training automation — trigger logic, workflow spec, and run tracking.

Day 76 — CT trigger evaluation, Argo Workflow manifest builder, and CT run
result tracking for automated retrain → registry → deploy pipelines.

Classes:
  TriggerType    — enumeration of CT trigger strategies
  CTTrigger      — trigger condition with cooldown and threshold
  CTWorkflowStep — one step in a CT workflow DAG
  CTWorkflowSpec — Argo Workflow manifest builder for CT
  CTRun          — result of one completed CT run

See: docs/phase11/day76_ct_automation.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TriggerType(str, Enum):
    """Strategy used to decide when to re-train."""
    SCHEDULE = "schedule"
    DATA_DRIFT = "data_drift"
    LABEL_DRIFT = "label_drift"
    VOLUME_SPIKE = "volume_spike"
    MANUAL = "manual"


# ── CTTrigger ─────────────────────────────────────────────────────────────────

@dataclass
class CTTrigger:
    """One CT trigger condition.

    Attributes:
        trigger_type:   Strategy from TriggerType.
        threshold:      Metric value that crosses to fire the trigger.
                        Meaning varies by type:
                        DATA_DRIFT  → PSI value (e.g. 0.2)
                        LABEL_DRIFT → AUC drop below baseline (e.g. 0.03)
                        VOLUME_SPIKE → new rows since last run (e.g. 10000)
                        SCHEDULE / MANUAL → unused (set to 0.0).
        cooldown_hours: Minimum hours between consecutive CT runs.
    """

    trigger_type: TriggerType
    threshold: float = 0.0
    cooldown_hours: float = 6.0

    def __post_init__(self) -> None:
        if self.cooldown_hours < 0:
            raise ValueError("cooldown_hours must be >= 0")
        if self.trigger_type not in (TriggerType.SCHEDULE, TriggerType.MANUAL):
            if self.threshold <= 0:
                raise ValueError(
                    f"threshold must be > 0 for trigger_type={self.trigger_type.value}"
                )

    def should_trigger(self, metric_value: float) -> bool:
        """Return True if the current metric value crosses the trigger threshold.

        For SCHEDULE and MANUAL, always returns True (schedule/manual decision
        is made externally — this method just signals readiness).
        """
        if self.trigger_type in (TriggerType.SCHEDULE, TriggerType.MANUAL):
            return True
        if self.trigger_type in (TriggerType.DATA_DRIFT, TriggerType.VOLUME_SPIKE):
            return metric_value >= self.threshold
        if self.trigger_type == TriggerType.LABEL_DRIFT:
            return metric_value >= self.threshold
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_type": self.trigger_type.value,
            "threshold": self.threshold,
            "cooldown_hours": self.cooldown_hours,
        }


# ── CTWorkflowStep ────────────────────────────────────────────────────────────

@dataclass
class CTWorkflowStep:
    """One step in a CT workflow DAG.

    Attributes:
        name:         Step name (slug).
        command:      Shell command to run in the container.
        dependencies: Names of steps that must complete before this one.
        cpu_request:  CPU request (e.g. "2").
        memory_request: Memory request (e.g. "4Gi").
    """

    name: str
    command: str
    dependencies: list[str] = field(default_factory=list)
    cpu_request: str = "2"
    memory_request: str = "4Gi"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CTWorkflowStep.name cannot be empty")
        if not self.command:
            raise ValueError("CTWorkflowStep.command cannot be empty")

    def to_task_dict(self) -> dict[str, Any]:
        task: dict[str, Any] = {
            "name": self.name,
            "template": "run-step",
            "arguments": {"parameters": [{"name": "cmd", "value": self.command}]},
        }
        if self.dependencies:
            task["dependencies"] = self.dependencies
        return task


# ── CTWorkflowSpec ────────────────────────────────────────────────────────────

@dataclass
class CTWorkflowSpec:
    """Argo Workflow manifest builder for a CT pipeline.

    Attributes:
        name:        Workflow name.
        namespace:   Kubernetes namespace.
        image:       Training container image.
        mlflow_uri:  MLflow tracking server URI (injected as env var).
        data_version: Data version to train on.
        steps:       Ordered list of CTWorkflowStep objects.
        service_account: K8s service account with required RBAC.
    """

    name: str
    namespace: str = "ml-training"
    image: str = "ghcr.io/arbarikcp/credit-risk-trainer:latest"
    mlflow_uri: str = "http://mlflow.mlops-infra.svc:5000"
    data_version: str = "latest"
    steps: list[CTWorkflowStep] = field(default_factory=list)
    service_account: str = "ml-workflow-sa"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CTWorkflowSpec.name cannot be empty")
        if not self.steps:
            raise ValueError("CTWorkflowSpec.steps cannot be empty")

    def to_manifest(self) -> dict[str, Any]:
        run_step_template = {
            "name": "run-step",
            "inputs": {"parameters": [{"name": "cmd"}]},
            "container": {
                "image": self.image,
                "command": ["sh", "-c"],
                "args": ["{{inputs.parameters.cmd}}"],
                "env": [
                    {
                        "name": "MLFLOW_TRACKING_URI",
                        "valueFrom": {
                            "configMapKeyRef": {
                                "name": "ml-config",
                                "key": "MLFLOW_URI",
                            }
                        },
                    },
                    {"name": "DATA_VERSION", "value": self.data_version},
                ],
                "resources": {
                    "requests": {"cpu": "2", "memory": "4Gi"},
                    "limits": {"cpu": "4", "memory": "8Gi"},
                },
            },
        }

        pipeline_template = {
            "name": "retrain-pipeline",
            "dag": {"tasks": [s.to_task_dict() for s in self.steps]},
        }

        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "entrypoint": "retrain-pipeline",
                "serviceAccountName": self.service_account,
                "templates": [pipeline_template, run_step_template],
            },
        }

    @staticmethod
    def default_ct_pipeline(
        name: str = "ct-retrain",
        namespace: str = "ml-training",
        image: str = "ghcr.io/arbarikcp/credit-risk-trainer:latest",
    ) -> "CTWorkflowSpec":
        """Return a standard 5-step CT pipeline (validate → featurize → train → evaluate → register)."""
        steps = [
            CTWorkflowStep(
                name="data-validation",
                command="python -m data.validate --output /tmp/validate.json",
            ),
            CTWorkflowStep(
                name="feature-engineering",
                command="python -m training.features",
                dependencies=["data-validation"],
            ),
            CTWorkflowStep(
                name="train",
                command="python -m training.mlflow_train --params params.yaml",
                dependencies=["feature-engineering"],
                cpu_request="4",
                memory_request="8Gi",
            ),
            CTWorkflowStep(
                name="evaluate",
                command="python -m ci.ml_tests --check auc --threshold 0.78",
                dependencies=["train"],
            ),
            CTWorkflowStep(
                name="register-model",
                command="python -m ci.milestone1_gate --promote-if-pass",
                dependencies=["evaluate"],
            ),
        ]
        return CTWorkflowSpec(name=name, namespace=namespace, image=image, steps=steps)


# ── CTRun ─────────────────────────────────────────────────────────────────────

@dataclass
class CTRun:
    """Result of one completed CT run.

    Attributes:
        workflow_name:     Argo Workflow name.
        trigger:           Trigger that initiated this run.
        status:            "succeeded" / "failed" / "running".
        mlflow_run_id:     MLflow run ID for the training job.
        new_model_version: Registry version registered (empty if not promoted).
        auc_before:        AUC of the model before this CT run.
        auc_after:         AUC of the new model.
        promoted:          Whether the new model was promoted.
    """

    workflow_name: str
    trigger: CTTrigger
    status: str = "running"
    mlflow_run_id: str = ""
    new_model_version: str = ""
    auc_before: float = 0.0
    auc_after: float = 0.0
    promoted: bool = False

    def __post_init__(self) -> None:
        if not self.workflow_name:
            raise ValueError("CTRun.workflow_name cannot be empty")
        valid_statuses = {"running", "succeeded", "failed"}
        if self.status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")

    def improvement(self) -> float:
        """Return AUC improvement (positive = better)."""
        return self.auc_after - self.auc_before

    def is_regression(self, tolerance: float = 0.01) -> bool:
        """Return True if the new model is worse than baseline by more than tolerance."""
        return self.improvement() < -tolerance

    def summary(self) -> str:
        direction = "▲" if self.improvement() >= 0 else "▼"
        return (
            f"CTRun({self.workflow_name}): status={self.status} "
            f"AUC {self.auc_before:.3f} → {self.auc_after:.3f} "
            f"{direction}{abs(self.improvement()):.3f} promoted={self.promoted}"
        )
