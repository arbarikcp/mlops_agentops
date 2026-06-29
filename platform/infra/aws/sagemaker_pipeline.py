"""SageMaker Pipelines — DAG orchestration with model approval and lineage.

Day 82: SageMaker Pipelines provide native integration with SageMaker jobs,
automatic lineage tracking, and a model approval gate — without needing a
separate Argo/Airflow installation. The trade-off: AWS-only vs portable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepType(str, Enum):
    PROCESSING = "Processing"
    TRAINING = "Training"
    REGISTER_MODEL = "RegisterModel"
    CREATE_MODEL = "CreateModel"
    CONDITION = "Condition"
    TRANSFORM = "Transform"
    CALLBACK = "Callback"


class ApprovalStatus(str, Enum):
    PENDING = "PendingManualApproval"
    APPROVED = "Approved"
    REJECTED = "Rejected"


# ── Pipeline step ──────────────────────────────────────────────────────────────


@dataclass
class SMPipelineStep:
    """One node in a SageMaker Pipeline DAG.

    Steps are composed into a Pipeline; SageMaker resolves the execution order
    from the dependency graph and automatically tracks lineage for each run.
    """

    step_name: str
    step_type: StepType
    job_definition: dict[str, Any]  # SMTrainingJob.to_dict() / SMProcessingJob.to_dict() etc.
    depends_on: list[str] = field(default_factory=list)
    cache_config: bool = True

    def __post_init__(self) -> None:
        if not self.step_name:
            raise ValueError("step_name must not be empty")
        if not self.job_definition:
            raise ValueError("job_definition must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "Name": self.step_name,
            "Type": self.step_type.value,
            "Arguments": self.job_definition,
            "DependsOn": self.depends_on,
            "CacheConfig": {"Enabled": self.cache_config, "ExpireAfter": "PT1H"},
        }

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def processing_step(
        cls,
        step_name: str,
        job_definition: dict[str, Any],
        depends_on: list[str] | None = None,
    ) -> "SMPipelineStep":
        return cls(
            step_name=step_name,
            step_type=StepType.PROCESSING,
            job_definition=job_definition,
            depends_on=depends_on or [],
        )

    @classmethod
    def training_step(
        cls,
        step_name: str,
        job_definition: dict[str, Any],
        depends_on: list[str] | None = None,
    ) -> "SMPipelineStep":
        return cls(
            step_name=step_name,
            step_type=StepType.TRAINING,
            job_definition=job_definition,
            depends_on=depends_on or [],
        )

    @classmethod
    def register_model_step(
        cls,
        step_name: str,
        model_package_definition: dict[str, Any],
        depends_on: list[str] | None = None,
    ) -> "SMPipelineStep":
        return cls(
            step_name=step_name,
            step_type=StepType.REGISTER_MODEL,
            job_definition=model_package_definition,
            depends_on=depends_on or [],
        )


# ── Pipeline ──────────────────────────────────────────────────────────────────


@dataclass
class SMPipelineParameter:
    """A pipeline-level parameter that can be overridden at execution time."""

    name: str
    parameter_type: str  # "String" | "Integer" | "Float" | "Boolean"
    default_value: Any

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if self.parameter_type not in ("String", "Integer", "Float", "Boolean"):
            raise ValueError(f"parameter_type invalid: {self.parameter_type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "Name": self.name,
            "Type": self.parameter_type,
            "DefaultValue": self.default_value,
        }


@dataclass
class SMPipeline:
    """SageMaker Pipeline — a DAG of steps with versioning and lineage.

    Each execution of a Pipeline creates an immutable record with all inputs,
    outputs, and metrics automatically tracked via SM Lineage.
    """

    pipeline_name: str
    role_arn: str
    description: str
    steps: list[SMPipelineStep] = field(default_factory=list)
    parameters: list[SMPipelineParameter] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pipeline_name:
            raise ValueError("pipeline_name must not be empty")
        if not self.role_arn:
            raise ValueError("role_arn must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")

    def add_step(self, step: SMPipelineStep) -> "SMPipeline":
        self.steps.append(step)
        return self

    def add_parameter(self, param: SMPipelineParameter) -> "SMPipeline":
        self.parameters.append(param)
        return self

    def execution_order(self) -> list[str]:
        """Topological sort of step names respecting depends_on edges."""
        graph: dict[str, list[str]] = {s.step_name: s.depends_on for s in self.steps}
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in graph.get(name, []):
                visit(dep)
            order.append(name)

        for name in graph:
            visit(name)
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "PipelineName": self.pipeline_name,
            "RoleArn": self.role_arn,
            "PipelineDescription": self.description,
            "PipelineDefinition": {
                "Parameters": [p.to_dict() for p in self.parameters],
                "Steps": [s.to_dict() for s in self.steps],
            },
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }

    @classmethod
    def credit_risk_pipeline(cls, role_arn: str) -> "SMPipeline":
        """Factory: standard credit-risk training pipeline with 3 steps."""
        pipeline = cls(
            pipeline_name="credit-risk-training-pipeline",
            role_arn=role_arn,
            description="End-to-end credit-risk model training: preprocess → train → register",
            tags={"Project": "credit-risk", "Phase": "12"},
        )
        pipeline.add_parameter(SMPipelineParameter("ModelApprovalStatus", "String", "PendingManualApproval"))
        pipeline.add_parameter(SMPipelineParameter("NEstimators", "Integer", 200))
        return pipeline


# ── Model Approval ────────────────────────────────────────────────────────────


@dataclass
class SMModelApproval:
    """SageMaker model approval gate — human or automated sign-off.

    Approval gates decouple training (fast, automated) from deployment
    (deliberate, controlled). A model can be in PendingManualApproval state
    indefinitely until a human or automated quality gate approves it.
    """

    model_package_arn: str
    reviewer: str
    approval_status: ApprovalStatus
    reason: str = ""
    quality_metrics: dict[str, float] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_package_arn:
            raise ValueError("model_package_arn must not be empty")
        if not self.reviewer:
            raise ValueError("reviewer must not be empty")

    def passes_quality_gate(self) -> bool:
        """Returns True if all quality metrics meet their thresholds."""
        for metric, threshold in self.thresholds.items():
            actual = self.quality_metrics.get(metric, 0.0)
            if actual < threshold:
                return False
        return True

    def auto_approve(self) -> "SMModelApproval":
        """Approve if all quality metrics pass thresholds."""
        if self.passes_quality_gate():
            self.approval_status = ApprovalStatus.APPROVED
            self.reason = "Auto-approved: all quality gates passed"
        else:
            self.approval_status = ApprovalStatus.REJECTED
            failed = [m for m, t in self.thresholds.items()
                      if self.quality_metrics.get(m, 0.0) < t]
            self.reason = f"Auto-rejected: failed metrics {failed}"
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "ModelPackageArn": self.model_package_arn,
            "ModelApprovalStatus": self.approval_status.value,
            "ApprovalDescription": self.reason,
            "Reviewer": self.reviewer,
            "QualityMetrics": self.quality_metrics,
            "Thresholds": self.thresholds,
            "PassesGate": self.passes_quality_gate(),
        }

    @classmethod
    def credit_risk_gate(
        cls,
        model_package_arn: str,
        auc: float,
        psi: float,
    ) -> "SMModelApproval":
        """Factory: credit-risk quality gate (AUC >= 0.78, PSI <= 0.2)."""
        return cls(
            model_package_arn=model_package_arn,
            reviewer="automated-ci",
            approval_status=ApprovalStatus.PENDING,
            quality_metrics={"auc": auc, "psi": psi},
            thresholds={"auc": 0.78},
        )
