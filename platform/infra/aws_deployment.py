"""End-to-end AWS deployment plan — ties together all Phase 12 components.

Day 89: Assembles S3 (DVC), ECR, SageMaker training → registry → endpoint,
Argo CD deploying to EKS, and CloudWatch monitoring into a complete
deployment specification. Models the full lifecycle as ordered stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageType(str, Enum):
    DATA = "data"
    TRAINING = "training"
    REGISTRY = "registry"
    APPROVAL = "approval"
    SERVING = "serving"
    MONITORING = "monitoring"
    GITOPS = "gitops"
    VALIDATION = "validation"


# ── Deployment Stage ───────────────────────────────────────────────────────────


@dataclass
class DeploymentStage:
    """A single stage in the end-to-end AWS deployment plan.

    Each stage has a type, the AWS service it uses, prerequisite stages,
    and a configuration dict (from the relevant Phase 12 builder).
    """

    stage_name: str
    stage_type: StageType
    aws_service: str  # e.g. "S3", "SageMaker", "EKS", "ECR"
    description: str
    config: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    outputs: dict[str, str] = field(default_factory=dict)
    rollback_action: str = ""

    def __post_init__(self) -> None:
        if not self.stage_name:
            raise ValueError("stage_name must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")
        if not self.aws_service:
            raise ValueError("aws_service must not be empty")

    def mark_complete(self, outputs: dict[str, str] | None = None) -> "DeploymentStage":
        self.status = StageStatus.COMPLETED
        if outputs:
            self.outputs.update(outputs)
        return self

    def mark_failed(self, reason: str = "") -> "DeploymentStage":
        self.status = StageStatus.FAILED
        if reason:
            self.outputs["failure_reason"] = reason
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "stageName": self.stage_name,
            "stageType": self.stage_type.value,
            "awsService": self.aws_service,
            "description": self.description,
            "dependsOn": self.depends_on,
            "status": self.status.value,
            "config": self.config,
            "outputs": self.outputs,
            "rollbackAction": self.rollback_action,
        }


# ── Deployment Report ──────────────────────────────────────────────────────────


@dataclass
class DeploymentReport:
    """Report of a completed (or in-progress) deployment plan execution."""

    plan_name: str
    environment: str
    stages_completed: int
    stages_failed: int
    stages_total: int
    endpoint_url: str = ""
    model_package_arn: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.plan_name:
            raise ValueError("plan_name must not be empty")
        if not self.environment:
            raise ValueError("environment must not be empty")

    @property
    def success_rate(self) -> float:
        if self.stages_total == 0:
            return 0.0
        return self.stages_completed / self.stages_total

    @property
    def is_success(self) -> bool:
        return self.stages_failed == 0 and self.stages_completed == self.stages_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "planName": self.plan_name,
            "environment": self.environment,
            "stagesCompleted": self.stages_completed,
            "stagesFailed": self.stages_failed,
            "stagesTotal": self.stages_total,
            "successRate": round(self.success_rate, 2),
            "isSuccess": self.is_success,
            "endpointUrl": self.endpoint_url,
            "modelPackageArn": self.model_package_arn,
            "artifacts": self.artifacts,
            "issues": self.issues,
        }


# ── Deployment Plan ────────────────────────────────────────────────────────────


@dataclass
class AWSDeploymentPlan:
    """End-to-end AWS ML deployment plan.

    Assembles all Phase 12 components into an ordered, dependency-aware
    deployment specification. The plan can be serialised to a dict for
    storage in DynamoDB or displayed in a CI/CD dashboard.
    """

    plan_name: str
    environment: str  # "dev" | "staging" | "prod"
    account_id: str
    region: str
    stages: list[DeploymentStage] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plan_name:
            raise ValueError("plan_name must not be empty")
        if not self.environment:
            raise ValueError("environment must not be empty")
        if not self.account_id:
            raise ValueError("account_id must not be empty")
        if not self.region:
            raise ValueError("region must not be empty")
        if self.environment not in ("dev", "staging", "prod"):
            raise ValueError(f"environment must be dev/staging/prod, got: {self.environment!r}")

    def add_stage(self, stage: DeploymentStage) -> "AWSDeploymentPlan":
        self.stages.append(stage)
        return self

    def get_stage(self, stage_name: str) -> DeploymentStage | None:
        return next((s for s in self.stages if s.stage_name == stage_name), None)

    def execution_order(self) -> list[str]:
        """Topological sort of stage names honouring depends_on."""
        graph: dict[str, list[str]] = {s.stage_name: s.depends_on for s in self.stages}
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

    def generate_report(self) -> DeploymentReport:
        """Summarise current state of all stages into a DeploymentReport."""
        completed = sum(1 for s in self.stages if s.status == StageStatus.COMPLETED)
        failed = sum(1 for s in self.stages if s.status == StageStatus.FAILED)

        # Collect artifacts from stage outputs
        artifacts: dict[str, str] = {}
        endpoint_url = ""
        model_pkg_arn = ""
        for stage in self.stages:
            artifacts.update(stage.outputs)
            if "endpoint_url" in stage.outputs:
                endpoint_url = stage.outputs["endpoint_url"]
            if "model_package_arn" in stage.outputs:
                model_pkg_arn = stage.outputs["model_package_arn"]

        issues = [
            f"Stage '{s.stage_name}' failed: {s.outputs.get('failure_reason', 'unknown')}"
            for s in self.stages
            if s.status == StageStatus.FAILED
        ]
        return DeploymentReport(
            plan_name=self.plan_name,
            environment=self.environment,
            stages_completed=completed,
            stages_failed=failed,
            stages_total=len(self.stages),
            endpoint_url=endpoint_url,
            model_package_arn=model_pkg_arn,
            artifacts=artifacts,
            issues=issues,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "planName": self.plan_name,
            "environment": self.environment,
            "accountId": self.account_id,
            "region": self.region,
            "executionOrder": self.execution_order(),
            "stages": [s.to_dict() for s in self.stages],
            "tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }

    @classmethod
    def credit_risk_full_deploy(
        cls,
        account_id: str,
        region: str = "us-east-1",
        environment: str = "prod",
    ) -> "AWSDeploymentPlan":
        """Factory: complete credit-risk model deployment plan on AWS.

        Stages:
          1. data-prep      — DVC pull from S3
          2. build-image    — ECR build + push
          3. train          — SageMaker training job (spot)
          4. register       — SageMaker model package
          5. approve        — Human approval gate
          6. endpoint       — SageMaker real-time endpoint
          7. monitor        — SageMaker Model Monitor
          8. gitops         — Argo CD sync for EKS services
        """
        bucket = f"mlops-artifacts-{account_id}"
        ecr_base = f"{account_id}.dkr.ecr.{region}.amazonaws.com"

        plan = cls(
            plan_name=f"credit-risk-deploy-{environment}",
            environment=environment,
            account_id=account_id,
            region=region,
            tags={"Project": "credit-risk", "Phase": "12", "Env": environment},
        )

        plan.add_stage(DeploymentStage(
            stage_name="data-prep",
            stage_type=StageType.DATA,
            aws_service="S3",
            description="DVC pull training data from S3 and validate schema",
            config={
                "dvc_remote": f"s3://{bucket}/dvc-cache",
                "data_path": "data/credit_risk/v3/",
                "pandera_schema": "platform/data/schema.py",
            },
            rollback_action="No rollback needed — read-only",
        ))

        plan.add_stage(DeploymentStage(
            stage_name="build-image",
            stage_type=StageType.TRAINING,
            aws_service="ECR",
            description="Build training container and push to ECR",
            config={
                "dockerfile": "platform/Dockerfile.training",
                "ecr_uri": f"{ecr_base}/credit-risk-training:latest",
                "lifecycle_policy": "keep_last_10_tagged",
            },
            depends_on=["data-prep"],
            rollback_action="Retag previous :latest image",
        ))

        plan.add_stage(DeploymentStage(
            stage_name="train",
            stage_type=StageType.TRAINING,
            aws_service="SageMaker",
            description="SageMaker training job with spot instances",
            config={
                "instance_type": "ml.m5.xlarge",
                "use_spot": True,
                "output_s3": f"s3://{bucket}/models/credit-risk/",
                "hyperparameters": {"n_estimators": "200", "max_depth": "6"},
            },
            depends_on=["build-image"],
            rollback_action="No rollback — immutable artifact",
        ))

        plan.add_stage(DeploymentStage(
            stage_name="register",
            stage_type=StageType.REGISTRY,
            aws_service="SageMaker",
            description="Register model package in SageMaker Model Registry",
            config={
                "model_package_group": "credit-risk-models",
                "approval_status": "PendingManualApproval",
                "quality_threshold_auc": 0.78,
            },
            depends_on=["train"],
        ))

        plan.add_stage(DeploymentStage(
            stage_name="approve",
            stage_type=StageType.APPROVAL,
            aws_service="SageMaker",
            description="Human or automated model quality gate approval",
            config={
                "reviewer": "automated-ci",
                "thresholds": {"auc": 0.78, "psi_max": 0.2},
            },
            depends_on=["register"],
            rollback_action="Reject model package — blocks deploy",
        ))

        plan.add_stage(DeploymentStage(
            stage_name="endpoint",
            stage_type=StageType.SERVING,
            aws_service="SageMaker",
            description="Deploy approved model to SageMaker real-time endpoint",
            config={
                "endpoint_name": f"credit-risk-{environment}",
                "instance_type": "ml.m5.large",
                "data_capture": f"s3://{bucket}/data-capture/",
            },
            depends_on=["approve"],
            rollback_action="Swap EndpointConfig back to previous version",
        ))

        plan.add_stage(DeploymentStage(
            stage_name="monitor",
            stage_type=StageType.MONITORING,
            aws_service="SageMaker",
            description="Enable data quality and model quality monitors",
            config={
                "schedule": "daily",
                "baseline_s3": f"s3://{bucket}/baselines/credit-risk/",
                "output_s3": f"s3://{bucket}/monitors/credit-risk/",
                "cloudwatch_namespace": "mlops/credit-risk",
            },
            depends_on=["endpoint"],
        ))

        plan.add_stage(DeploymentStage(
            stage_name="gitops",
            stage_type=StageType.GITOPS,
            aws_service="EKS",
            description="Argo CD sync — deploy supporting services to EKS",
            config={
                "argo_app": "credit-risk-serving",
                "helm_chart": "platform/infra/helm/credit-risk",
                "target_revision": "HEAD",
                "sync_policy": "automated",
            },
            depends_on=["endpoint"],
        ))

        return plan
