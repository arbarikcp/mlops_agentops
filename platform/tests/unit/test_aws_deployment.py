"""Unit tests for infra.aws_deployment (Day 89)."""

import pytest

from infra.aws_deployment import (
    AWSDeploymentPlan,
    DeploymentStage,
    DeploymentReport,
    StageStatus,
    StageType,
)


# ── DeploymentStage ───────────────────────────────────────────────────────────

class TestDeploymentStage:
    def _make(self, **kwargs):
        defaults = dict(
            stage_name="train",
            stage_type=StageType.TRAINING,
            aws_service="SageMaker",
            description="Run training job",
            config={"instance_type": "ml.m5.xlarge"},
        )
        defaults.update(kwargs)
        return DeploymentStage(**defaults)

    def test_empty_stage_name_raises(self):
        with pytest.raises(ValueError, match="stage_name"):
            self._make(stage_name="")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            self._make(description="")

    def test_empty_aws_service_raises(self):
        with pytest.raises(ValueError, match="aws_service"):
            self._make(aws_service="")

    def test_default_status_pending(self):
        stage = self._make()
        assert stage.status == StageStatus.PENDING

    def test_mark_complete(self):
        stage = self._make()
        result = stage.mark_complete({"model_s3": "s3://b/model.tar.gz"})
        assert result is stage
        assert stage.status == StageStatus.COMPLETED
        assert stage.outputs["model_s3"] == "s3://b/model.tar.gz"

    def test_mark_failed(self):
        stage = self._make()
        stage.mark_failed("OOM error")
        assert stage.status == StageStatus.FAILED
        assert stage.outputs["failure_reason"] == "OOM error"

    def test_to_dict_structure(self):
        stage = self._make(depends_on=["data-prep"])
        d = stage.to_dict()
        assert d["stageName"] == "train"
        assert d["stageType"] == "training"
        assert "data-prep" in d["dependsOn"]
        assert d["status"] == "pending"

    def test_config_in_dict(self):
        stage = self._make(config={"instance_type": "ml.g4dn.xlarge", "use_spot": True})
        d = stage.to_dict()
        assert d["config"]["instance_type"] == "ml.g4dn.xlarge"


# ── DeploymentReport ──────────────────────────────────────────────────────────

class TestDeploymentReport:
    def test_empty_plan_name_raises(self):
        with pytest.raises(ValueError, match="plan_name"):
            DeploymentReport("", "prod", 5, 0, 5)

    def test_empty_environment_raises(self):
        with pytest.raises(ValueError, match="environment"):
            DeploymentReport("plan", "", 5, 0, 5)

    def test_success_rate_calculation(self):
        r = DeploymentReport("plan", "prod", 7, 0, 8)
        assert abs(r.success_rate - 7/8) < 0.001

    def test_zero_total_success_rate(self):
        r = DeploymentReport("plan", "prod", 0, 0, 0)
        assert r.success_rate == 0.0

    def test_is_success_true(self):
        r = DeploymentReport("plan", "prod", 5, 0, 5)
        assert r.is_success is True

    def test_is_success_false_when_failed(self):
        r = DeploymentReport("plan", "prod", 4, 1, 5)
        assert r.is_success is False

    def test_is_success_false_when_incomplete(self):
        r = DeploymentReport("plan", "prod", 3, 0, 5)
        assert r.is_success is False

    def test_to_dict_structure(self):
        r = DeploymentReport("plan", "prod", 5, 0, 5, endpoint_url="https://ep.example.com")
        d = r.to_dict()
        assert d["isSuccess"] is True
        assert d["endpointUrl"] == "https://ep.example.com"
        assert d["successRate"] == 1.0


# ── AWSDeploymentPlan ──────────────────────────────────────────────────────────

class TestAWSDeploymentPlan:
    def _make(self, **kwargs):
        defaults = dict(
            plan_name="credit-risk-deploy",
            environment="prod",
            account_id="123456789012",
            region="us-east-1",
        )
        defaults.update(kwargs)
        return AWSDeploymentPlan(**defaults)

    def test_empty_plan_name_raises(self):
        with pytest.raises(ValueError, match="plan_name"):
            self._make(plan_name="")

    def test_empty_environment_raises(self):
        with pytest.raises(ValueError, match="environment"):
            self._make(environment="")

    def test_empty_account_id_raises(self):
        with pytest.raises(ValueError, match="account_id"):
            self._make(account_id="")

    def test_empty_region_raises(self):
        with pytest.raises(ValueError, match="region"):
            self._make(region="")

    def test_invalid_environment_raises(self):
        with pytest.raises(ValueError, match="environment"):
            self._make(environment="uat")

    def test_valid_environments(self):
        for env in ("dev", "staging", "prod"):
            plan = self._make(environment=env)
            assert plan.environment == env

    def test_add_stage_chaining(self):
        plan = self._make()
        stage = DeploymentStage("s1", StageType.DATA, "S3", "desc", {})
        result = plan.add_stage(stage)
        assert result is plan
        assert len(plan.stages) == 1

    def test_get_stage(self):
        plan = self._make()
        stage = DeploymentStage("train", StageType.TRAINING, "SageMaker", "desc", {})
        plan.add_stage(stage)
        found = plan.get_stage("train")
        assert found is stage

    def test_get_stage_not_found(self):
        plan = self._make()
        assert plan.get_stage("nonexistent") is None

    def test_execution_order_respects_deps(self):
        plan = self._make()
        plan.add_stage(DeploymentStage("train", StageType.TRAINING, "SM", "desc", {}, depends_on=["data-prep"]))
        plan.add_stage(DeploymentStage("data-prep", StageType.DATA, "S3", "desc", {}))
        order = plan.execution_order()
        assert order.index("data-prep") < order.index("train")

    def test_to_dict_structure(self):
        plan = self._make()
        plan.add_stage(DeploymentStage("s1", StageType.DATA, "S3", "desc", {}))
        d = plan.to_dict()
        assert d["planName"] == "credit-risk-deploy"
        assert len(d["stages"]) == 1
        assert "executionOrder" in d

    def test_generate_report_all_pending(self):
        plan = self._make()
        plan.add_stage(DeploymentStage("s1", StageType.DATA, "S3", "desc", {}))
        plan.add_stage(DeploymentStage("s2", StageType.TRAINING, "SM", "desc", {}))
        report = plan.generate_report()
        assert report.stages_total == 2
        assert report.stages_completed == 0
        assert report.is_success is False

    def test_generate_report_all_complete(self):
        plan = self._make()
        for i in range(3):
            stage = DeploymentStage(f"s{i}", StageType.DATA, "S3", "desc", {})
            stage.mark_complete()
            plan.add_stage(stage)
        report = plan.generate_report()
        assert report.is_success is True

    def test_generate_report_collects_outputs(self):
        plan = self._make()
        stage = DeploymentStage("endpoint", StageType.SERVING, "SM", "desc", {})
        stage.mark_complete({"endpoint_url": "https://runtime.sagemaker.amazonaws.com/endpoint"})
        plan.add_stage(stage)
        report = plan.generate_report()
        assert report.endpoint_url == "https://runtime.sagemaker.amazonaws.com/endpoint"

    def test_generate_report_issues_from_failed(self):
        plan = self._make()
        stage = DeploymentStage("train", StageType.TRAINING, "SM", "desc", {})
        stage.mark_failed("OOM")
        plan.add_stage(stage)
        report = plan.generate_report()
        assert len(report.issues) == 1
        assert "OOM" in report.issues[0]

    def test_credit_risk_full_deploy_factory(self):
        plan = AWSDeploymentPlan.credit_risk_full_deploy("123456789012")
        assert len(plan.stages) == 8
        stage_names = [s.stage_name for s in plan.stages]
        for expected in ["data-prep", "build-image", "train", "register", "approve", "endpoint", "monitor", "gitops"]:
            assert expected in stage_names

    def test_credit_risk_execution_order(self):
        plan = AWSDeploymentPlan.credit_risk_full_deploy("123456789012")
        order = plan.execution_order()
        # train must come after build-image
        assert order.index("build-image") < order.index("train")
        # approve must come after register
        assert order.index("register") < order.index("approve")
        # endpoint must come after approve
        assert order.index("approve") < order.index("endpoint")
