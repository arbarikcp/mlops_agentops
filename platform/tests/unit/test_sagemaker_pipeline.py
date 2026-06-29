"""Unit tests for infra.aws.sagemaker_pipeline (Day 82)."""

import pytest

from infra.aws.sagemaker_pipeline import (
    SMPipelineStep,
    SMPipeline,
    SMPipelineParameter,
    SMModelApproval,
    StepType,
    ApprovalStatus,
)

ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"
JOB_DEF = {"TrainingJobName": "test-job", "RoleArn": ROLE}


# ── SMPipelineStep ────────────────────────────────────────────────────────────

class TestSMPipelineStep:
    def test_empty_step_name_raises(self):
        with pytest.raises(ValueError, match="step_name"):
            SMPipelineStep("", StepType.TRAINING, JOB_DEF)

    def test_empty_job_definition_raises(self):
        with pytest.raises(ValueError, match="job_definition"):
            SMPipelineStep("step1", StepType.TRAINING, {})

    def test_to_dict_structure(self):
        step = SMPipelineStep("train-step", StepType.TRAINING, JOB_DEF)
        d = step.to_dict()
        assert d["Name"] == "train-step"
        assert d["Type"] == "Training"
        assert d["CacheConfig"]["Enabled"] is True

    def test_depends_on_in_dict(self):
        step = SMPipelineStep("step2", StepType.PROCESSING, JOB_DEF, depends_on=["step1"])
        d = step.to_dict()
        assert "step1" in d["DependsOn"]

    def test_processing_step_factory(self):
        step = SMPipelineStep.processing_step("preprocess", JOB_DEF)
        assert step.step_type == StepType.PROCESSING

    def test_training_step_factory(self):
        step = SMPipelineStep.training_step("train", JOB_DEF, depends_on=["preprocess"])
        assert step.step_type == StepType.TRAINING
        assert "preprocess" in step.depends_on

    def test_register_model_step_factory(self):
        step = SMPipelineStep.register_model_step("register", JOB_DEF)
        assert step.step_type == StepType.REGISTER_MODEL


# ── SMPipelineParameter ───────────────────────────────────────────────────────

class TestSMPipelineParameter:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            SMPipelineParameter("", "String", "default")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="parameter_type"):
            SMPipelineParameter("p", "Dict", "x")

    def test_to_dict(self):
        p = SMPipelineParameter("NEstimators", "Integer", 200)
        d = p.to_dict()
        assert d["Name"] == "NEstimators"
        assert d["Type"] == "Integer"
        assert d["DefaultValue"] == 200


# ── SMPipeline ────────────────────────────────────────────────────────────────

class TestSMPipeline:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="pipeline_name"):
            SMPipeline("", ROLE, "desc")

    def test_empty_role_raises(self):
        with pytest.raises(ValueError, match="role_arn"):
            SMPipeline("pipeline", "", "desc")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            SMPipeline("pipeline", ROLE, "")

    def test_add_step_chaining(self):
        pipeline = SMPipeline("p", ROLE, "desc")
        step = SMPipelineStep.training_step("train", JOB_DEF)
        result = pipeline.add_step(step)
        assert result is pipeline
        assert len(pipeline.steps) == 1

    def test_to_dict_structure(self):
        pipeline = SMPipeline("p", ROLE, "desc")
        pipeline.add_step(SMPipelineStep.training_step("train", JOB_DEF))
        d = pipeline.to_dict()
        assert d["PipelineName"] == "p"
        assert len(d["PipelineDefinition"]["Steps"]) == 1

    def test_execution_order_respects_deps(self):
        pipeline = SMPipeline("p", ROLE, "desc")
        pipeline.add_step(SMPipelineStep.training_step("train", JOB_DEF, depends_on=["preprocess"]))
        pipeline.add_step(SMPipelineStep.processing_step("preprocess", JOB_DEF))
        order = pipeline.execution_order()
        assert order.index("preprocess") < order.index("train")

    def test_credit_risk_pipeline_factory(self):
        pipeline = SMPipeline.credit_risk_pipeline(ROLE)
        d = pipeline.to_dict()
        assert "credit-risk" in d["PipelineName"]
        assert len(d["PipelineDefinition"]["Parameters"]) >= 2

    def test_tags_in_dict(self):
        pipeline = SMPipeline("p", ROLE, "desc", tags={"Env": "prod"})
        d = pipeline.to_dict()
        assert any(t["Key"] == "Env" for t in d["Tags"])


# ── SMModelApproval ───────────────────────────────────────────────────────────

class TestSMModelApproval:
    PKG_ARN = "arn:aws:sagemaker:us-east-1:123:model-package/credit-risk-models/1"

    def test_empty_arn_raises(self):
        with pytest.raises(ValueError, match="model_package_arn"):
            SMModelApproval("", "reviewer", ApprovalStatus.PENDING)

    def test_empty_reviewer_raises(self):
        with pytest.raises(ValueError, match="reviewer"):
            SMModelApproval(self.PKG_ARN, "", ApprovalStatus.PENDING)

    def test_passes_quality_gate(self):
        approval = SMModelApproval(
            self.PKG_ARN, "ci",
            ApprovalStatus.PENDING,
            quality_metrics={"auc": 0.85},
            thresholds={"auc": 0.78},
        )
        assert approval.passes_quality_gate() is True

    def test_fails_quality_gate(self):
        approval = SMModelApproval(
            self.PKG_ARN, "ci",
            ApprovalStatus.PENDING,
            quality_metrics={"auc": 0.72},
            thresholds={"auc": 0.78},
        )
        assert approval.passes_quality_gate() is False

    def test_auto_approve_when_passes(self):
        approval = SMModelApproval(
            self.PKG_ARN, "ci",
            ApprovalStatus.PENDING,
            quality_metrics={"auc": 0.85},
            thresholds={"auc": 0.78},
        )
        approval.auto_approve()
        assert approval.approval_status == ApprovalStatus.APPROVED

    def test_auto_reject_when_fails(self):
        approval = SMModelApproval(
            self.PKG_ARN, "ci",
            ApprovalStatus.PENDING,
            quality_metrics={"auc": 0.65},
            thresholds={"auc": 0.78},
        )
        approval.auto_approve()
        assert approval.approval_status == ApprovalStatus.REJECTED

    def test_credit_risk_gate_factory(self):
        approval = SMModelApproval.credit_risk_gate(self.PKG_ARN, auc=0.82, psi=0.15)
        d = approval.to_dict()
        assert d["QualityMetrics"]["auc"] == 0.82
        assert d["Thresholds"]["auc"] == 0.78

    def test_to_dict_passes_gate_field(self):
        approval = SMModelApproval.credit_risk_gate(self.PKG_ARN, auc=0.80, psi=0.10)
        d = approval.to_dict()
        assert d["PassesGate"] is True
