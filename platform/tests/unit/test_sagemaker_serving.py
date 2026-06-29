"""Unit tests for infra.aws.sagemaker_serving (Day 81)."""

import pytest

from infra.aws.sagemaker_serving import (
    SMModelPackage,
    SMEndpointConfig,
    SMEndpoint,
    EndpointType,
)


# ── SMModelPackage ────────────────────────────────────────────────────────────

class TestSMModelPackage:
    def _make(self, **kwargs):
        defaults = dict(
            model_package_group_name="credit-risk-models",
            model_description="Credit risk XGBoost v3",
            inference_image_uri="123.dkr.ecr.us-east-1.amazonaws.com/cr:v1",
            model_s3_uri="s3://bucket/models/v3/model.tar.gz",
        )
        defaults.update(kwargs)
        return SMModelPackage(**defaults)

    def test_empty_group_name_raises(self):
        with pytest.raises(ValueError, match="model_package_group_name"):
            self._make(model_package_group_name="")

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="inference_image_uri"):
            self._make(inference_image_uri="")

    def test_empty_s3_raises(self):
        with pytest.raises(ValueError, match="model_s3_uri"):
            self._make(model_s3_uri="")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="model_description"):
            self._make(model_description="")

    def test_invalid_approval_status_raises(self):
        with pytest.raises(ValueError, match="approval_status"):
            self._make(approval_status="MaybeApproved")

    def test_default_approval_pending(self):
        pkg = self._make()
        assert pkg.approval_status == "PendingManualApproval"

    def test_approve_method(self):
        pkg = self._make()
        result = pkg.approve()
        assert result is pkg
        assert pkg.approval_status == "Approved"

    def test_reject_method(self):
        pkg = self._make()
        pkg.reject()
        assert pkg.approval_status == "Rejected"

    def test_to_dict_structure(self):
        pkg = self._make(metrics={"auc": 0.83})
        d = pkg.to_dict()
        assert d["ModelPackageGroupName"] == "credit-risk-models"
        assert d["ModelApprovalStatus"] == "PendingManualApproval"
        assert "InferenceSpecification" in d
        assert d["CustomerMetadataProperties"]["auc"] == "0.83"

    def test_content_types_in_dict(self):
        pkg = self._make()
        d = pkg.to_dict()
        assert "text/csv" in d["InferenceSpecification"]["SupportedContentTypes"]


# ── SMEndpointConfig ──────────────────────────────────────────────────────────

class TestSMEndpointConfig:
    PKG_ARN = "arn:aws:sagemaker:us-east-1:123:model-package/credit-risk-models/1"

    def test_empty_config_name_raises(self):
        with pytest.raises(ValueError, match="config_name"):
            SMEndpointConfig("", EndpointType.REAL_TIME, self.PKG_ARN)

    def test_empty_model_package_arn_raises(self):
        with pytest.raises(ValueError, match="model_package_arn"):
            SMEndpointConfig("cfg", EndpointType.REAL_TIME, "")

    def test_real_time_factory(self):
        cfg = SMEndpointConfig.real_time("rt-cfg", self.PKG_ARN, "ml.m5.large")
        d = cfg.to_dict()
        assert d["EndpointType"] == "RealTime"
        assert d["ProductionVariants"][0]["InstanceType"] == "ml.m5.large"

    def test_real_time_data_capture(self):
        cfg = SMEndpointConfig.real_time("rt-cfg", self.PKG_ARN, data_capture_s3="s3://b/cap/")
        d = cfg.to_dict()
        assert "DataCaptureConfig" in d
        assert d["DataCaptureConfig"]["EnableCapture"] is True

    def test_serverless_factory(self):
        cfg = SMEndpointConfig.serverless("sl-cfg", self.PKG_ARN, memory_mb=4096, max_concurrency=20)
        d = cfg.to_dict()
        assert d["EndpointType"] == "Serverless"
        assert d["ProductionVariants"][0]["ServerlessConfig"]["MemorySizeInMB"] == 4096

    def test_async_factory(self):
        cfg = SMEndpointConfig.async_inference("async-cfg", self.PKG_ARN, "s3://b/out/")
        d = cfg.to_dict()
        assert d["EndpointType"] == "Async"
        assert "AsyncInferenceConfig" in d
        assert d["AsyncInferenceConfig"]["OutputConfig"]["S3OutputPath"] == "s3://b/out/"

    def test_batch_factory(self):
        cfg = SMEndpointConfig.batch_transform("bt-cfg", self.PKG_ARN, instance_count=3)
        d = cfg.to_dict()
        assert d["EndpointType"] == "BatchTransform"
        assert d["BatchTransformConfig"]["InstanceCount"] == 3

    def test_tags_in_dict(self):
        cfg = SMEndpointConfig("cfg", EndpointType.REAL_TIME, self.PKG_ARN, tags={"Env": "prod"})
        d = cfg.to_dict()
        assert any(t["Key"] == "Env" for t in d["Tags"])


# ── SMEndpoint ────────────────────────────────────────────────────────────────

class TestSMEndpoint:
    def test_empty_endpoint_name_raises(self):
        with pytest.raises(ValueError, match="endpoint_name"):
            SMEndpoint("", "config-v1")

    def test_empty_config_name_raises(self):
        with pytest.raises(ValueError, match="endpoint_config_name"):
            SMEndpoint("ep", "")

    def test_to_dict_structure(self):
        ep = SMEndpoint("credit-risk-prod", "credit-risk-config-v3")
        d = ep.to_dict()
        assert d["EndpointName"] == "credit-risk-prod"
        assert d["EndpointConfigName"] == "credit-risk-config-v3"

    def test_update_config_returns_dict(self):
        ep = SMEndpoint("credit-risk-prod", "config-v1")
        update = ep.update_config("config-v2")
        assert update["EndpointConfigName"] == "config-v2"
        assert update["EndpointName"] == "credit-risk-prod"

    def test_tags_propagated(self):
        ep = SMEndpoint("ep", "cfg", tags={"Env": "prod"})
        d = ep.to_dict()
        assert d["Tags"][0]["Key"] == "Env"
