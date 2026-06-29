"""Unit tests for infra.terraform_config (Day 86)."""

import pytest

from infra.terraform_config import (
    TFVariable,
    TFResource,
    TFOutput,
    TFModule,
    TFConfig,
)


# ── TFVariable ────────────────────────────────────────────────────────────────

class TestTFVariable:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            TFVariable("", "string", "desc")

    def test_empty_type_raises(self):
        with pytest.raises(ValueError, match="var_type"):
            TFVariable("my_var", "", "desc")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            TFVariable("my_var", "string", "")

    def test_to_dict_with_default(self):
        var = TFVariable("region", "string", "AWS region", default="us-east-1")
        d = var.to_dict()
        assert d["region"]["default"] == "us-east-1"
        assert d["region"]["type"] == "string"

    def test_to_dict_no_default_when_none(self):
        var = TFVariable("bucket", "string", "S3 bucket name")
        d = var.to_dict()
        assert "default" not in d["bucket"]

    def test_sensitive_variable(self):
        var = TFVariable("db_password", "string", "DB password", sensitive=True)
        d = var.to_dict()
        assert d["db_password"]["sensitive"] is True

    def test_validation_in_dict(self):
        var = TFVariable("region", "string", "Region",
                         validation_condition='can(regex("^us-", var.region))',
                         validation_error="Must be a US region")
        d = var.to_dict()
        assert "validation" in d["region"]


# ── TFResource ────────────────────────────────────────────────────────────────

class TestTFResource:
    def test_empty_resource_type_raises(self):
        with pytest.raises(ValueError, match="resource_type"):
            TFResource("", "my_bucket", {"bucket": "test"})

    def test_empty_resource_name_raises(self):
        with pytest.raises(ValueError, match="resource_name"):
            TFResource("aws_s3_bucket", "", {"bucket": "test"})

    def test_empty_arguments_raises(self):
        with pytest.raises(ValueError, match="arguments"):
            TFResource("aws_s3_bucket", "my_bucket", {})

    def test_ref_property(self):
        r = TFResource("aws_s3_bucket", "artifacts", {"bucket": "test"})
        assert r.ref == "aws_s3_bucket.artifacts"

    def test_id_ref_property(self):
        r = TFResource("aws_s3_bucket", "artifacts", {"bucket": "test"})
        assert r.id_ref == "aws_s3_bucket.artifacts.id"

    def test_arn_ref_property(self):
        r = TFResource("aws_iam_role", "sm_role", {"name": "sm"})
        assert r.arn_ref == "aws_iam_role.sm_role.arn"

    def test_to_dict_structure(self):
        r = TFResource("aws_s3_bucket", "artifacts", {"bucket": "test"})
        d = r.to_dict()
        assert "resource" in d
        assert "aws_s3_bucket" in d["resource"]
        assert "artifacts" in d["resource"]["aws_s3_bucket"]

    def test_depends_on_in_dict(self):
        r = TFResource("aws_s3_bucket", "b", {"bucket": "x"}, depends_on=["aws_iam_role.sm"])
        d = r.to_dict()
        assert "depends_on" in d["resource"]["aws_s3_bucket"]["b"]

    def test_s3_bucket_factory(self):
        r = TFResource.s3_bucket("artifacts")
        d = r.to_dict()
        assert "lifecycle" in d["resource"]["aws_s3_bucket"]["artifacts"]
        assert d["resource"]["aws_s3_bucket"]["artifacts"]["lifecycle"]["prevent_destroy"] is True

    def test_iam_role_factory(self):
        r = TFResource.iam_role("sagemaker_role", "sagemaker")
        d = r.to_dict()
        role_cfg = d["resource"]["aws_iam_role"]["sagemaker_role"]
        assert "sagemaker.amazonaws.com" in str(role_cfg["assume_role_policy"])

    def test_ecr_repository_factory(self):
        r = TFResource.ecr_repository("credit-risk")
        d = r.to_dict()
        assert d["resource"]["aws_ecr_repository"]["credit-risk"]["image_tag_mutability"] == "IMMUTABLE"

    def test_lifecycle_in_dict(self):
        r = TFResource("aws_s3_bucket", "b", {"bucket": "x"}, lifecycle={"prevent_destroy": True})
        d = r.to_dict()
        assert d["resource"]["aws_s3_bucket"]["b"]["lifecycle"]["prevent_destroy"] is True


# ── TFOutput ──────────────────────────────────────────────────────────────────

class TestTFOutput:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            TFOutput("", "aws_s3_bucket.b.arn", "desc")

    def test_empty_value_raises(self):
        with pytest.raises(ValueError, match="value"):
            TFOutput("bucket_arn", "", "desc")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            TFOutput("bucket_arn", "aws_s3_bucket.b.arn", "")

    def test_to_dict(self):
        o = TFOutput("bucket_arn", "aws_s3_bucket.b.arn", "ARN of bucket")
        d = o.to_dict()
        assert d["bucket_arn"]["value"] == "aws_s3_bucket.b.arn"
        assert d["bucket_arn"]["sensitive"] is False

    def test_sensitive_output(self):
        o = TFOutput("db_pass", "aws_db_instance.db.password", "DB pass", sensitive=True)
        assert o.to_dict()["db_pass"]["sensitive"] is True


# ── TFModule ──────────────────────────────────────────────────────────────────

class TestTFModule:
    def test_empty_module_name_raises(self):
        with pytest.raises(ValueError, match="module_name"):
            TFModule("", "terraform-aws-modules/vpc/aws")

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="source"):
            TFModule("vpc", "")

    def test_to_dict_structure(self):
        m = TFModule("vpc", "terraform-aws-modules/vpc/aws", version="5.0.0",
                     inputs={"cidr": "10.0.0.0/16"})
        d = m.to_dict()
        assert "module" in d
        assert d["module"]["vpc"]["source"] == "terraform-aws-modules/vpc/aws"
        assert d["module"]["vpc"]["version"] == "5.0.0"
        assert d["module"]["vpc"]["cidr"] == "10.0.0.0/16"

    def test_vpc_factory(self):
        m = TFModule.vpc("main", "10.0.0.0/16")
        d = m.to_dict()
        assert d["module"]["main"]["enable_nat_gateway"] is True

    def test_eks_factory(self):
        m = TFModule.eks("cluster", "main")
        d = m.to_dict()
        assert "cluster_version" in d["module"]["cluster"]


# ── TFConfig ──────────────────────────────────────────────────────────────────

class TestTFConfig:
    def _make(self, **kwargs):
        defaults = dict(
            config_name="ml-platform",
            provider="aws",
            provider_config={"source": "hashicorp/aws", "version": "~> 5.0"},
        )
        defaults.update(kwargs)
        return TFConfig(**defaults)

    def test_empty_config_name_raises(self):
        with pytest.raises(ValueError, match="config_name"):
            self._make(config_name="")

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            self._make(provider="")

    def test_empty_provider_config_raises(self):
        with pytest.raises(ValueError, match="provider_config"):
            self._make(provider_config={})

    def test_to_dict_has_terraform_block(self):
        cfg = self._make()
        d = cfg.to_dict()
        assert "terraform" in d
        assert "required_providers" in d["terraform"]

    def test_to_dict_has_backend(self):
        cfg = self._make()
        d = cfg.to_dict()
        assert "backend" in d["terraform"]

    def test_add_variable_chaining(self):
        cfg = self._make()
        var = TFVariable("prefix", "string", "Prefix")
        result = cfg.add_variable(var)
        assert result is cfg
        d = cfg.to_dict()
        assert "prefix" in d["variable"]

    def test_add_resource(self):
        cfg = self._make()
        r = TFResource.s3_bucket("artifacts")
        cfg.add_resource(r)
        d = cfg.to_dict()
        assert "aws_s3_bucket" in d["resource"]

    def test_add_module(self):
        cfg = self._make()
        m = TFModule.vpc("main", "10.0.0.0/16")
        cfg.add_module(m)
        d = cfg.to_dict()
        assert "main" in d["module"]

    def test_add_output(self):
        cfg = self._make()
        o = TFOutput("bucket_arn", "aws_s3_bucket.a.arn", "Bucket ARN")
        cfg.add_output(o)
        d = cfg.to_dict()
        assert "bucket_arn" in d["output"]

    def test_ml_platform_aws_factory(self):
        cfg = TFConfig.ml_platform_aws()
        d = cfg.to_dict()
        assert "variable" in d
        assert "resource" in d
        assert "aws_s3_bucket" in d["resource"]
        assert "output" in d
