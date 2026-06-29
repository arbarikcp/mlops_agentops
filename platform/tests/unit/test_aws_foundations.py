"""Unit tests for infra.aws.foundations (Day 79)."""

import pytest

from infra.aws.foundations import (
    IAMStatement,
    IAMPolicyDoc,
    ECRLifecycleRule,
    ECRRepository,
    SubnetConfig,
    VPCEndpoint,
    VPCConfig,
)


# ── IAMStatement ──────────────────────────────────────────────────────────────

class TestIAMStatement:
    def test_valid_allow(self):
        s = IAMStatement("Allow", ["s3:GetObject"], ["arn:aws:s3:::bucket/*"])
        d = s.to_dict()
        assert d["Effect"] == "Allow"
        assert "s3:GetObject" in d["Action"]

    def test_valid_deny(self):
        s = IAMStatement("Deny", ["sagemaker:Delete*"], ["*"])
        assert s.to_dict()["Effect"] == "Deny"

    def test_invalid_effect(self):
        with pytest.raises(ValueError, match="effect"):
            IAMStatement("AllowAll", ["s3:*"], ["*"])

    def test_empty_actions_raises(self):
        with pytest.raises(ValueError, match="actions"):
            IAMStatement("Allow", [], ["*"])

    def test_empty_resources_raises(self):
        with pytest.raises(ValueError, match="resources"):
            IAMStatement("Allow", ["s3:Get*"], [])

    def test_with_principals(self):
        s = IAMStatement("Allow", ["s3:Put*"], ["*"], principals=["arn:aws:iam::123:role/SM"])
        d = s.to_dict()
        assert "Principal" in d
        assert "arn:aws:iam::123:role/SM" in d["Principal"]["AWS"]

    def test_with_conditions(self):
        s = IAMStatement("Allow", ["sagemaker:*"], ["*"],
                         conditions={"StringEquals": {"sagemaker:ResourceTag/Env": "dev"}})
        d = s.to_dict()
        assert "Condition" in d


# ── IAMPolicyDoc ──────────────────────────────────────────────────────────────

class TestIAMPolicyDoc:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            IAMPolicyDoc("", "desc")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            IAMPolicyDoc("MyPolicy", "")

    def test_to_dict_version(self):
        doc = IAMPolicyDoc("P", "desc")
        assert doc.to_dict()["Version"] == "2012-10-17"

    def test_add_statement_chaining(self):
        doc = IAMPolicyDoc("P", "desc")
        s = IAMStatement("Allow", ["s3:Get*"], ["*"])
        result = doc.add_statement(s)
        assert result is doc
        assert len(doc.statements) == 1

    def test_sagemaker_execution_role_factory(self):
        doc = IAMPolicyDoc.sagemaker_execution_role("123456789012")
        d = doc.to_dict()
        actions_flat = [a for stmt in d["Statement"] for a in stmt["Action"]]
        assert "s3:GetObject" in actions_flat
        assert "ecr:GetDownloadUrlForLayer" in actions_flat

    def test_ecr_push_policy_factory(self):
        doc = IAMPolicyDoc.ecr_push_policy("123456789012", "credit-risk")
        d = doc.to_dict()
        assert len(d["Statement"]) == 1
        assert "ecr:PutImage" in d["Statement"][0]["Action"]

    def test_ml_data_scientist_policy_factory(self):
        doc = IAMPolicyDoc.ml_data_scientist_policy("123456789012")
        d = doc.to_dict()
        # Should have allow + deny statements
        effects = [s["Effect"] for s in d["Statement"]]
        assert "Allow" in effects
        assert "Deny" in effects


# ── ECRLifecycleRule ──────────────────────────────────────────────────────────

class TestECRLifecycleRule:
    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            ECRLifecycleRule(1, "", "tagged", "imageCountMoreThan", 10)

    def test_invalid_tag_status(self):
        with pytest.raises(ValueError, match="tag_status"):
            ECRLifecycleRule(1, "desc", "invalid", "imageCountMoreThan", 10)

    def test_invalid_count_type(self):
        with pytest.raises(ValueError, match="count_type"):
            ECRLifecycleRule(1, "desc", "tagged", "badType", 10)

    def test_zero_count_raises(self):
        with pytest.raises(ValueError, match="count_number"):
            ECRLifecycleRule(1, "desc", "tagged", "imageCountMoreThan", 0)

    def test_keep_last_n_factory(self):
        rule = ECRLifecycleRule.keep_last_n_tagged(5)
        d = rule.to_dict()
        assert d["selection"]["countNumber"] == 5
        assert d["selection"]["countType"] == "imageCountMoreThan"
        assert d["action"]["type"] == "expire"

    def test_expire_untagged_factory(self):
        rule = ECRLifecycleRule.expire_untagged_after_days(14)
        d = rule.to_dict()
        assert d["selection"]["tagStatus"] == "untagged"
        assert "countUnit" in d["selection"]

    def test_rule_priority(self):
        rule = ECRLifecycleRule(3, "desc", "any", "imageCountMoreThan", 50)
        assert rule.to_dict()["rulePriority"] == 3


# ── ECRRepository ─────────────────────────────────────────────────────────────

class TestECRRepository:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            ECRRepository("", "123456789012")

    def test_empty_account_raises(self):
        with pytest.raises(ValueError, match="account_id"):
            ECRRepository("repo", "")

    def test_uri_format(self):
        repo = ECRRepository("credit-risk", "123456789012", region="us-west-2")
        assert "123456789012" in repo.uri
        assert "us-west-2" in repo.uri
        assert "credit-risk" in repo.uri

    def test_to_dict_with_lifecycle(self):
        repo = ECRRepository("credit-risk", "123456789012")
        repo.lifecycle_rules = [ECRLifecycleRule.keep_last_n_tagged(10)]
        d = repo.to_dict()
        assert d["lifecyclePolicy"]["rules"]
        assert d["imageScanningConfiguration"]["scanOnPush"] is True


# ── VPCConfig ─────────────────────────────────────────────────────────────────

class TestVPCConfig:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            VPCConfig("", "10.0.0.0/16")

    def test_empty_cidr_raises(self):
        with pytest.raises(ValueError, match="cidr_block"):
            VPCConfig("vpc", "")

    def test_subnet_type_validation(self):
        with pytest.raises(ValueError, match="subnet_type"):
            SubnetConfig("bad", "10.0.0.0/24", "us-east-1a", "dmz")

    def test_vpc_endpoint_type_validation(self):
        with pytest.raises(ValueError, match="endpoint_type"):
            VPCEndpoint("s3", "NATGateway")

    def test_ml_platform_vpc_factory(self):
        vpc = VPCConfig.ml_platform_vpc()
        assert len(vpc.private_subnets) == 2
        assert len(vpc.public_subnets) == 2
        assert len(vpc.endpoints) == 4

    def test_to_dict_structure(self):
        vpc = VPCConfig.ml_platform_vpc("test-vpc")
        d = vpc.to_dict()
        assert d["vpcName"] == "test-vpc"
        assert d["cidrBlock"] == "10.0.0.0/16"
        assert len(d["subnets"]) == 4
        assert len(d["endpoints"]) == 4

    def test_gateway_endpoint_no_private_dns(self):
        ep = VPCEndpoint("s3", "Gateway")
        d = ep.to_dict()
        assert d["privateDnsEnabled"] is False

    def test_interface_endpoint_private_dns(self):
        ep = VPCEndpoint("ecr.api", "Interface", ["private-a"])
        d = ep.to_dict()
        assert d["privateDnsEnabled"] is True
