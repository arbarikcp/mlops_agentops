"""Unit tests for infra.aws.security (Day 85)."""

import pytest

from infra.aws.security import (
    SpotConfig,
    SpotInterruptionHandling,
    KMSKeyPolicy,
    KMSConfig,
    BudgetAlert,
    BudgetAlertType,
    BudgetGuardrail,
    PrivateLinkConfig,
)


# ── SpotConfig ────────────────────────────────────────────────────────────────

class TestSpotConfig:
    def test_empty_instance_type_raises(self):
        with pytest.raises(ValueError, match="instance_type"):
            SpotConfig("", 0.10)

    def test_zero_price_raises(self):
        with pytest.raises(ValueError, match="max_price_per_hour"):
            SpotConfig("ml.m5.xlarge", 0.0)

    def test_wait_less_than_run_raises(self):
        with pytest.raises(ValueError, match="max_wait_seconds"):
            SpotConfig("ml.m5.xlarge", 0.10, max_run_seconds=3600, max_wait_seconds=1800)

    def test_estimated_savings_70_pct(self):
        cfg = SpotConfig("ml.m5.xlarge", 0.10)
        assert cfg.estimated_savings_pct == 70.0

    def test_to_dict_enable_managed_spot(self):
        cfg = SpotConfig("ml.m5.xlarge", 0.15)
        d = cfg.to_dict()
        assert d["enableManagedSpot"] is True

    def test_checkpoint_in_dict_when_set(self):
        cfg = SpotConfig("ml.m5.xlarge", 0.10, checkpoint_s3_uri="s3://b/ckpt/")
        d = cfg.to_dict()
        assert "checkpointConfig" in d
        assert d["checkpointConfig"]["S3Uri"] == "s3://b/ckpt/"

    def test_no_checkpoint_when_empty(self):
        cfg = SpotConfig("ml.m5.xlarge", 0.10)
        d = cfg.to_dict()
        assert "checkpointConfig" not in d

    def test_ml_training_spot_factory(self):
        cfg = SpotConfig.ml_training_spot("ml.g4dn.xlarge", "s3://b/ckpt/")
        assert cfg.interruption_handling == SpotInterruptionHandling.CHECKPOINT
        assert cfg.instance_type == "ml.g4dn.xlarge"

    def test_factory_defaults(self):
        cfg = SpotConfig.ml_training_spot()
        d = cfg.to_dict()
        assert d["instanceType"] == "ml.m5.xlarge"


# ── KMSConfig ─────────────────────────────────────────────────────────────────

class TestKMSConfig:
    def test_empty_alias_raises(self):
        with pytest.raises(ValueError, match="key_alias"):
            KMSConfig("", "desc", "123456789012")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            KMSConfig("alias/key", "", "123456789012")

    def test_empty_account_raises(self):
        with pytest.raises(ValueError, match="account_id"):
            KMSConfig("alias/key", "desc", "")

    def test_deletion_window_too_small_raises(self):
        with pytest.raises(ValueError, match="deletion_window_days"):
            KMSConfig("alias/key", "desc", "123456789012", deletion_window_days=5)

    def test_key_arn_placeholder(self):
        cfg = KMSConfig("mlops/key", "desc", "123456789012")
        assert "123456789012" in cfg.key_arn_placeholder
        assert "mlops/key" in cfg.key_arn_placeholder

    def test_to_dict_structure(self):
        cfg = KMSConfig("mlops/key", "desc", "123456789012")
        d = cfg.to_dict()
        assert d["AliasName"] == "alias/mlops/key"
        assert d["EnableKeyRotation"] is True
        assert "KeyPolicy" in d

    def test_default_root_policy_when_no_policies(self):
        cfg = KMSConfig("mlops/key", "desc", "123456789012")
        d = cfg.to_dict()
        stmts = d["KeyPolicy"]["Statement"]
        assert any("EnableRootAccess" == s["Sid"] for s in stmts)

    def test_ml_artifacts_key_factory(self):
        cfg = KMSConfig.ml_artifacts_key("123456789012", "arn:aws:iam::123456789012:role/SM")
        d = cfg.to_dict()
        stmts = d["KeyPolicy"]["Statement"]
        assert any("AllowSageMakerRole" == s["Sid"] for s in stmts)

    def test_kms_key_policy_empty_sid_raises(self):
        with pytest.raises(ValueError, match="sid"):
            KMSKeyPolicy("", ["arn:aws:iam::123:root"], ["kms:Decrypt"])

    def test_kms_key_policy_empty_principals_raises(self):
        with pytest.raises(ValueError, match="principals"):
            KMSKeyPolicy("mySid", [], ["kms:Decrypt"])


# ── BudgetGuardrail ───────────────────────────────────────────────────────────

class TestBudgetGuardrail:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="budget_name"):
            BudgetGuardrail("", "123456789012", 500.0)

    def test_empty_account_raises(self):
        with pytest.raises(ValueError, match="account_id"):
            BudgetGuardrail("budget", "", 500.0)

    def test_zero_limit_raises(self):
        with pytest.raises(ValueError, match="monthly_limit_usd"):
            BudgetGuardrail("budget", "123456789012", 0.0)

    def test_to_dict_structure(self):
        g = BudgetGuardrail("budget", "123456789012", 1000.0)
        d = g.to_dict()
        assert d["Budget"]["BudgetLimit"]["Amount"] == "1000.0"
        assert d["Budget"]["TimeUnit"] == "MONTHLY"

    def test_budget_alert_threshold_raises_for_zero(self):
        with pytest.raises(ValueError, match="threshold_pct"):
            BudgetAlert(0.0, BudgetAlertType.ACTUAL, ["a@b.com"])

    def test_budget_alert_empty_emails_raises(self):
        with pytest.raises(ValueError, match="notification_emails"):
            BudgetAlert(80.0, BudgetAlertType.ACTUAL, [])

    def test_budget_alert_to_dict(self):
        alert = BudgetAlert(80.0, BudgetAlertType.ACTUAL, ["ops@company.com"])
        d = alert.to_dict()
        assert d["Threshold"] == 80.0
        assert d["NotificationType"] == "ACTUAL"
        assert d["Subscribers"][0]["Address"] == "ops@company.com"

    def test_ml_platform_budget_factory(self):
        g = BudgetGuardrail.ml_platform_budget("123456789012", 2000.0, ["eng@co.com"])
        assert len(g.alerts) == 3
        d = g.to_dict()
        assert len(d["NotificationsWithSubscribers"]) == 3

    def test_service_filter_in_dict(self):
        g = BudgetGuardrail("b", "123456789012", 100.0, service_filter="AmazonSageMaker")
        d = g.to_dict()
        assert "AmazonSageMaker" in d["Budget"]["CostFilters"]["Service"]


# ── PrivateLinkConfig ─────────────────────────────────────────────────────────

class TestPrivateLinkConfig:
    def _make(self, **kwargs):
        defaults = dict(
            endpoint_name="vpce-sagemaker",
            service_name="com.amazonaws.us-east-1.sagemaker.api",
            vpc_id="vpc-12345",
            subnet_ids=["subnet-a", "subnet-b"],
            security_group_ids=["sg-xyz"],
        )
        defaults.update(kwargs)
        return PrivateLinkConfig(**defaults)

    def test_empty_endpoint_name_raises(self):
        with pytest.raises(ValueError, match="endpoint_name"):
            self._make(endpoint_name="")

    def test_empty_service_name_raises(self):
        with pytest.raises(ValueError, match="service_name"):
            self._make(service_name="")

    def test_empty_vpc_id_raises(self):
        with pytest.raises(ValueError, match="vpc_id"):
            self._make(vpc_id="")

    def test_empty_subnet_ids_raises(self):
        with pytest.raises(ValueError, match="subnet_ids"):
            self._make(subnet_ids=[])

    def test_invalid_endpoint_type_raises(self):
        with pytest.raises(ValueError, match="endpoint_type"):
            self._make(endpoint_type="NAT")

    def test_interface_endpoint_has_private_dns(self):
        cfg = self._make(endpoint_type="Interface")
        d = cfg.to_dict()
        assert d["PrivateDnsEnabled"] is True
        assert "SecurityGroupIds" in d

    def test_gateway_endpoint_no_security_groups(self):
        cfg = self._make(endpoint_type="Gateway")
        d = cfg.to_dict()
        assert "SecurityGroupIds" not in d

    def test_sagemaker_endpoints_factory(self):
        endpoints = PrivateLinkConfig.sagemaker_endpoints(
            "vpc-123", ["subnet-a"], ["sg-x"]
        )
        assert len(endpoints) == 5
        service_names = [e.service_name for e in endpoints]
        assert any("sagemaker.api" in s for s in service_names)
        assert any("ecr.api" in s for s in service_names)
