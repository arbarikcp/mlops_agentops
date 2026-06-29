"""AWS cost and security builders — Spot, KMS, PrivateLink, Budget guardrails.

Day 85: Four pillars of AWS ML security and cost management:
1. Spot instances — 70% cost reduction with graceful interruption handling
2. KMS encryption — envelope encryption for model artifacts at rest
3. PrivateLink — private connectivity; data never traverses public internet
4. Budget guardrails — hard spending caps to prevent runaway training costs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── Spot Config ───────────────────────────────────────────────────────────────


class SpotInterruptionHandling(str, Enum):
    CHECKPOINT = "checkpoint"   # save state and resume on new spot instance
    TERMINATE = "terminate"     # accept loss, just re-run from scratch
    HIBERNATE = "hibernate"     # suspend to EBS and resume


@dataclass
class SpotConfig:
    """Spot instance configuration for ML training jobs.

    Spot instances are EC2 capacity offered at 60-90% discount but can be
    reclaimed by AWS with 2-minute warning. SageMaker handles checkpointing
    and re-queuing automatically via EnableManagedSpotTraining.
    """

    instance_type: str
    max_price_per_hour: float  # USD — bid ceiling
    interruption_handling: SpotInterruptionHandling = SpotInterruptionHandling.CHECKPOINT
    checkpoint_s3_uri: str = ""
    max_wait_seconds: int = 7200  # how long to wait for spot capacity
    max_run_seconds: int = 3600

    def __post_init__(self) -> None:
        if not self.instance_type:
            raise ValueError("instance_type must not be empty")
        if self.max_price_per_hour <= 0:
            raise ValueError("max_price_per_hour must be > 0")
        if self.max_wait_seconds < self.max_run_seconds:
            raise ValueError("max_wait_seconds must be >= max_run_seconds")

    @property
    def estimated_savings_pct(self) -> float:
        """Approximate spot discount vs on-demand (heuristic: 70%)."""
        return 70.0

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "instanceType": self.instance_type,
            "spotMaxPricePerHour": self.max_price_per_hour,
            "interruptionHandling": self.interruption_handling.value,
            "maxWaitTimeInSeconds": self.max_wait_seconds,
            "maxRuntimeInSeconds": self.max_run_seconds,
            "enableManagedSpot": True,
            "estimatedSavingsPct": self.estimated_savings_pct,
        }
        if self.checkpoint_s3_uri:
            cfg["checkpointConfig"] = {
                "S3Uri": self.checkpoint_s3_uri,
                "LocalPath": "/opt/ml/checkpoints",
            }
        return cfg

    @classmethod
    def ml_training_spot(
        cls,
        instance_type: str = "ml.m5.xlarge",
        checkpoint_s3: str = "",
    ) -> "SpotConfig":
        """Factory: standard ML training spot config."""
        return cls(
            instance_type=instance_type,
            max_price_per_hour=0.20,
            interruption_handling=SpotInterruptionHandling.CHECKPOINT,
            checkpoint_s3_uri=checkpoint_s3,
            max_wait_seconds=7200,
            max_run_seconds=3600,
        )


# ── KMS ───────────────────────────────────────────────────────────────────────


@dataclass
class KMSKeyPolicy:
    """KMS key policy statement."""

    sid: str
    principals: list[str]
    actions: list[str]
    allow: bool = True
    conditions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sid:
            raise ValueError("sid must not be empty")
        if not self.principals:
            raise ValueError("principals must not be empty")

    def to_dict(self) -> dict[str, Any]:
        stmt: dict[str, Any] = {
            "Sid": self.sid,
            "Effect": "Allow" if self.allow else "Deny",
            "Principal": {"AWS": self.principals},
            "Action": self.actions,
            "Resource": "*",
        }
        if self.conditions:
            stmt["Condition"] = self.conditions
        return stmt


@dataclass
class KMSConfig:
    """KMS Customer Managed Key (CMK) for ML artifact encryption.

    Envelope encryption: KMS generates a data key that encrypts the artifact;
    only the encrypted data key is stored alongside the artifact. To decrypt,
    you must have kms:Decrypt permission — creating an audit trail of every
    model access in CloudTrail.
    """

    key_alias: str
    description: str
    account_id: str
    region: str = "us-east-1"
    enable_key_rotation: bool = True
    deletion_window_days: int = 30
    key_policies: list[KMSKeyPolicy] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.key_alias:
            raise ValueError("key_alias must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")
        if not self.account_id:
            raise ValueError("account_id must not be empty")
        if self.deletion_window_days < 7:
            raise ValueError("deletion_window_days must be >= 7 (AWS minimum)")

    @property
    def key_arn_placeholder(self) -> str:
        """ARN placeholder (real ARN assigned by AWS at creation time)."""
        return f"arn:aws:kms:{self.region}:{self.account_id}:alias/{self.key_alias}"

    def to_dict(self) -> dict[str, Any]:
        policy_doc = {
            "Version": "2012-10-17",
            "Statement": [p.to_dict() for p in self.key_policies],
        }
        if not self.key_policies:
            # Default: account root can manage
            policy_doc["Statement"] = [{
                "Sid": "EnableRootAccess",
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::{self.account_id}:root"},
                "Action": "kms:*",
                "Resource": "*",
            }]
        return {
            "AliasName": f"alias/{self.key_alias}",
            "Description": self.description,
            "EnableKeyRotation": self.enable_key_rotation,
            "PendingWindowInDays": self.deletion_window_days,
            "KeyPolicy": policy_doc,
            "Tags": [{"TagKey": k, "TagValue": v} for k, v in self.tags.items()],
        }

    @classmethod
    def ml_artifacts_key(cls, account_id: str, sagemaker_role_arn: str) -> "KMSConfig":
        """Factory: KMS key for SageMaker model artifacts."""
        cfg = cls(
            key_alias="mlops/model-artifacts",
            description="CMK for encrypting SageMaker model artifacts",
            account_id=account_id,
            tags={"Project": "mlops", "Purpose": "model-encryption"},
        )
        cfg.key_policies = [
            KMSKeyPolicy(
                sid="AllowRootAdmin",
                principals=[f"arn:aws:iam::{account_id}:root"],
                actions=["kms:*"],
            ),
            KMSKeyPolicy(
                sid="AllowSageMakerRole",
                principals=[sagemaker_role_arn],
                actions=["kms:GenerateDataKey", "kms:Decrypt", "kms:DescribeKey"],
            ),
        ]
        return cfg


# ── Budget Guardrail ───────────────────────────────────────────────────────────


class BudgetAlertType(str, Enum):
    ACTUAL = "ACTUAL"       # alert when actual spend exceeds threshold
    FORECASTED = "FORECASTED"  # alert when forecasted spend will exceed threshold


@dataclass
class BudgetAlert:
    """A single budget alert threshold."""

    threshold_pct: float  # e.g. 80.0 for 80%
    alert_type: BudgetAlertType
    notification_emails: list[str]

    def __post_init__(self) -> None:
        if not 0 < self.threshold_pct <= 200:
            raise ValueError("threshold_pct must be in (0, 200]")
        if not self.notification_emails:
            raise ValueError("notification_emails must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "NotificationType": self.alert_type.value,
            "ComparisonOperator": "GREATER_THAN",
            "Threshold": self.threshold_pct,
            "ThresholdType": "PERCENTAGE",
            "NotificationState": "ALARM",
            "Subscribers": [
                {"SubscriptionType": "EMAIL", "Address": email}
                for email in self.notification_emails
            ],
        }


@dataclass
class BudgetGuardrail:
    """AWS Budget guardrail — prevents runaway ML training costs.

    Sets a monthly spend limit and sends alerts at configurable thresholds.
    At 100% actual spend, optionally applies an SCP to deny further SageMaker
    training job creation.
    """

    budget_name: str
    account_id: str
    monthly_limit_usd: float
    alerts: list[BudgetAlert] = field(default_factory=list)
    service_filter: str = "AmazonSageMaker"  # AWS Cost Explorer service name
    auto_stop_at_limit: bool = False

    def __post_init__(self) -> None:
        if not self.budget_name:
            raise ValueError("budget_name must not be empty")
        if not self.account_id:
            raise ValueError("account_id must not be empty")
        if self.monthly_limit_usd <= 0:
            raise ValueError("monthly_limit_usd must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "AccountId": self.account_id,
            "Budget": {
                "BudgetName": self.budget_name,
                "BudgetLimit": {
                    "Amount": str(self.monthly_limit_usd),
                    "Unit": "USD",
                },
                "TimeUnit": "MONTHLY",
                "BudgetType": "COST",
                "CostFilters": {"Service": [self.service_filter]},
            },
            "NotificationsWithSubscribers": [a.to_dict() for a in self.alerts],
            "AutoStopAtLimit": self.auto_stop_at_limit,
        }

    @classmethod
    def ml_platform_budget(
        cls,
        account_id: str,
        monthly_limit: float,
        alert_emails: list[str],
    ) -> "BudgetGuardrail":
        """Factory: standard ML platform budget with 80%/100% alerts."""
        guardrail = cls(
            budget_name="mlops-sagemaker-budget",
            account_id=account_id,
            monthly_limit_usd=monthly_limit,
        )
        guardrail.alerts = [
            BudgetAlert(80.0, BudgetAlertType.ACTUAL, alert_emails),
            BudgetAlert(100.0, BudgetAlertType.ACTUAL, alert_emails),
            BudgetAlert(110.0, BudgetAlertType.FORECASTED, alert_emails),
        ]
        return guardrail


# ── PrivateLink ───────────────────────────────────────────────────────────────


@dataclass
class PrivateLinkConfig:
    """VPC PrivateLink endpoint for private AWS service access.

    PrivateLink routes traffic to AWS services (S3, ECR, SageMaker) over the
    AWS backbone — no data traverses the public internet. This satisfies:
    - Data exfiltration controls (traffic stays in AWS network)
    - Compliance requirements (HIPAA, PCI-DSS, SOC2)
    - Cost reduction (avoids NAT Gateway data processing charges)
    """

    endpoint_name: str
    service_name: str  # e.g. "com.amazonaws.us-east-1.sagemaker.api"
    vpc_id: str
    subnet_ids: list[str]
    security_group_ids: list[str]
    endpoint_type: str = "Interface"  # "Interface" | "Gateway"
    private_dns_enabled: bool = True
    region: str = "us-east-1"
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.endpoint_name:
            raise ValueError("endpoint_name must not be empty")
        if not self.service_name:
            raise ValueError("service_name must not be empty")
        if not self.vpc_id:
            raise ValueError("vpc_id must not be empty")
        if not self.subnet_ids:
            raise ValueError("subnet_ids must not be empty")
        if self.endpoint_type not in ("Interface", "Gateway"):
            raise ValueError(f"endpoint_type invalid: {self.endpoint_type!r}")

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "VpcEndpointName": self.endpoint_name,
            "ServiceName": self.service_name,
            "VpcId": self.vpc_id,
            "VpcEndpointType": self.endpoint_type,
            "SubnetIds": self.subnet_ids,
            "TagSpecifications": [{"Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()]}],
        }
        if self.endpoint_type == "Interface":
            cfg["PrivateDnsEnabled"] = self.private_dns_enabled
            cfg["SecurityGroupIds"] = self.security_group_ids
        return cfg

    @classmethod
    def sagemaker_endpoints(cls, vpc_id: str, subnet_ids: list[str], sg_ids: list[str], region: str = "us-east-1") -> list["PrivateLinkConfig"]:
        """Factory: all PrivateLink endpoints needed for SageMaker in a private VPC."""
        services = ["sagemaker.api", "sagemaker.runtime", "ecr.api", "ecr.dkr", "sts"]
        return [
            cls(
                endpoint_name=f"vpce-{svc.replace('.', '-')}",
                service_name=f"com.amazonaws.{region}.{svc}",
                vpc_id=vpc_id,
                subnet_ids=subnet_ids,
                security_group_ids=sg_ids,
                region=region,
                tags={"Purpose": "sagemaker-private-access"},
            )
            for svc in services
        ]
