"""AWS foundations — IAM, ECR, VPC builders for ML workloads.

Day 79: Establishes IAM least-privilege patterns, ECR image lifecycle
management, and VPC networking topology for ML workloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── IAM ──────────────────────────────────────────────────────────────────────


@dataclass
class IAMStatement:
    """Single IAM policy statement."""

    effect: str  # "Allow" | "Deny"
    actions: list[str]
    resources: list[str]
    principals: list[str] = field(default_factory=list)
    conditions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.effect not in ("Allow", "Deny"):
            raise ValueError(f"effect must be Allow or Deny, got: {self.effect!r}")
        if not self.actions:
            raise ValueError("actions must not be empty")
        if not self.resources:
            raise ValueError("resources must not be empty")

    def to_dict(self) -> dict[str, Any]:
        stmt: dict[str, Any] = {
            "Effect": self.effect,
            "Action": self.actions,
            "Resource": self.resources,
        }
        if self.principals:
            stmt["Principal"] = {"AWS": self.principals}
        if self.conditions:
            stmt["Condition"] = self.conditions
        return stmt


@dataclass
class IAMPolicyDoc:
    """IAM policy document builder following least-privilege principle.

    Composes a set of IAMStatements into a JSON-serialisable policy document.
    Use factory methods for common ML IAM patterns.
    """

    name: str
    description: str
    statements: list[IAMStatement] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")

    def add_statement(self, stmt: IAMStatement) -> "IAMPolicyDoc":
        """Append a statement and return self for chaining."""
        self.statements.append(stmt)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "Version": "2012-10-17",
            "Statement": [s.to_dict() for s in self.statements],
        }

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def sagemaker_execution_role(cls, account_id: str, region: str = "us-east-1") -> "IAMPolicyDoc":
        """Least-privilege SageMaker execution role policy."""
        doc = cls(
            name="SageMakerExecutionPolicy",
            description="Least-privilege policy for SageMaker training and inference roles",
        )
        doc.add_statement(IAMStatement(
            effect="Allow",
            actions=[
                "s3:GetObject", "s3:PutObject", "s3:ListBucket",
                "s3:GetBucketLocation",
            ],
            resources=[
                f"arn:aws:s3:::mlops-artifacts-{account_id}",
                f"arn:aws:s3:::mlops-artifacts-{account_id}/*",
            ],
        ))
        doc.add_statement(IAMStatement(
            effect="Allow",
            actions=[
                "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
                "ecr:BatchCheckLayerAvailability", "ecr:GetAuthorizationToken",
            ],
            resources=[f"arn:aws:ecr:{region}:{account_id}:repository/credit-risk-*"],
        ))
        doc.add_statement(IAMStatement(
            effect="Allow",
            actions=["cloudwatch:PutMetricData", "logs:CreateLogGroup",
                     "logs:CreateLogStream", "logs:PutLogEvents"],
            resources=["*"],
        ))
        return doc

    @classmethod
    def ecr_push_policy(cls, account_id: str, repo_name: str, region: str = "us-east-1") -> "IAMPolicyDoc":
        """Policy allowing CI/CD to push images to ECR."""
        doc = cls(
            name=f"ECRPushPolicy-{repo_name}",
            description=f"Allows push to ECR repository {repo_name}",
        )
        doc.add_statement(IAMStatement(
            effect="Allow",
            actions=[
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:InitiateLayerUpload",
                "ecr:UploadLayerPart",
                "ecr:CompleteLayerUpload",
                "ecr:PutImage",
            ],
            resources=[f"arn:aws:ecr:{region}:{account_id}:repository/{repo_name}"],
        ))
        return doc

    @classmethod
    def ml_data_scientist_policy(cls, account_id: str) -> "IAMPolicyDoc":
        """Policy for human ML practitioners — read-only on prod, full on dev."""
        doc = cls(
            name="MLDataScientistPolicy",
            description="Data scientist access: full dev, read-only prod",
        )
        doc.add_statement(IAMStatement(
            effect="Allow",
            actions=["sagemaker:*"],
            resources=["*"],
            conditions={"StringEquals": {"sagemaker:ResourceTag/Env": "dev"}},
        ))
        doc.add_statement(IAMStatement(
            effect="Allow",
            actions=["sagemaker:Describe*", "sagemaker:List*"],
            resources=["*"],
        ))
        doc.add_statement(IAMStatement(
            effect="Deny",
            actions=["sagemaker:Delete*", "sagemaker:Update*"],
            resources=["*"],
            conditions={"StringEquals": {"sagemaker:ResourceTag/Env": "prod"}},
        ))
        return doc


# ── ECR ──────────────────────────────────────────────────────────────────────


@dataclass
class ECRLifecycleRule:
    """ECR repository lifecycle rule — controls image retention.

    Keeps images clean and reduces storage costs by expiring old/untagged images.
    """

    rule_priority: int
    description: str
    tag_status: str  # "tagged" | "untagged" | "any"
    count_type: str  # "imageCountMoreThan" | "sinceImagePushed"
    count_number: int
    count_unit: str = "days"  # only for sinceImagePushed

    def __post_init__(self) -> None:
        if not self.description:
            raise ValueError("description must not be empty")
        if self.tag_status not in ("tagged", "untagged", "any"):
            raise ValueError(f"tag_status invalid: {self.tag_status!r}")
        if self.count_type not in ("imageCountMoreThan", "sinceImagePushed"):
            raise ValueError(f"count_type invalid: {self.count_type!r}")
        if self.count_number <= 0:
            raise ValueError("count_number must be positive")

    def to_dict(self) -> dict[str, Any]:
        selection: dict[str, Any] = {
            "tagStatus": self.tag_status,
            "countType": self.count_type,
            "countNumber": self.count_number,
        }
        if self.count_type == "sinceImagePushed":
            selection["countUnit"] = self.count_unit
        return {
            "rulePriority": self.rule_priority,
            "description": self.description,
            "selection": selection,
            "action": {"type": "expire"},
        }

    @classmethod
    def keep_last_n_tagged(cls, n: int = 10) -> "ECRLifecycleRule":
        """Expire tagged images beyond the last n."""
        return cls(
            rule_priority=1,
            description=f"Keep last {n} tagged images",
            tag_status="tagged",
            count_type="imageCountMoreThan",
            count_number=n,
        )

    @classmethod
    def expire_untagged_after_days(cls, days: int = 7) -> "ECRLifecycleRule":
        """Expire untagged images older than days."""
        return cls(
            rule_priority=2,
            description=f"Expire untagged images after {days} days",
            tag_status="untagged",
            count_type="sinceImagePushed",
            count_number=days,
            count_unit="days",
        )


@dataclass
class ECRRepository:
    """ECR repository configuration with lifecycle policy."""

    name: str
    account_id: str
    region: str = "us-east-1"
    lifecycle_rules: list[ECRLifecycleRule] = field(default_factory=list)
    scan_on_push: bool = True
    image_tag_mutability: str = "IMMUTABLE"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.account_id:
            raise ValueError("account_id must not be empty")

    @property
    def uri(self) -> str:
        return f"{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/{self.name}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "repositoryName": self.name,
            "imageTagMutability": self.image_tag_mutability,
            "imageScanningConfiguration": {"scanOnPush": self.scan_on_push},
            "lifecyclePolicy": {
                "rules": [r.to_dict() for r in self.lifecycle_rules],
            },
            "uri": self.uri,
        }


# ── VPC ──────────────────────────────────────────────────────────────────────


@dataclass
class SubnetConfig:
    """VPC subnet configuration."""

    name: str
    cidr: str
    availability_zone: str
    subnet_type: str = "private"  # "private" | "public"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.cidr:
            raise ValueError("cidr must not be empty")
        if self.subnet_type not in ("private", "public"):
            raise ValueError(f"subnet_type must be private or public, got: {self.subnet_type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cidr": self.cidr,
            "availabilityZone": self.availability_zone,
            "type": self.subnet_type,
        }


@dataclass
class VPCEndpoint:
    """VPC endpoint for private connectivity to AWS services (PrivateLink / Gateway)."""

    service: str  # e.g. "s3", "ecr.api", "ecr.dkr", "sagemaker.api"
    endpoint_type: str = "Interface"  # "Interface" | "Gateway"
    subnet_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.service:
            raise ValueError("service must not be empty")
        if self.endpoint_type not in ("Interface", "Gateway"):
            raise ValueError(f"endpoint_type must be Interface or Gateway, got: {self.endpoint_type!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": f"com.amazonaws.us-east-1.{self.service}",
            "type": self.endpoint_type,
            "subnetNames": self.subnet_names,
            "privateDnsEnabled": self.endpoint_type == "Interface",
        }


@dataclass
class VPCConfig:
    """VPC configuration for ML workloads — private subnets with S3/ECR endpoints.

    ML workloads should run in private subnets with no direct internet access.
    VPC endpoints route traffic to AWS services over the AWS backbone (cheaper,
    more secure, avoids NAT Gateway data charges).
    """

    name: str
    cidr_block: str
    subnets: list[SubnetConfig] = field(default_factory=list)
    endpoints: list[VPCEndpoint] = field(default_factory=list)
    enable_dns_hostnames: bool = True
    enable_dns_resolution: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.cidr_block:
            raise ValueError("cidr_block must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vpcName": self.name,
            "cidrBlock": self.cidr_block,
            "enableDnsHostnames": self.enable_dns_hostnames,
            "enableDnsResolution": self.enable_dns_resolution,
            "subnets": [s.to_dict() for s in self.subnets],
            "endpoints": [e.to_dict() for e in self.endpoints],
        }

    @classmethod
    def ml_platform_vpc(cls, name: str = "mlops-vpc") -> "VPCConfig":
        """Standard ML platform VPC: 2 private + 2 public subnets, S3/ECR endpoints."""
        vpc = cls(name=name, cidr_block="10.0.0.0/16")
        vpc.subnets = [
            SubnetConfig("private-a", "10.0.1.0/24", "us-east-1a", "private"),
            SubnetConfig("private-b", "10.0.2.0/24", "us-east-1b", "private"),
            SubnetConfig("public-a", "10.0.101.0/24", "us-east-1a", "public"),
            SubnetConfig("public-b", "10.0.102.0/24", "us-east-1b", "public"),
        ]
        vpc.endpoints = [
            VPCEndpoint("s3", "Gateway"),
            VPCEndpoint("ecr.api", "Interface", ["private-a", "private-b"]),
            VPCEndpoint("ecr.dkr", "Interface", ["private-a", "private-b"]),
            VPCEndpoint("sagemaker.api", "Interface", ["private-a", "private-b"]),
        ]
        return vpc

    @property
    def private_subnets(self) -> list[SubnetConfig]:
        return [s for s in self.subnets if s.subnet_type == "private"]

    @property
    def public_subnets(self) -> list[SubnetConfig]:
        return [s for s in self.subnets if s.subnet_type == "public"]
