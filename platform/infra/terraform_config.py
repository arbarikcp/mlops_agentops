"""Terraform configuration builders for ML infrastructure.

Day 86: Terraform provides multi-cloud infrastructure management with better
state handling and module ecosystem than AWS CloudFormation. Generates
equivalent of .tf JSON configurations as Python dicts for testing and
template generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Variables ─────────────────────────────────────────────────────────────────


@dataclass
class TFVariable:
    """Terraform variable declaration — parameterises a configuration.

    Variables enable reusable modules and separate environment-specific values
    from the infrastructure definition.
    """

    name: str
    var_type: str  # "string" | "number" | "bool" | "list" | "map" | "object"
    description: str
    default: Any = None
    sensitive: bool = False
    validation_condition: str = ""
    validation_error: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.var_type:
            raise ValueError("var_type must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "type": self.var_type,
            "description": self.description,
            "sensitive": self.sensitive,
        }
        if self.default is not None:
            cfg["default"] = self.default
        if self.validation_condition:
            cfg["validation"] = {
                "condition": self.validation_condition,
                "error_message": self.validation_error,
            }
        return {self.name: cfg}


# ── Resources ─────────────────────────────────────────────────────────────────


@dataclass
class TFResource:
    """Terraform resource block — a single managed infrastructure object.

    Resource types follow the provider_resource pattern:
    - aws_s3_bucket, aws_iam_role, aws_sagemaker_domain
    - google_storage_bucket, google_container_cluster
    """

    resource_type: str  # e.g. "aws_s3_bucket"
    resource_name: str  # local identifier within the TF config
    arguments: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)
    lifecycle: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.resource_type:
            raise ValueError("resource_type must not be empty")
        if not self.resource_name:
            raise ValueError("resource_name must not be empty")
        if not self.arguments:
            raise ValueError("arguments must not be empty")

    @property
    def ref(self) -> str:
        """Terraform reference string: resource_type.resource_name."""
        return f"{self.resource_type}.{self.resource_name}"

    @property
    def id_ref(self) -> str:
        """Terraform id attribute reference."""
        return f"{self.ref}.id"

    @property
    def arn_ref(self) -> str:
        """Terraform ARN attribute reference (AWS resources)."""
        return f"{self.ref}.arn"

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = dict(self.arguments)
        if self.depends_on:
            body["depends_on"] = self.depends_on
        if self.lifecycle:
            body["lifecycle"] = self.lifecycle
        return {"resource": {self.resource_type: {self.resource_name: body}}}

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def s3_bucket(cls, name: str, bucket_name_ref: str = "", versioning: bool = True) -> "TFResource":
        """S3 bucket with versioning and server-side encryption."""
        return cls(
            resource_type="aws_s3_bucket",
            resource_name=name,
            arguments={
                "bucket": bucket_name_ref or f"${{var.{name}_name}}",
                "versioning": {"enabled": versioning},
                "server_side_encryption_configuration": {
                    "rule": {
                        "apply_server_side_encryption_by_default": {
                            "sse_algorithm": "aws:kms"
                        }
                    }
                },
                "tags": {"ManagedBy": "terraform", "Project": "mlops"},
            },
            lifecycle={"prevent_destroy": True},
        )

    @classmethod
    def iam_role(cls, name: str, service: str) -> "TFResource":
        """IAM role with trust policy for an AWS service."""
        return cls(
            resource_type="aws_iam_role",
            resource_name=name,
            arguments={
                "name": f"${{var.prefix}}-{name}",
                "assume_role_policy": {
                    "Version": "2012-10-17",
                    "Statement": [{
                        "Effect": "Allow",
                        "Principal": {"Service": f"{service}.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }],
                },
                "tags": {"ManagedBy": "terraform"},
            },
        )

    @classmethod
    def ecr_repository(cls, name: str) -> "TFResource":
        """ECR repository with image scanning."""
        return cls(
            resource_type="aws_ecr_repository",
            resource_name=name,
            arguments={
                "name": f"${{var.prefix}}/{name}",
                "image_tag_mutability": "IMMUTABLE",
                "image_scanning_configuration": {"scan_on_push": True},
            },
        )

    @classmethod
    def sagemaker_domain(cls, name: str, vpc_ref: str, subnet_ref: str, role_ref: str) -> "TFResource":
        """SageMaker Studio domain."""
        return cls(
            resource_type="aws_sagemaker_domain",
            resource_name=name,
            arguments={
                "domain_name": f"${{var.prefix}}-domain",
                "auth_mode": "SSO",
                "vpc_id": f"${{{vpc_ref}}}",
                "subnet_ids": [f"${{{subnet_ref}}}"],
                "default_user_settings": {
                    "execution_role": f"${{{role_ref}}}",
                },
            },
        )


# ── Outputs ───────────────────────────────────────────────────────────────────


@dataclass
class TFOutput:
    """Terraform output — exposes resource attributes for cross-module use."""

    name: str
    value: str  # Terraform reference expression e.g. "aws_s3_bucket.mlops.arn"
    description: str
    sensitive: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must not be empty")
        if not self.value:
            raise ValueError("value must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "value": self.value,
            "description": self.description,
            "sensitive": self.sensitive,
        }
        return {self.name: cfg}


# ── Modules ───────────────────────────────────────────────────────────────────


@dataclass
class TFModule:
    """Terraform module call — encapsulates reusable infrastructure patterns.

    Modules are the primary reuse mechanism in Terraform. A module source
    can be a local path, Git URL, or Terraform Registry address.
    """

    module_name: str
    source: str
    version: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.module_name:
            raise ValueError("module_name must not be empty")
        if not self.source:
            raise ValueError("source must not be empty")

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"source": self.source}
        if self.version:
            cfg["version"] = self.version
        cfg.update(self.inputs)
        return {"module": {self.module_name: cfg}}

    @classmethod
    def vpc(cls, name: str, cidr: str) -> "TFModule":
        """AWS VPC module from Terraform Registry."""
        return cls(
            module_name=name,
            source="terraform-aws-modules/vpc/aws",
            version="5.0.0",
            inputs={
                "name": name,
                "cidr": cidr,
                "enable_nat_gateway": True,
                "single_nat_gateway": False,
                "enable_dns_hostnames": True,
            },
        )

    @classmethod
    def eks(cls, name: str, vpc_module_ref: str) -> "TFModule":
        """AWS EKS module."""
        return cls(
            module_name=name,
            source="terraform-aws-modules/eks/aws",
            version="20.0.0",
            inputs={
                "cluster_name": f"${{var.prefix}}-eks",
                "cluster_version": "1.29",
                "vpc_id": f"${{module.{vpc_module_ref}.vpc_id}}",
                "subnet_ids": f"${{module.{vpc_module_ref}.private_subnets}}",
            },
        )


# ── Complete Config ───────────────────────────────────────────────────────────


@dataclass
class TFConfig:
    """Complete Terraform configuration for an environment.

    Assembles variables, resources, modules, and outputs into a JSON-equivalent
    dict that maps 1:1 to a main.tf configuration file.
    """

    config_name: str
    provider: str  # "aws" | "google" | "azurerm"
    provider_config: dict[str, Any]
    variables: list[TFVariable] = field(default_factory=list)
    resources: list[TFResource] = field(default_factory=list)
    modules: list[TFModule] = field(default_factory=list)
    outputs: list[TFOutput] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.config_name:
            raise ValueError("config_name must not be empty")
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.provider_config:
            raise ValueError("provider_config must not be empty")

    def add_variable(self, var: TFVariable) -> "TFConfig":
        self.variables.append(var)
        return self

    def add_resource(self, resource: TFResource) -> "TFConfig":
        self.resources.append(resource)
        return self

    def add_module(self, module: TFModule) -> "TFConfig":
        self.modules.append(module)
        return self

    def add_output(self, output: TFOutput) -> "TFConfig":
        self.outputs.append(output)
        return self

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "terraform": {
                "required_providers": {
                    self.provider: self.provider_config,
                },
                "backend": {"s3": {"bucket": "${var.tf_state_bucket}", "key": f"{self.config_name}/terraform.tfstate"}},
            },
            "provider": {self.provider: {}},
            "variable": {},
            "resource": {},
            "module": {},
            "output": {},
        }
        for var in self.variables:
            cfg["variable"].update(var.to_dict())
        for resource in self.resources:
            r = resource.to_dict()["resource"]
            for rtype, rvals in r.items():
                cfg["resource"].setdefault(rtype, {}).update(rvals)
        for module in self.modules:
            cfg["module"].update(module.to_dict()["module"])
        for output in self.outputs:
            cfg["output"].update(output.to_dict())
        return cfg

    @classmethod
    def ml_platform_aws(cls) -> "TFConfig":
        """Factory: standard ML platform AWS Terraform configuration."""
        config = cls(
            config_name="ml-platform",
            provider="aws",
            provider_config={
                "source": "hashicorp/aws",
                "version": "~> 5.0",
            },
        )
        config.add_variable(TFVariable("prefix", "string", "Resource name prefix", default="mlops"))
        config.add_variable(TFVariable("region", "string", "AWS region", default="us-east-1"))
        config.add_variable(TFVariable("tf_state_bucket", "string", "S3 bucket for TF state"))
        config.add_resource(TFResource.s3_bucket("artifacts"))
        config.add_resource(TFResource.ecr_repository("credit-risk"))
        config.add_resource(TFResource.iam_role("sagemaker_role", "sagemaker"))
        config.add_output(TFOutput(
            "artifacts_bucket_arn",
            "aws_s3_bucket.artifacts.arn",
            "ARN of the ML artifacts S3 bucket",
        ))
        return config
