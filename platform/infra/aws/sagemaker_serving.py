"""SageMaker model registry and endpoint builders.

Day 81: Four endpoint types cover different latency/throughput tradeoffs:
  - Real-time: low-latency synchronous inference (p99 < 200ms)
  - Serverless: bursty/infrequent traffic, zero idle cost
  - Async: long-running inference (>60s), queue-based
  - Batch Transform: offline bulk scoring, no persistent endpoint
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EndpointType(str, Enum):
    REAL_TIME = "RealTime"
    SERVERLESS = "Serverless"
    ASYNC = "Async"
    BATCH_TRANSFORM = "BatchTransform"


# ── Model Package ─────────────────────────────────────────────────────────────


@dataclass
class SMModelPackage:
    """SageMaker Model Package — a versioned, approvable artifact in the registry.

    Separates model artifacts from deployment configuration, enabling a human
    approval gate before any model reaches production.
    """

    model_package_group_name: str
    model_description: str
    inference_image_uri: str
    model_s3_uri: str
    supported_content_types: list[str] = field(default_factory=lambda: ["text/csv"])
    supported_response_types: list[str] = field(default_factory=lambda: ["application/json"])
    approval_status: str = "PendingManualApproval"
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_package_group_name:
            raise ValueError("model_package_group_name must not be empty")
        if not self.inference_image_uri:
            raise ValueError("inference_image_uri must not be empty")
        if not self.model_s3_uri:
            raise ValueError("model_s3_uri must not be empty")
        if not self.model_description:
            raise ValueError("model_description must not be empty")
        if self.approval_status not in ("PendingManualApproval", "Approved", "Rejected"):
            raise ValueError(f"approval_status invalid: {self.approval_status!r}")

    def approve(self) -> "SMModelPackage":
        self.approval_status = "Approved"
        return self

    def reject(self) -> "SMModelPackage":
        self.approval_status = "Rejected"
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "ModelPackageGroupName": self.model_package_group_name,
            "ModelPackageDescription": self.model_description,
            "InferenceSpecification": {
                "Containers": [{
                    "Image": self.inference_image_uri,
                    "ModelDataUrl": self.model_s3_uri,
                }],
                "SupportedContentTypes": self.supported_content_types,
                "SupportedResponseMIMETypes": self.supported_response_types,
            },
            "ModelApprovalStatus": self.approval_status,
            "CustomerMetadataProperties": {k: str(v) for k, v in self.metrics.items()},
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }


# ── Endpoint Configs ───────────────────────────────────────────────────────────


@dataclass
class SMEndpointConfig:
    """SageMaker endpoint configuration — describes how to serve a model.

    Supports all four endpoint types. Create via factory methods for the
    appropriate type.
    """

    config_name: str
    endpoint_type: EndpointType
    model_package_arn: str
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    # Serverless-specific
    serverless_memory_mb: int = 2048
    serverless_max_concurrency: int = 10
    # Async-specific
    async_output_s3_uri: str = ""
    async_max_concurrent_invocations: int = 5
    # Batch-specific
    batch_instance_type: str = "ml.m5.xlarge"
    batch_instance_count: int = 1
    data_capture_s3_uri: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.config_name:
            raise ValueError("config_name must not be empty")
        if not self.model_package_arn:
            raise ValueError("model_package_arn must not be empty")

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "EndpointConfigName": self.config_name,
            "EndpointType": self.endpoint_type.value,
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }

        if self.endpoint_type == EndpointType.REAL_TIME:
            cfg["ProductionVariants"] = [{
                "VariantName": "AllTraffic",
                "ModelPackageArn": self.model_package_arn,
                "InstanceType": self.instance_type,
                "InitialInstanceCount": self.instance_count,
                "InitialVariantWeight": 1.0,
            }]
            if self.data_capture_s3_uri:
                cfg["DataCaptureConfig"] = {
                    "EnableCapture": True,
                    "InitialSamplingPercentage": 20,
                    "DestinationS3Uri": self.data_capture_s3_uri,
                    "CaptureOptions": [
                        {"CaptureMode": "Input"},
                        {"CaptureMode": "Output"},
                    ],
                }

        elif self.endpoint_type == EndpointType.SERVERLESS:
            cfg["ProductionVariants"] = [{
                "VariantName": "AllTraffic",
                "ModelPackageArn": self.model_package_arn,
                "ServerlessConfig": {
                    "MemorySizeInMB": self.serverless_memory_mb,
                    "MaxConcurrency": self.serverless_max_concurrency,
                },
            }]

        elif self.endpoint_type == EndpointType.ASYNC:
            cfg["ProductionVariants"] = [{
                "VariantName": "AllTraffic",
                "ModelPackageArn": self.model_package_arn,
                "InstanceType": self.instance_type,
                "InitialInstanceCount": self.instance_count,
            }]
            cfg["AsyncInferenceConfig"] = {
                "OutputConfig": {"S3OutputPath": self.async_output_s3_uri},
                "ClientConfig": {
                    "MaxConcurrentInvocationsPerInstance": self.async_max_concurrent_invocations,
                },
            }

        elif self.endpoint_type == EndpointType.BATCH_TRANSFORM:
            # Batch Transform doesn't use an endpoint config in the traditional sense
            # but we represent it here for consistency
            cfg["BatchTransformConfig"] = {
                "InstanceType": self.batch_instance_type,
                "InstanceCount": self.batch_instance_count,
                "ModelPackageArn": self.model_package_arn,
            }

        return cfg

    # ── Factory methods ──────────────────────────────────────────────────────

    @classmethod
    def real_time(
        cls,
        config_name: str,
        model_package_arn: str,
        instance_type: str = "ml.m5.large",
        data_capture_s3: str = "",
    ) -> "SMEndpointConfig":
        return cls(
            config_name=config_name,
            endpoint_type=EndpointType.REAL_TIME,
            model_package_arn=model_package_arn,
            instance_type=instance_type,
            data_capture_s3_uri=data_capture_s3,
        )

    @classmethod
    def serverless(
        cls,
        config_name: str,
        model_package_arn: str,
        memory_mb: int = 2048,
        max_concurrency: int = 10,
    ) -> "SMEndpointConfig":
        return cls(
            config_name=config_name,
            endpoint_type=EndpointType.SERVERLESS,
            model_package_arn=model_package_arn,
            serverless_memory_mb=memory_mb,
            serverless_max_concurrency=max_concurrency,
        )

    @classmethod
    def async_inference(
        cls,
        config_name: str,
        model_package_arn: str,
        output_s3: str,
        instance_type: str = "ml.m5.large",
    ) -> "SMEndpointConfig":
        return cls(
            config_name=config_name,
            endpoint_type=EndpointType.ASYNC,
            model_package_arn=model_package_arn,
            async_output_s3_uri=output_s3,
            instance_type=instance_type,
        )

    @classmethod
    def batch_transform(
        cls,
        config_name: str,
        model_package_arn: str,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 2,
    ) -> "SMEndpointConfig":
        return cls(
            config_name=config_name,
            endpoint_type=EndpointType.BATCH_TRANSFORM,
            model_package_arn=model_package_arn,
            batch_instance_type=instance_type,
            batch_instance_count=instance_count,
        )


# ── Endpoint ──────────────────────────────────────────────────────────────────


@dataclass
class SMEndpoint:
    """SageMaker endpoint — the live serving resource.

    Decouples model configuration (EndpointConfig) from the deployment target
    so configs can be pre-created and swapped atomically (blue/green).
    """

    endpoint_name: str
    endpoint_config_name: str
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.endpoint_name:
            raise ValueError("endpoint_name must not be empty")
        if not self.endpoint_config_name:
            raise ValueError("endpoint_config_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "EndpointName": self.endpoint_name,
            "EndpointConfigName": self.endpoint_config_name,
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }

    def update_config(self, new_config_name: str) -> dict[str, Any]:
        """Return the update request dict (atomic blue/green swap)."""
        return {
            "EndpointName": self.endpoint_name,
            "EndpointConfigName": new_config_name,
        }
