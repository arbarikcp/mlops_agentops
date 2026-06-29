"""AWS serving — EKS inference config and Bedrock managed foundation models.

Day 84: EKS beats SageMaker endpoints when you need custom runtimes, GPU sharing
(MIG/MPS), or cost optimisation at scale. Bedrock provides managed foundation
models via API with no serving infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── EKS Inference ─────────────────────────────────────────────────────────────


@dataclass
class EKSResourceSpec:
    """Kubernetes resource requests and limits for inference containers."""

    cpu_request: str = "500m"
    memory_request: str = "1Gi"
    cpu_limit: str = "2"
    memory_limit: str = "4Gi"
    gpu_count: int = 0  # 0 = CPU-only

    def __post_init__(self) -> None:
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        limits: dict[str, Any] = {
            "cpu": self.cpu_limit,
            "memory": self.memory_limit,
        }
        requests: dict[str, Any] = {
            "cpu": self.cpu_request,
            "memory": self.memory_request,
        }
        if self.gpu_count > 0:
            limits["nvidia.com/gpu"] = str(self.gpu_count)
            requests["nvidia.com/gpu"] = str(self.gpu_count)
        return {"requests": requests, "limits": limits}


@dataclass
class EKSInferenceConfig:
    """EKS inference deployment spec for an ML model.

    Use EKS over SageMaker endpoints when:
    - Custom runtime needed (e.g. vLLM, Triton with custom backends)
    - GPU sharing (MIG partitioning, CUDA MPS)
    - Multi-model serving on same instance for cost
    - Existing K8s operational expertise and tooling
    - Cost at scale: SageMaker endpoint overhead ~20% vs raw EC2
    """

    deployment_name: str
    namespace: str
    image_uri: str
    model_s3_uri: str
    replicas: int
    resources: EKSResourceSpec
    service_port: int = 8080
    health_check_path: str = "/health"
    model_download_init_container: bool = True
    node_selector: dict[str, str] = field(default_factory=dict)
    tolerations: list[dict[str, str]] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.deployment_name:
            raise ValueError("deployment_name must not be empty")
        if not self.namespace:
            raise ValueError("namespace must not be empty")
        if not self.image_uri:
            raise ValueError("image_uri must not be empty")
        if not self.model_s3_uri:
            raise ValueError("model_s3_uri must not be empty")
        if self.replicas < 1:
            raise ValueError("replicas must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        containers: list[dict[str, Any]] = [
            {
                "name": self.deployment_name,
                "image": self.image_uri,
                "ports": [{"containerPort": self.service_port}],
                "resources": self.resources.to_dict(),
                "env": [{"name": k, "value": v} for k, v in self.env_vars.items()]
                + [{"name": "MODEL_S3_URI", "value": self.model_s3_uri}],
                "livenessProbe": {
                    "httpGet": {"path": self.health_check_path, "port": self.service_port},
                    "initialDelaySeconds": 30,
                    "periodSeconds": 10,
                },
                "readinessProbe": {
                    "httpGet": {"path": self.health_check_path, "port": self.service_port},
                    "initialDelaySeconds": 10,
                    "periodSeconds": 5,
                },
            }
        ]

        spec: dict[str, Any] = {
            "containers": containers,
        }
        if self.node_selector:
            spec["nodeSelector"] = self.node_selector
        if self.tolerations:
            spec["tolerations"] = self.tolerations

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.deployment_name,
                "namespace": self.namespace,
                "labels": {**{"app": self.deployment_name}, **self.labels},
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {"matchLabels": {"app": self.deployment_name}},
                "template": {
                    "metadata": {"labels": {"app": self.deployment_name}},
                    "spec": spec,
                },
            },
        }

    def to_service_dict(self) -> dict[str, Any]:
        """Companion Kubernetes Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.deployment_name}-svc",
                "namespace": self.namespace,
            },
            "spec": {
                "selector": {"app": self.deployment_name},
                "ports": [{"port": 80, "targetPort": self.service_port}],
                "type": "ClusterIP",
            },
        }

    @classmethod
    def cpu_inference(
        cls,
        name: str,
        image_uri: str,
        model_s3_uri: str,
        replicas: int = 2,
    ) -> "EKSInferenceConfig":
        """Factory: CPU-only inference deployment."""
        return cls(
            deployment_name=name,
            namespace="ml-serving",
            image_uri=image_uri,
            model_s3_uri=model_s3_uri,
            replicas=replicas,
            resources=EKSResourceSpec(cpu_request="500m", cpu_limit="2", memory_limit="2Gi"),
        )

    @classmethod
    def gpu_inference(
        cls,
        name: str,
        image_uri: str,
        model_s3_uri: str,
        gpu_count: int = 1,
    ) -> "EKSInferenceConfig":
        """Factory: GPU inference deployment with NVIDIA tolerations."""
        return cls(
            deployment_name=name,
            namespace="ml-serving",
            image_uri=image_uri,
            model_s3_uri=model_s3_uri,
            replicas=1,
            resources=EKSResourceSpec(
                cpu_request="2", memory_request="8Gi",
                cpu_limit="4", memory_limit="16Gi",
                gpu_count=gpu_count,
            ),
            node_selector={"accelerator": "nvidia-gpu"},
            tolerations=[{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
        )


# ── Bedrock ───────────────────────────────────────────────────────────────────


@dataclass
class BedrockGuardrailConfig:
    """Bedrock guardrail — content filtering for foundation model responses."""

    guardrail_name: str
    topics_to_deny: list[str] = field(default_factory=list)
    pii_action: str = "ANONYMIZE"  # "ANONYMIZE" | "BLOCK"
    hate_speech_threshold: str = "HIGH"  # "LOW" | "MEDIUM" | "HIGH"

    def __post_init__(self) -> None:
        if not self.guardrail_name:
            raise ValueError("guardrail_name must not be empty")
        if self.pii_action not in ("ANONYMIZE", "BLOCK"):
            raise ValueError(f"pii_action invalid: {self.pii_action!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.guardrail_name,
            "topicPolicyConfig": {
                "topicsConfig": [{"name": t, "type": "DENY"} for t in self.topics_to_deny]
            },
            "sensitiveInformationPolicyConfig": {
                "piiEntitiesConfig": [{"type": "EMAIL", "action": self.pii_action}]
            },
            "contentPolicyConfig": {
                "filtersConfig": [{"type": "HATE", "inputStrength": self.hate_speech_threshold}]
            },
        }


@dataclass
class BedrockConfig:
    """AWS Bedrock configuration — managed foundation model via API.

    Bedrock provides serverless access to foundation models (Claude, Titan,
    Llama, Mistral, etc.) without any infrastructure management. Trade-off:
    higher per-token cost vs self-hosted but zero operational overhead.
    """

    model_id: str
    region: str = "us-east-1"
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    system_prompt: str = ""
    guardrail: BedrockGuardrailConfig | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError("temperature must be in [0, 1]")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    def invoke_body(self, prompt: str) -> dict[str, Any]:
        """Build the request body for bedrock:InvokeModel."""
        body: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system_prompt:
            body["system"] = self.system_prompt
        return body

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "modelId": self.model_id,
            "region": self.region,
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            },
            "tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }
        if self.system_prompt:
            cfg["systemPrompt"] = self.system_prompt
        if self.guardrail:
            cfg["guardrailConfig"] = self.guardrail.to_dict()
        return cfg

    @classmethod
    def claude_sonnet(cls, system_prompt: str = "") -> "BedrockConfig":
        """Factory: Claude 3 Sonnet via Bedrock."""
        return cls(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            system_prompt=system_prompt,
            max_tokens=2048,
            temperature=0.5,
        )

    @classmethod
    def titan_embedding(cls) -> "BedrockConfig":
        """Factory: Amazon Titan text embedding model."""
        return cls(
            model_id="amazon.titan-embed-text-v1",
            max_tokens=8192,
            temperature=0.0,
        )
