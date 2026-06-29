"""K8s autoscaling and Kueue manifest builders.

Days 67–68 — HPASpec, KEDAScaledObject, and KueueJobConfig generate manifest
dicts for CPU/memory HPA, queue-depth KEDA scaling, and Kueue-enabled training
jobs.

Classes:
  HPAMetric          — one HPA metric (Resource or External)
  HPASpec            — HorizontalPodAutoscaler manifest builder
  KEDAScaledObject   — KEDA ScaledObject manifest builder (SQS / Prometheus)
  KueueJobConfig     — Wraps a Job spec with Kueue queue annotation

See: docs/phase9/day67_autoscaling.md, docs/phase9/day68_kueue.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── HPAMetric ─────────────────────────────────────────────────────────────────

@dataclass
class HPAMetric:
    """One HPA metric specification.

    Attributes:
        resource:            "cpu" or "memory".
        target_utilization:  Integer percentage (e.g., 70 for 70%).
    """

    resource: str
    target_utilization: int

    _VALID = {"cpu", "memory"}

    def __post_init__(self) -> None:
        if self.resource not in self._VALID:
            raise ValueError(f"resource must be one of {self._VALID}; got {self.resource!r}")
        if not (1 <= self.target_utilization <= 100):
            raise ValueError("target_utilization must be 1–100")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Resource",
            "resource": {
                "name": self.resource,
                "target": {
                    "type": "Utilization",
                    "averageUtilization": self.target_utilization,
                },
            },
        }


# ── HPASpec ───────────────────────────────────────────────────────────────────

@dataclass
class HPASpec:
    """HorizontalPodAutoscaler manifest builder.

    Attributes:
        name:            HPA resource name.
        namespace:       Target namespace.
        deployment_name: Name of the Deployment to scale.
        min_replicas:    Minimum replicas.
        max_replicas:    Maximum replicas.
        metrics:         List of HPAMetric objects.
    """

    name: str
    namespace: str = "ml-serving"
    deployment_name: str = ""
    min_replicas: int = 2
    max_replicas: int = 20
    metrics: list[HPAMetric] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HPASpec.name cannot be empty")
        if self.min_replicas < 1:
            raise ValueError("min_replicas must be >= 1")
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas cannot exceed max_replicas")
        if not self.deployment_name:
            self.deployment_name = self.name

    def to_manifest(self) -> dict[str, Any]:
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.deployment_name,
                },
                "minReplicas": self.min_replicas,
                "maxReplicas": self.max_replicas,
                "metrics": [m.to_dict() for m in self.metrics],
            },
        }


# ── KEDAScaledObject ──────────────────────────────────────────────────────────

@dataclass
class KEDAScaledObject:
    """KEDA ScaledObject manifest builder.

    Supports SQS queue depth and Prometheus custom metric triggers.

    Attributes:
        name:              ScaledObject name.
        namespace:         Target namespace.
        deployment_name:   Deployment to scale.
        trigger_type:      "aws-sqs-queue" or "prometheus".
        min_replicas:      Minimum replicas (0 = scale-to-zero).
        max_replicas:      Maximum replicas.
        trigger_metadata:  Trigger-specific key-value metadata.
    """

    name: str
    namespace: str = "ml-serving"
    deployment_name: str = ""
    trigger_type: str = "aws-sqs-queue"
    min_replicas: int = 0
    max_replicas: int = 50
    trigger_metadata: dict[str, str] = field(default_factory=dict)

    _VALID_TRIGGERS = {"aws-sqs-queue", "prometheus", "redis", "kafka"}

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("KEDAScaledObject.name cannot be empty")
        if self.trigger_type not in self._VALID_TRIGGERS:
            raise ValueError(f"trigger_type must be one of {self._VALID_TRIGGERS}")
        if self.min_replicas < 0:
            raise ValueError("min_replicas cannot be negative")
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas cannot exceed max_replicas")
        if not self.deployment_name:
            self.deployment_name = self.name

    def to_manifest(self) -> dict[str, Any]:
        return {
            "apiVersion": "keda.sh/v1alpha1",
            "kind": "ScaledObject",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "scaleTargetRef": {"name": self.deployment_name},
                "minReplicaCount": self.min_replicas,
                "maxReplicaCount": self.max_replicas,
                "triggers": [{
                    "type": self.trigger_type,
                    "metadata": self.trigger_metadata,
                }],
            },
        }


# ── KueueJobConfig ────────────────────────────────────────────────────────────

@dataclass
class KueueJobConfig:
    """Wraps a K8s Job spec with Kueue queue annotation and GPU resources.

    Attributes:
        job_name:      Job name.
        namespace:     Target namespace.
        queue_name:    Kueue LocalQueue name.
        image:         Training container image.
        gpu_count:     Number of GPUs requested.
        cpu_request:   CPU request string.
        memory_request: Memory request string.
        command:       Container command.
    """

    job_name: str
    namespace: str = "ml-training"
    queue_name: str = "default-queue"
    image: str = ""
    gpu_count: int = 1
    cpu_request: str = "4"
    memory_request: str = "16Gi"
    command: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.job_name:
            raise ValueError("KueueJobConfig.job_name cannot be empty")
        if self.gpu_count < 0:
            raise ValueError("gpu_count cannot be negative")

    def to_manifest(self) -> dict[str, Any]:
        resources: dict[str, Any] = {
            "requests": {
                "cpu": self.cpu_request,
                "memory": self.memory_request,
            },
            "limits": {},
        }
        if self.gpu_count > 0:
            resources["requests"]["nvidia.com/gpu"] = str(self.gpu_count)
            resources["limits"]["nvidia.com/gpu"] = str(self.gpu_count)

        container: dict[str, Any] = {
            "name": self.job_name,
            "image": self.image,
            "resources": resources,
        }
        if self.command:
            container["command"] = self.command

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": self.job_name,
                "namespace": self.namespace,
                "labels": {
                    "kueue.x-k8s.io/queue-name": self.queue_name,
                },
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [container],
                        "restartPolicy": "OnFailure",
                    }
                }
            },
        }
