"""
vllm_k8s.py — vLLM on Kubernetes + Capacity Planning (Day 99)

Covers K8s Deployment, Service, HPA, PodMonitor manifests
and capacity planning for production vLLM serving.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from llm.vllm_config import VLLMEngineConfig


_VALID_SERVICE_TYPES = {"ClusterIP", "NodePort", "LoadBalancer"}


@dataclass
class VLLMDeploymentSpec:
    """Kubernetes Deployment specification for vLLM."""

    name: str
    image: str
    engine_config: VLLMEngineConfig
    replicas: int = 1
    gpu_resource: str = "nvidia.com/gpu"
    memory_limit: str = "24Gi"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.image:
            raise ValueError("image must be non-empty")
        if self.replicas < 1:
            raise ValueError(f"replicas must be >= 1, got {self.replicas}")

    def to_manifest(self) -> dict:
        """Generate Kubernetes Deployment manifest dict."""
        gpu_count = self.engine_config.total_parallel_size()
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.name,
                "labels": {"app": self.name},
            },
            "spec": {
                "replicas": self.replicas,
                "selector": {"matchLabels": {"app": self.name}},
                "template": {
                    "metadata": {"labels": {"app": self.name}},
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm",
                                "image": self.image,
                                "ports": [{"containerPort": 8000, "name": "http"}],
                                "resources": {
                                    "requests": {
                                        self.gpu_resource: str(gpu_count),
                                        "memory": self.memory_limit,
                                    },
                                    "limits": {
                                        self.gpu_resource: str(gpu_count),
                                        "memory": self.memory_limit,
                                    },
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/health", "port": 8000},
                                    "initialDelaySeconds": 15,
                                    "periodSeconds": 5,
                                },
                            }
                        ],
                        "tolerations": [
                            {
                                "key": "nvidia.com/gpu",
                                "operator": "Exists",
                                "effect": "NoSchedule",
                            }
                        ],
                    },
                },
            },
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "image": self.image,
            "engine_config": self.engine_config.to_dict(),
            "replicas": self.replicas,
            "gpu_resource": self.gpu_resource,
            "memory_limit": self.memory_limit,
        }


@dataclass
class VLLMServiceSpec:
    """Kubernetes Service specification for vLLM."""

    name: str
    deployment_name: str
    port: int = 8000
    service_type: str = "ClusterIP"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.deployment_name:
            raise ValueError("deployment_name must be non-empty")
        if self.service_type not in _VALID_SERVICE_TYPES:
            raise ValueError(
                f"service_type must be one of {_VALID_SERVICE_TYPES}, "
                f"got {self.service_type!r}"
            )

    def to_manifest(self) -> dict:
        """Generate Kubernetes Service manifest dict."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": self.name},
            "spec": {
                "type": self.service_type,
                "selector": {"app": self.deployment_name},
                "ports": [
                    {
                        "name": "http",
                        "protocol": "TCP",
                        "port": self.port,
                        "targetPort": 8000,
                    }
                ],
            },
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "deployment_name": self.deployment_name,
            "port": self.port,
            "service_type": self.service_type,
        }


@dataclass
class VLLMHPASpec:
    """Kubernetes HPA spec with custom vLLM throughput metric."""

    name: str
    deployment_name: str
    min_replicas: int = 1
    max_replicas: int = 10
    target_rps: float = 10.0

    def __post_init__(self) -> None:
        if self.min_replicas > self.max_replicas:
            raise ValueError(
                f"min_replicas ({self.min_replicas}) must be <= "
                f"max_replicas ({self.max_replicas})"
            )
        if self.target_rps <= 0:
            raise ValueError(f"target_rps must be > 0, got {self.target_rps}")

    def to_manifest(self) -> dict:
        """Generate HPA manifest with custom vllm_request_rate metric."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": self.name},
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.deployment_name,
                },
                "minReplicas": self.min_replicas,
                "maxReplicas": self.max_replicas,
                "metrics": [
                    {
                        "type": "Pods",
                        "pods": {
                            "metric": {"name": "vllm_request_rate"},
                            "target": {
                                "type": "AverageValue",
                                "averageValue": str(self.target_rps),
                            },
                        },
                    }
                ],
            },
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "deployment_name": self.deployment_name,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "target_rps": self.target_rps,
        }


@dataclass
class CapacityPlan:
    """Capacity planning for vLLM serving fleet."""

    target_rps: float
    single_replica_throughput: float
    safety_factor: float = 1.2
    gpu_cost_per_hour: float = 3.0

    def __post_init__(self) -> None:
        for name, val in [
            ("target_rps", self.target_rps),
            ("single_replica_throughput", self.single_replica_throughput),
            ("safety_factor", self.safety_factor),
            ("gpu_cost_per_hour", self.gpu_cost_per_hour),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if self.safety_factor < 1.0:
            raise ValueError(
                f"safety_factor must be >= 1.0, got {self.safety_factor}"
            )

    def replicas_needed(self) -> int:
        """Replicas needed = ceil(target_rps * safety_factor / single_replica_throughput)."""
        return math.ceil(
            self.target_rps * self.safety_factor / self.single_replica_throughput
        )

    def hourly_cost_usd(self) -> float:
        """Hourly fleet cost."""
        return self.replicas_needed() * self.gpu_cost_per_hour

    def to_dict(self) -> dict:
        return {
            "target_rps": self.target_rps,
            "single_replica_throughput": self.single_replica_throughput,
            "safety_factor": self.safety_factor,
            "gpu_cost_per_hour": self.gpu_cost_per_hour,
            "replicas_needed": self.replicas_needed(),
            "hourly_cost_usd": self.hourly_cost_usd(),
        }


@dataclass
class PodMonitorSpec:
    """Prometheus PodMonitor CRD for scraping vLLM metrics."""

    name: str
    namespace: str = "ml-serving"
    port_name: str = "metrics"
    scrape_interval: str = "15s"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")

    def to_manifest(self) -> dict:
        """Generate PodMonitor CRD manifest dict."""
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "PodMonitor",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "spec": {
                "podMetricsEndpoints": [
                    {
                        "port": self.port_name,
                        "interval": self.scrape_interval,
                        "path": "/metrics",
                    }
                ],
                "selector": {
                    "matchLabels": {"app": self.name},
                },
                "namespaceSelector": {
                    "matchNames": [self.namespace],
                },
            },
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "namespace": self.namespace,
            "port_name": self.port_name,
            "scrape_interval": self.scrape_interval,
        }
