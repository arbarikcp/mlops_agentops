"""Kubernetes manifest builder for ML serving workloads.

Day 59 — generates typed K8s manifest dicts (Deployment, Service, ConfigMap,
Namespace). Manifests are returned as Python dicts compatible with PyYAML's
`yaml.dump()`. No kubectl or cluster connection required.

Classes:
  ResourceRequirements  — CPU/memory/GPU requests + limits
  ContainerSpec         — one container definition (image, port, env, resources)
  ProbeSpec             — liveness / readiness HTTP probe config
  DeploymentSpec        — Deployment manifest builder (rolling update, init-containers)
  ServiceSpec           — Service manifest builder (ClusterIP / NodePort / LoadBalancer)
  K8sManifestSet        — groups Deployment + Service + optional ConfigMap

See: docs/phase9/day59_k8s_fundamentals.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── ResourceRequirements ──────────────────────────────────────────────────────

@dataclass
class ResourceRequirements:
    """CPU, memory, and optional GPU resource requests and limits.

    Attributes:
        cpu_request:    Kubernetes CPU request string (e.g., "500m", "1").
        memory_request: Kubernetes memory request string (e.g., "512Mi").
        cpu_limit:      Kubernetes CPU limit string.
        memory_limit:   Kubernetes memory limit string.
        gpu_count:      Number of NVIDIA GPUs (0 = no GPU).
    """

    cpu_request: str = "500m"
    memory_request: str = "512Mi"
    cpu_limit: str = "2"
    memory_limit: str = "2Gi"
    gpu_count: int = 0

    def __post_init__(self) -> None:
        if self.gpu_count < 0:
            raise ValueError("gpu_count cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        requests: dict[str, str] = {
            "cpu": self.cpu_request,
            "memory": self.memory_request,
        }
        limits: dict[str, str] = {
            "cpu": self.cpu_limit,
            "memory": self.memory_limit,
        }
        if self.gpu_count > 0:
            requests["nvidia.com/gpu"] = str(self.gpu_count)
            limits["nvidia.com/gpu"] = str(self.gpu_count)
        return {"requests": requests, "limits": limits}


# ── ProbeSpec ─────────────────────────────────────────────────────────────────

@dataclass
class ProbeSpec:
    """HTTP liveness or readiness probe configuration.

    Attributes:
        path:                 HTTP GET path (default "/health").
        port:                 Port to probe.
        initial_delay_s:      Seconds before first probe (default 15).
        period_s:             Seconds between probes (default 10).
        failure_threshold:    Failures before action (default 3).
    """

    path: str = "/health"
    port: int = 8080
    initial_delay_s: int = 15
    period_s: int = 10
    failure_threshold: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "httpGet": {"path": self.path, "port": self.port},
            "initialDelaySeconds": self.initial_delay_s,
            "periodSeconds": self.period_s,
            "failureThreshold": self.failure_threshold,
        }


# ── ContainerSpec ─────────────────────────────────────────────────────────────

@dataclass
class ContainerSpec:
    """One container definition inside a pod spec.

    Attributes:
        name:       Container name.
        image:      Docker image reference (e.g., "credit-risk-api:v1").
        port:       Container port to expose (0 = none).
        env:        Environment variable dict (key → value; all serialised as strings).
        resources:  CPU/memory/GPU requirements.
        command:    Optional command override (empty = use image CMD).
        is_init:    True if this is an init-container.
    """

    name: str
    image: str
    port: int = 0
    env: dict[str, str] = field(default_factory=dict)
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    command: list[str] = field(default_factory=list)
    is_init: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ContainerSpec.name cannot be empty")
        if not self.image:
            raise ValueError("ContainerSpec.image cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "resources": self.resources.to_dict(),
        }
        if self.port:
            d["ports"] = [{"containerPort": self.port, "name": "http"}]
        if self.env:
            d["env"] = [{"name": k, "value": str(v)} for k, v in self.env.items()]
        if self.command:
            d["command"] = self.command
        return d


# ── DeploymentSpec ────────────────────────────────────────────────────────────

@dataclass
class DeploymentSpec:
    """Kubernetes Deployment manifest builder.

    Attributes:
        name:           Deployment name.
        namespace:      Target namespace (default "ml-serving").
        replicas:       Desired replica count (default 3).
        containers:     Main containers (at least one required).
        init_containers: Init-containers run before main containers start.
        labels:         Additional metadata labels.
        max_unavailable: Rolling update — max pods unavailable (default 1).
        max_surge:      Rolling update — max extra pods (default 1).
    """

    name: str
    namespace: str = "ml-serving"
    replicas: int = 3
    containers: list[ContainerSpec] = field(default_factory=list)
    init_containers: list[ContainerSpec] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    max_unavailable: int = 1
    max_surge: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DeploymentSpec.name cannot be empty")
        if self.replicas < 1:
            raise ValueError("replicas must be >= 1")

    def _pod_labels(self) -> dict[str, str]:
        base = {"app": self.name}
        base.update(self.labels)
        return base

    def to_manifest(self) -> dict[str, Any]:
        """Return the Deployment manifest as a Python dict."""
        pod_labels = self._pod_labels()
        spec: dict[str, Any] = {
            "replicas": self.replicas,
            "selector": {"matchLabels": {"app": self.name}},
            "strategy": {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxUnavailable": self.max_unavailable,
                    "maxSurge": self.max_surge,
                },
            },
            "template": {
                "metadata": {"labels": pod_labels},
                "spec": {
                    "containers": [c.to_dict() for c in self.containers],
                },
            },
        }
        if self.init_containers:
            spec["template"]["spec"]["initContainers"] = [
                c.to_dict() for c in self.init_containers
            ]

        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
                "labels": pod_labels,
            },
            "spec": spec,
        }


# ── ServiceSpec ───────────────────────────────────────────────────────────────

@dataclass
class ServiceSpec:
    """Kubernetes Service manifest builder.

    Attributes:
        name:         Service name.
        namespace:    Target namespace (default "ml-serving").
        port:         Exposed service port (default 80).
        target_port:  Container port to forward to (default 8080).
        service_type: "ClusterIP", "NodePort", or "LoadBalancer".
        selector:     Pod label selector (default {"app": name}).
        node_port:    NodePort value (30000–32767, 0 = auto-assign).
    """

    name: str
    namespace: str = "ml-serving"
    port: int = 80
    target_port: int = 8080
    service_type: str = "ClusterIP"
    selector: dict[str, str] = field(default_factory=dict)
    node_port: int = 0

    _VALID_TYPES = {"ClusterIP", "NodePort", "LoadBalancer"}

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ServiceSpec.name cannot be empty")
        if self.service_type not in self._VALID_TYPES:
            raise ValueError(f"service_type must be one of {self._VALID_TYPES}")
        if not self.selector:
            self.selector = {"app": self.name}

    def to_manifest(self) -> dict[str, Any]:
        port_spec: dict[str, Any] = {
            "name": "http",
            "port": self.port,
            "targetPort": self.target_port,
            "protocol": "TCP",
        }
        if self.service_type == "NodePort" and self.node_port:
            port_spec["nodePort"] = self.node_port

        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "type": self.service_type,
                "selector": self.selector,
                "ports": [port_spec],
            },
        }


# ── K8sManifestSet ────────────────────────────────────────────────────────────

@dataclass
class K8sManifestSet:
    """Groups a Deployment + Service + optional ConfigMap into one manifest set.

    Attributes:
        deployment: The Deployment manifest builder.
        service:    The Service manifest builder.
        configmap:  Optional ConfigMap data dict (key → value).
        namespace:  Namespace to create if emitting a Namespace manifest.
    """

    deployment: DeploymentSpec
    service: ServiceSpec
    configmap: dict[str, str] = field(default_factory=dict)
    namespace: str = "ml-serving"

    def to_manifest_list(self) -> list[dict[str, Any]]:
        """Return all manifests as a list of dicts (ready for yaml.dump_all)."""
        manifests: list[dict[str, Any]] = [
            # Namespace first
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": self.namespace},
            },
            self.deployment.to_manifest(),
            self.service.to_manifest(),
        ]
        if self.configmap:
            manifests.append({
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {
                    "name": f"{self.deployment.name}-config",
                    "namespace": self.namespace,
                },
                "data": self.configmap,
            })
        return manifests

    def manifest_kinds(self) -> list[str]:
        return [m["kind"] for m in self.to_manifest_list()]
