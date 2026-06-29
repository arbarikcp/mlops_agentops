"""Helm chart configuration builder.

Day 61 — generates Helm chart metadata and values as Python dicts so they can
be validated without requiring the helm CLI. Used for CI checks that verify
chart configuration before helm install.

Classes:
  HelmValues  — typed values.yaml representation for the credit-risk chart
  HelmChart   — chart metadata + values; renders install/upgrade commands

See: docs/phase9/day61_helm_chart.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HelmValues:
    """Typed representation of the credit-risk chart values.yaml.

    Attributes:
        replica_count:         Desired replica count.
        image_repo:            Container image repository.
        image_tag:             Container image tag (commit SHA in CI).
        pull_policy:           ImagePullPolicy string.
        service_type:          K8s service type.
        service_port:          Service port.
        cpu_request:           CPU request string.
        memory_request:        Memory request string.
        cpu_limit:             CPU limit string.
        memory_limit:          Memory limit string.
        mlflow_uri:            MLflow tracking server URI.
        model_version:         Model registry version string.
        model_s3_path:         S3 path for model artifact.
        autoscaling_enabled:   Whether HPA is enabled.
        min_replicas:          HPA minimum replicas.
        max_replicas:          HPA maximum replicas.
        target_cpu_pct:        HPA target CPU utilization.
    """

    replica_count: int = 3
    image_repo: str = "credit-risk-api"
    image_tag: str = "v1"
    pull_policy: str = "IfNotPresent"
    service_type: str = "ClusterIP"
    service_port: int = 80
    cpu_request: str = "500m"
    memory_request: str = "512Mi"
    cpu_limit: str = "2"
    memory_limit: str = "2Gi"
    mlflow_uri: str = "http://mlflow:5000"
    model_version: str = "credit-risk-v1.2"
    model_s3_path: str = "s3://ml-models/credit-risk/v1.2/model.pkl"
    autoscaling_enabled: bool = False
    min_replicas: int = 2
    max_replicas: int = 10
    target_cpu_pct: int = 70

    def __post_init__(self) -> None:
        if self.replica_count < 1:
            raise ValueError("replica_count must be >= 1")
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas cannot exceed max_replicas")

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicaCount": self.replica_count,
            "image": {
                "repository": self.image_repo,
                "tag": self.image_tag,
                "pullPolicy": self.pull_policy,
            },
            "service": {
                "type": self.service_type,
                "port": self.service_port,
            },
            "resources": {
                "requests": {"cpu": self.cpu_request, "memory": self.memory_request},
                "limits": {"cpu": self.cpu_limit, "memory": self.memory_limit},
            },
            "config": {
                "mlflowUri": self.mlflow_uri,
                "modelVersion": self.model_version,
                "modelS3Path": self.model_s3_path,
            },
            "autoscaling": {
                "enabled": self.autoscaling_enabled,
                "minReplicas": self.min_replicas,
                "maxReplicas": self.max_replicas,
                "targetCPUUtilizationPercentage": self.target_cpu_pct,
            },
        }


@dataclass
class HelmChart:
    """Helm chart metadata and values builder.

    Attributes:
        name:          Chart name (also used as release name).
        chart_version: SemVer chart version string.
        app_version:   Application version string (model version).
        description:   Chart description.
        values:        HelmValues instance.
    """

    name: str
    chart_version: str = "0.1.0"
    app_version: str = "1.2.0"
    description: str = ""
    values: HelmValues = field(default_factory=HelmValues)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("HelmChart.name cannot be empty")

    def to_chart_yaml(self) -> dict[str, Any]:
        """Return Chart.yaml as a Python dict."""
        return {
            "apiVersion": "v2",
            "name": self.name,
            "description": self.description or f"{self.name} ML serving chart",
            "type": "application",
            "version": self.chart_version,
            "appVersion": self.app_version,
        }

    def to_values_dict(self) -> dict[str, Any]:
        """Return values.yaml as a Python dict."""
        return self.values.to_dict()

    def render_install_cmd(
        self,
        namespace: str = "ml-serving",
        chart_path: str = "./infra/helm/credit-risk",
        extra_sets: dict[str, str] | None = None,
        upgrade: bool = True,
    ) -> str:
        """Return the helm install/upgrade CLI command string.

        Args:
            namespace:   Target K8s namespace.
            chart_path:  Path to the chart directory.
            extra_sets:  Additional --set key=value overrides.
            upgrade:     If True, uses 'helm upgrade --install'; else 'helm install'.

        Returns:
            Shell command string ready for CI scripts.
        """
        verb = "upgrade --install" if upgrade else "install"
        parts = [
            f"helm {verb} {self.name} {chart_path}",
            f"--namespace {namespace}",
            f"--values {chart_path}/values.yaml",
        ]
        if extra_sets:
            for k, v in extra_sets.items():
                parts.append(f"--set {k}={v}")
        return " \\\n  ".join(parts)
