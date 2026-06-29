"""Kubernetes Ingress manifest builder.

Day 60 — generates Ingress manifests for the kind cluster (NGINX controller).
Companions the k8s_manifests.py deployment/service builder from Day 59.

Classes:
  IngressRule  — one path rule mapping host+path to a backend service
  IngressSpec  — full Ingress manifest builder

See: docs/phase9/day60_kind_cluster.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IngressRule:
    """One HTTP path rule inside an Ingress.

    Attributes:
        path:         URL path prefix (e.g., "/predict").
        service_name: Backend service name.
        service_port: Backend service port number.
        path_type:    "Prefix" (default) or "Exact".
        host:         Virtual host (default "localhost").
    """

    path: str
    service_name: str
    service_port: int = 80
    path_type: str = "Prefix"
    host: str = "localhost"

    def __post_init__(self) -> None:
        if not self.path.startswith("/"):
            raise ValueError(f"IngressRule.path must start with '/'; got {self.path!r}")
        if self.path_type not in {"Prefix", "Exact"}:
            raise ValueError(f"path_type must be 'Prefix' or 'Exact'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "pathType": self.path_type,
            "backend": {
                "service": {
                    "name": self.service_name,
                    "port": {"number": self.service_port},
                }
            },
        }


@dataclass
class IngressSpec:
    """Kubernetes Ingress manifest builder.

    Attributes:
        name:          Ingress resource name.
        namespace:     Target namespace (default "ml-serving").
        rules:         List of IngressRule objects.
        ingress_class: Ingress class name (default "nginx").
        annotations:   Metadata annotations dict.
    """

    name: str
    namespace: str = "ml-serving"
    rules: list[IngressRule] = field(default_factory=list)
    ingress_class: str = "nginx"
    annotations: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("IngressSpec.name cannot be empty")

    def add_rule(self, rule: IngressRule) -> None:
        self.rules.append(rule)

    def _group_by_host(self) -> dict[str, list[IngressRule]]:
        groups: dict[str, list[IngressRule]] = {}
        for r in self.rules:
            groups.setdefault(r.host, []).append(r)
        return groups

    def to_manifest(self) -> dict[str, Any]:
        http_rules = []
        for host, host_rules in self._group_by_host().items():
            http_rules.append({
                "host": host,
                "http": {"paths": [r.to_dict() for r in host_rules]},
            })

        manifest: dict[str, Any] = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "spec": {
                "ingressClassName": self.ingress_class,
                "rules": http_rules,
            },
        }
        if self.annotations:
            manifest["metadata"]["annotations"] = self.annotations
        return manifest
