"""Prometheus Operator + RBAC manifest builders for K8s monitoring.

Day 69 — ServiceMonitor, ClusterRole, and ClusterRoleBinding manifest
builders for the Prometheus Operator. Includes a threat-checkpoint helper
that scans for common K8s secret misconfigurations.

Classes:
  PolicyRule          — one RBAC policy rule (apiGroups, resources, verbs)
  ClusterRoleSpec     — ClusterRole manifest builder
  ServiceMonitorSpec  — Prometheus Operator ServiceMonitor manifest builder
  SecretThreatChecker — validates K8s manifests against secret misconfig rules

See: docs/phase9/day69_k8s_observability.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── PolicyRule ────────────────────────────────────────────────────────────────

@dataclass
class PolicyRule:
    """One RBAC policy rule.

    Attributes:
        api_groups:         API groups ([""] = core).
        resources:          K8s resource types.
        verbs:              Allowed actions.
        non_resource_urls:  Non-resource URL paths (e.g., ["/metrics"]).
    """

    api_groups: list[str] = field(default_factory=lambda: [""])
    resources: list[str] = field(default_factory=list)
    verbs: list[str] = field(default_factory=lambda: ["get", "list", "watch"])
    non_resource_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"verbs": self.verbs}
        if self.non_resource_urls:
            d["nonResourceURLs"] = self.non_resource_urls
        else:
            d["apiGroups"] = self.api_groups
            d["resources"] = self.resources
        return d


# ── ClusterRoleSpec ───────────────────────────────────────────────────────────

@dataclass
class ClusterRoleSpec:
    """ClusterRole manifest builder.

    Attributes:
        name:  ClusterRole name.
        rules: List of PolicyRule objects.
    """

    name: str
    rules: list[PolicyRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ClusterRoleSpec.name cannot be empty")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": self.name},
            "rules": [r.to_dict() for r in self.rules],
        }

    @staticmethod
    def monitoring_reader() -> "ClusterRoleSpec":
        """Return the standard read-only ClusterRole for Prometheus."""
        return ClusterRoleSpec(
            name="monitoring-reader",
            rules=[
                PolicyRule(
                    api_groups=[""],
                    resources=["pods", "services", "endpoints"],
                    verbs=["get", "list", "watch"],
                ),
                PolicyRule(
                    non_resource_urls=["/metrics"],
                    verbs=["get"],
                ),
            ],
        )


# ── ServiceMonitorSpec ────────────────────────────────────────────────────────

@dataclass
class ServiceMonitorSpec:
    """Prometheus Operator ServiceMonitor manifest builder.

    Attributes:
        name:             ServiceMonitor name.
        namespace:        Namespace where ServiceMonitor lives (usually monitoring).
        app_label:        Label selector value for the `app` label.
        metrics_path:     HTTP path to scrape (default "/metrics").
        interval:         Scrape interval (default "30s").
        target_namespace: Namespace where the target Service lives.
        port_name:        Port name on the Service (default "http").
    """

    name: str
    namespace: str = "monitoring"
    app_label: str = ""
    metrics_path: str = "/metrics"
    interval: str = "30s"
    target_namespace: str = "ml-serving"
    port_name: str = "http"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ServiceMonitorSpec.name cannot be empty")
        if not self.app_label:
            self.app_label = self.name

    def to_manifest(self) -> dict[str, Any]:
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "selector": {"matchLabels": {"app": self.app_label}},
                "endpoints": [{
                    "port": self.port_name,
                    "path": self.metrics_path,
                    "interval": self.interval,
                }],
                "namespaceSelector": {"matchNames": [self.target_namespace]},
            },
        }


# ── SecretThreatChecker ───────────────────────────────────────────────────────

@dataclass
class ThreatFinding:
    """One detected secret misconfiguration.

    Attributes:
        rule:     Short identifier for the threat rule.
        severity: "HIGH", "MEDIUM", or "INFO".
        message:  Human-readable description.
    """

    rule: str
    severity: str
    message: str


class SecretThreatChecker:
    """Validates K8s manifest dicts against secret misconfiguration rules.

    Checks run against the threat table in day69_k8s_observability.md.
    """

    def check(self, manifest: dict[str, Any]) -> list[ThreatFinding]:
        """Run all threat checks against a manifest dict.

        Returns:
            List of ThreatFinding objects (empty = no issues found).
        """
        findings: list[ThreatFinding] = []
        kind = manifest.get("kind", "")
        data = manifest.get("data", {})

        # Rule 1: Credentials in ConfigMap
        if kind == "ConfigMap":
            secret_keys = {"password", "secret", "token", "key", "credential", "passwd"}
            for k in data:
                if any(s in k.lower() for s in secret_keys):
                    findings.append(ThreatFinding(
                        rule="cred-in-configmap",
                        severity="HIGH",
                        message=f"Key '{k}' in ConfigMap looks like a credential — use Secret instead",
                    ))

        # Rule 2: Secret exposed as env var (instead of volume mount)
        if kind in {"Deployment", "Pod", "Job"}:
            spec = manifest.get("spec", {})
            template = spec.get("template", spec)  # Job uses spec directly for pod
            pod_spec = template.get("spec", {})
            for container in pod_spec.get("containers", []):
                for env in container.get("env", []):
                    if "secretKeyRef" in env.get("valueFrom", {}):
                        findings.append(ThreatFinding(
                            rule="secret-as-env-var",
                            severity="MEDIUM",
                            message=(
                                f"Secret '{env['valueFrom']['secretKeyRef']['name']}' exposed "
                                f"as env var in container '{container['name']}' — "
                                f"prefer volume mount for secrets"
                            ),
                        ))

        return findings
