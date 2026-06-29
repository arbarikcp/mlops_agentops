"""Tests for infra/k8s_observability.py — PolicyRule, ClusterRoleSpec, ServiceMonitorSpec, SecretThreatChecker."""
from __future__ import annotations

import pytest

from infra.k8s_observability import (
    ClusterRoleSpec,
    PolicyRule,
    SecretThreatChecker,
    ServiceMonitorSpec,
    ThreatFinding,
)


# ── PolicyRule ──────────────────────────────────────────────────────────────────

class TestPolicyRule:
    def test_basic_resource_rule(self) -> None:
        r = PolicyRule(api_groups=[""], resources=["pods"], verbs=["get"])
        d = r.to_dict()
        assert d["apiGroups"] == [""]
        assert "pods" in d["resources"]

    def test_non_resource_url(self) -> None:
        r = PolicyRule(non_resource_urls=["/metrics"], verbs=["get"])
        d = r.to_dict()
        assert d["nonResourceURLs"] == ["/metrics"]
        assert "apiGroups" not in d

    def test_verbs_in_dict(self) -> None:
        r = PolicyRule(verbs=["get", "list"])
        assert r.to_dict()["verbs"] == ["get", "list"]


# ── ClusterRoleSpec ─────────────────────────────────────────────────────────────

class TestClusterRoleSpec:
    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ClusterRoleSpec(name="")

    def test_to_manifest(self) -> None:
        role = ClusterRoleSpec(name="test-role", rules=[
            PolicyRule(resources=["pods"], verbs=["get"]),
        ])
        m = role.to_manifest()
        assert m["kind"] == "ClusterRole"
        assert m["metadata"]["name"] == "test-role"
        assert len(m["rules"]) == 1

    def test_monitoring_reader_factory(self) -> None:
        role = ClusterRoleSpec.monitoring_reader()
        assert role.name == "monitoring-reader"
        assert len(role.rules) == 2

    def test_monitoring_reader_has_metrics_url(self) -> None:
        role = ClusterRoleSpec.monitoring_reader()
        m = role.to_manifest()
        has_metrics = any(
            "/metrics" in r.get("nonResourceURLs", [])
            for r in m["rules"]
        )
        assert has_metrics

    def test_monitoring_reader_has_pod_resource(self) -> None:
        role = ClusterRoleSpec.monitoring_reader()
        m = role.to_manifest()
        has_pods = any("pods" in r.get("resources", []) for r in m["rules"])
        assert has_pods


# ── ServiceMonitorSpec ──────────────────────────────────────────────────────────

class TestServiceMonitorSpec:
    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ServiceMonitorSpec(name="")

    def test_to_manifest(self) -> None:
        sm = ServiceMonitorSpec("credit-risk-api")
        m = sm.to_manifest()
        assert m["kind"] == "ServiceMonitor"
        assert m["apiVersion"] == "monitoring.coreos.com/v1"

    def test_default_app_label(self) -> None:
        sm = ServiceMonitorSpec("credit-risk-api")
        assert sm.app_label == "credit-risk-api"

    def test_selector_in_manifest(self) -> None:
        sm = ServiceMonitorSpec("api", app_label="my-api")
        m = sm.to_manifest()
        assert m["spec"]["selector"]["matchLabels"]["app"] == "my-api"

    def test_endpoint_path(self) -> None:
        sm = ServiceMonitorSpec("api", metrics_path="/custom-metrics")
        m = sm.to_manifest()
        assert m["spec"]["endpoints"][0]["path"] == "/custom-metrics"

    def test_interval(self) -> None:
        sm = ServiceMonitorSpec("api", interval="15s")
        m = sm.to_manifest()
        assert m["spec"]["endpoints"][0]["interval"] == "15s"

    def test_target_namespace(self) -> None:
        sm = ServiceMonitorSpec("api", target_namespace="custom-ns")
        m = sm.to_manifest()
        assert "custom-ns" in m["spec"]["namespaceSelector"]["matchNames"]


# ── SecretThreatChecker ─────────────────────────────────────────────────────────

class TestSecretThreatChecker:
    def test_clean_configmap_no_findings(self) -> None:
        manifest = {
            "kind": "ConfigMap",
            "data": {"MLFLOW_URI": "http://mlflow:5000", "LOG_LEVEL": "INFO"},
        }
        findings = SecretThreatChecker().check(manifest)
        assert len(findings) == 0

    def test_credential_in_configmap_detected(self) -> None:
        manifest = {
            "kind": "ConfigMap",
            "data": {"POSTGRES_PASSWORD": "changeme"},
        }
        findings = SecretThreatChecker().check(manifest)
        assert len(findings) == 1
        assert findings[0].rule == "cred-in-configmap"
        assert findings[0].severity == "HIGH"

    def test_multiple_cred_keys_in_configmap(self) -> None:
        manifest = {
            "kind": "ConfigMap",
            "data": {"DB_PASSWORD": "x", "API_TOKEN": "y", "MLFLOW_URI": "http://mlflow"},
        }
        findings = SecretThreatChecker().check(manifest)
        assert len(findings) == 2

    def test_secret_as_env_var_detected(self) -> None:
        manifest = {
            "kind": "Deployment",
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "api",
                            "env": [{
                                "name": "DB_PASS",
                                "valueFrom": {"secretKeyRef": {"name": "db-secret", "key": "password"}},
                            }],
                        }]
                    }
                }
            },
        }
        findings = SecretThreatChecker().check(manifest)
        assert len(findings) == 1
        assert findings[0].rule == "secret-as-env-var"
        assert findings[0].severity == "MEDIUM"

    def test_no_env_vars_no_findings(self) -> None:
        manifest = {
            "kind": "Deployment",
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{"name": "api", "image": "img:v1"}]
                    }
                }
            },
        }
        findings = SecretThreatChecker().check(manifest)
        assert len(findings) == 0

    def test_finding_type(self) -> None:
        manifest = {"kind": "ConfigMap", "data": {"SECRET_KEY": "x"}}
        findings = SecretThreatChecker().check(manifest)
        assert isinstance(findings[0], ThreatFinding)
