"""Tests for infra/ingress.py — IngressRule, IngressSpec."""
from __future__ import annotations

import pytest

from infra.ingress import IngressRule, IngressSpec


class TestIngressRule:
    def test_basic(self) -> None:
        r = IngressRule("/predict", "credit-risk-api", 80)
        d = r.to_dict()
        assert d["path"] == "/predict"
        assert d["backend"]["service"]["name"] == "credit-risk-api"

    def test_path_must_start_with_slash(self) -> None:
        with pytest.raises(ValueError, match="'/'"):
            IngressRule("predict", "api")

    def test_invalid_path_type(self) -> None:
        with pytest.raises(ValueError, match="path_type"):
            IngressRule("/x", "api", path_type="Regex")

    def test_exact_path_type(self) -> None:
        r = IngressRule("/health", "api", path_type="Exact")
        assert r.to_dict()["pathType"] == "Exact"

    def test_service_port_in_dict(self) -> None:
        r = IngressRule("/predict", "api", service_port=8080)
        assert r.to_dict()["backend"]["service"]["port"]["number"] == 8080


class TestIngressSpec:
    def _spec(self) -> IngressSpec:
        spec = IngressSpec("ml-ingress")
        spec.add_rule(IngressRule("/predict", "credit-risk-api"))
        spec.add_rule(IngressRule("/health", "credit-risk-api"))
        return spec

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            IngressSpec("")

    def test_manifest_kind(self) -> None:
        m = self._spec().to_manifest()
        assert m["kind"] == "Ingress"
        assert m["apiVersion"] == "networking.k8s.io/v1"

    def test_ingress_class(self) -> None:
        m = self._spec().to_manifest()
        assert m["spec"]["ingressClassName"] == "nginx"

    def test_rules_grouped_by_host(self) -> None:
        spec = IngressSpec("ing")
        spec.add_rule(IngressRule("/a", "svc-a", host="api.local"))
        spec.add_rule(IngressRule("/b", "svc-b", host="api.local"))
        spec.add_rule(IngressRule("/c", "svc-c", host="metrics.local"))
        m = spec.to_manifest()
        # Two hosts → two rule entries
        assert len(m["spec"]["rules"]) == 2
        api_rule = next(r for r in m["spec"]["rules"] if r["host"] == "api.local")
        assert len(api_rule["http"]["paths"]) == 2

    def test_annotations_included(self) -> None:
        spec = IngressSpec("ing", annotations={"nginx.ingress.kubernetes.io/rewrite-target": "/"})
        m = spec.to_manifest()
        assert "annotations" in m["metadata"]

    def test_no_annotations_no_key(self) -> None:
        m = self._spec().to_manifest()
        assert "annotations" not in m["metadata"]

    def test_add_rule(self) -> None:
        spec = IngressSpec("ing")
        assert len(spec.rules) == 0
        spec.add_rule(IngressRule("/x", "svc"))
        assert len(spec.rules) == 1

    def test_namespace(self) -> None:
        spec = IngressSpec("ing", namespace="custom")
        m = spec.to_manifest()
        assert m["metadata"]["namespace"] == "custom"
