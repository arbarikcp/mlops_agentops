"""Tests for infra/k8s_manifests.py — ResourceRequirements, ContainerSpec, DeploymentSpec, ServiceSpec, K8sManifestSet."""
from __future__ import annotations

import pytest

from infra.k8s_manifests import (
    ContainerSpec,
    DeploymentSpec,
    K8sManifestSet,
    ProbeSpec,
    ResourceRequirements,
    ServiceSpec,
)


# ── ResourceRequirements ────────────────────────────────────────────────────────

class TestResourceRequirements:
    def test_defaults_to_dict(self) -> None:
        d = ResourceRequirements().to_dict()
        assert d["requests"]["cpu"] == "500m"
        assert d["limits"]["memory"] == "2Gi"

    def test_no_gpu_by_default(self) -> None:
        d = ResourceRequirements().to_dict()
        assert "nvidia.com/gpu" not in d["requests"]

    def test_gpu_appears_in_both_requests_and_limits(self) -> None:
        d = ResourceRequirements(gpu_count=1).to_dict()
        assert d["requests"]["nvidia.com/gpu"] == "1"
        assert d["limits"]["nvidia.com/gpu"] == "1"

    def test_negative_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="gpu_count"):
            ResourceRequirements(gpu_count=-1)

    def test_custom_values(self) -> None:
        r = ResourceRequirements(cpu_request="250m", memory_limit="4Gi")
        d = r.to_dict()
        assert d["requests"]["cpu"] == "250m"
        assert d["limits"]["memory"] == "4Gi"


# ── ProbeSpec ───────────────────────────────────────────────────────────────────

class TestProbeSpec:
    def test_to_dict(self) -> None:
        p = ProbeSpec(path="/health", port=8080, initial_delay_s=15)
        d = p.to_dict()
        assert d["httpGet"]["path"] == "/health"
        assert d["httpGet"]["port"] == 8080
        assert d["initialDelaySeconds"] == 15

    def test_defaults(self) -> None:
        p = ProbeSpec()
        d = p.to_dict()
        assert d["failureThreshold"] == 3


# ── ContainerSpec ───────────────────────────────────────────────────────────────

class TestContainerSpec:
    def test_basic(self) -> None:
        c = ContainerSpec("api", "credit-risk:v1", port=8080)
        d = c.to_dict()
        assert d["name"] == "api"
        assert d["image"] == "credit-risk:v1"
        assert d["ports"][0]["containerPort"] == 8080

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ContainerSpec("", "img:v1")

    def test_empty_image_raises(self) -> None:
        with pytest.raises(ValueError, match="image"):
            ContainerSpec("api", "")

    def test_env_vars(self) -> None:
        c = ContainerSpec("api", "img:v1", env={"FOO": "bar"})
        d = c.to_dict()
        env_map = {e["name"]: e["value"] for e in d["env"]}
        assert env_map["FOO"] == "bar"

    def test_command_included(self) -> None:
        c = ContainerSpec("dl", "aws-cli", command=["aws", "s3", "cp"])
        d = c.to_dict()
        assert d["command"] == ["aws", "s3", "cp"]

    def test_no_port_no_ports_key(self) -> None:
        c = ContainerSpec("api", "img:v1", port=0)
        d = c.to_dict()
        assert "ports" not in d

    def test_resources_present(self) -> None:
        c = ContainerSpec("api", "img:v1")
        d = c.to_dict()
        assert "resources" in d
        assert "requests" in d["resources"]


# ── DeploymentSpec ──────────────────────────────────────────────────────────────

class TestDeploymentSpec:
    def _dep(self, **kw) -> DeploymentSpec:
        containers = kw.pop("containers", [ContainerSpec("api", "img:v1", port=8080)])
        return DeploymentSpec(name="test-api", containers=containers, **kw)

    def test_basic_manifest(self) -> None:
        m = self._dep().to_manifest()
        assert m["kind"] == "Deployment"
        assert m["metadata"]["name"] == "test-api"
        assert m["spec"]["replicas"] == 3

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            DeploymentSpec(name="")

    def test_replicas_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="replicas"):
            DeploymentSpec(name="api", replicas=0)

    def test_rolling_update_in_spec(self) -> None:
        m = self._dep().to_manifest()
        assert m["spec"]["strategy"]["type"] == "RollingUpdate"

    def test_init_containers_present(self) -> None:
        init = ContainerSpec("model-dl", "aws-cli", is_init=True, command=["aws"])
        m = self._dep(init_containers=[init]).to_manifest()
        assert "initContainers" in m["spec"]["template"]["spec"]

    def test_no_init_containers_no_key(self) -> None:
        m = self._dep().to_manifest()
        assert "initContainers" not in m["spec"]["template"]["spec"]

    def test_labels_in_pod_template(self) -> None:
        m = self._dep(labels={"version": "v1"}).to_manifest()
        pod_labels = m["spec"]["template"]["metadata"]["labels"]
        assert pod_labels["version"] == "v1"
        assert pod_labels["app"] == "test-api"

    def test_namespace(self) -> None:
        m = self._dep(namespace="custom-ns").to_manifest()
        assert m["metadata"]["namespace"] == "custom-ns"


# ── ServiceSpec ─────────────────────────────────────────────────────────────────

class TestServiceSpec:
    def test_cluster_ip(self) -> None:
        s = ServiceSpec("svc")
        m = s.to_manifest()
        assert m["spec"]["type"] == "ClusterIP"
        assert m["spec"]["ports"][0]["port"] == 80
        assert m["spec"]["ports"][0]["targetPort"] == 8080

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            ServiceSpec("")

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError, match="service_type"):
            ServiceSpec("svc", service_type="Ingress")

    def test_nodeport_includes_node_port(self) -> None:
        s = ServiceSpec("svc", service_type="NodePort", node_port=30080)
        m = s.to_manifest()
        assert m["spec"]["ports"][0]["nodePort"] == 30080

    def test_default_selector(self) -> None:
        s = ServiceSpec("my-svc")
        assert s.selector == {"app": "my-svc"}

    def test_custom_selector(self) -> None:
        s = ServiceSpec("svc", selector={"app": "api", "tier": "backend"})
        m = s.to_manifest()
        assert m["spec"]["selector"]["tier"] == "backend"

    def test_load_balancer_type(self) -> None:
        s = ServiceSpec("svc", service_type="LoadBalancer")
        m = s.to_manifest()
        assert m["spec"]["type"] == "LoadBalancer"


# ── K8sManifestSet ──────────────────────────────────────────────────────────────

class TestK8sManifestSet:
    def _set(self, with_configmap: bool = False) -> K8sManifestSet:
        dep = DeploymentSpec("api", containers=[ContainerSpec("api", "img:v1", port=8080)])
        svc = ServiceSpec("api")
        cfg = {"FOO": "bar"} if with_configmap else {}
        return K8sManifestSet(deployment=dep, service=svc, configmap=cfg)

    def test_manifest_list_without_configmap(self) -> None:
        manifests = self._set().to_manifest_list()
        kinds = [m["kind"] for m in manifests]
        assert "Namespace" in kinds
        assert "Deployment" in kinds
        assert "Service" in kinds
        assert "ConfigMap" not in kinds

    def test_manifest_list_with_configmap(self) -> None:
        manifests = self._set(with_configmap=True).to_manifest_list()
        kinds = [m["kind"] for m in manifests]
        assert "ConfigMap" in kinds

    def test_namespace_is_first(self) -> None:
        manifests = self._set().to_manifest_list()
        assert manifests[0]["kind"] == "Namespace"

    def test_manifest_kinds(self) -> None:
        kinds = self._set().manifest_kinds()
        assert "Deployment" in kinds
        assert "Service" in kinds

    def test_configmap_data(self) -> None:
        manifests = self._set(with_configmap=True).to_manifest_list()
        cm = next(m for m in manifests if m["kind"] == "ConfigMap")
        assert cm["data"]["FOO"] == "bar"
