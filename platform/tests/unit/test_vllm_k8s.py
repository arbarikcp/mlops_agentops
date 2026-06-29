"""Unit tests for platform/llm/vllm_k8s.py (Day 99)."""

import pytest
from llm.vllm_config import VLLMEngineConfig
from llm.vllm_k8s import (
    CapacityPlan,
    PodMonitorSpec,
    VLLMDeploymentSpec,
    VLLMHPASpec,
    VLLMServiceSpec,
)


def _engine(tp=1, pp=1):
    return VLLMEngineConfig(
        model="meta-llama/Llama-2-7b",
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
    )


# ── VLLMDeploymentSpec ─────────────────────────────────────────────────────

class TestVLLMDeploymentSpec:
    def test_to_manifest_kind(self):
        spec = VLLMDeploymentSpec(name="vllm", image="vllm/vllm-openai:latest", engine_config=_engine())
        m = spec.to_manifest()
        assert m["kind"] == "Deployment"
        assert m["apiVersion"] == "apps/v1"

    def test_to_manifest_gpu_resource(self):
        spec = VLLMDeploymentSpec(name="vllm", image="vllm:v0.4", engine_config=_engine(tp=2, pp=2))
        m = spec.to_manifest()
        container = m["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"

    def test_to_manifest_has_probes(self):
        spec = VLLMDeploymentSpec(name="vllm", image="vllm:latest", engine_config=_engine())
        m = spec.to_manifest()
        container = m["spec"]["template"]["spec"]["containers"][0]
        assert "livenessProbe" in container
        assert "readinessProbe" in container
        assert container["livenessProbe"]["httpGet"]["path"] == "/health"

    def test_to_manifest_replicas(self):
        spec = VLLMDeploymentSpec(
            name="vllm", image="vllm:latest", engine_config=_engine(), replicas=3
        )
        m = spec.to_manifest()
        assert m["spec"]["replicas"] == 3

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            VLLMDeploymentSpec(name="", image="vllm:latest", engine_config=_engine())

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="image"):
            VLLMDeploymentSpec(name="vllm", image="", engine_config=_engine())

    def test_invalid_replicas(self):
        with pytest.raises(ValueError, match="replicas"):
            VLLMDeploymentSpec(name="vllm", image="vllm:latest", engine_config=_engine(), replicas=0)

    def test_to_dict(self):
        spec = VLLMDeploymentSpec(name="vllm", image="vllm:latest", engine_config=_engine())
        d = spec.to_dict()
        assert d["name"] == "vllm"
        assert "engine_config" in d


# ── VLLMServiceSpec ────────────────────────────────────────────────────────

class TestVLLMServiceSpec:
    def test_to_manifest_kind(self):
        svc = VLLMServiceSpec(name="vllm-svc", deployment_name="vllm")
        m = svc.to_manifest()
        assert m["kind"] == "Service"
        assert m["spec"]["type"] == "ClusterIP"

    def test_loadbalancer(self):
        svc = VLLMServiceSpec(
            name="vllm-svc", deployment_name="vllm", service_type="LoadBalancer"
        )
        m = svc.to_manifest()
        assert m["spec"]["type"] == "LoadBalancer"

    def test_invalid_service_type(self):
        with pytest.raises(ValueError, match="service_type"):
            VLLMServiceSpec(name="svc", deployment_name="dep", service_type="ExternalName")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            VLLMServiceSpec(name="", deployment_name="dep")

    def test_empty_deployment_raises(self):
        with pytest.raises(ValueError, match="deployment_name"):
            VLLMServiceSpec(name="svc", deployment_name="")

    def test_to_manifest_port(self):
        svc = VLLMServiceSpec(name="svc", deployment_name="dep", port=9000)
        m = svc.to_manifest()
        assert m["spec"]["ports"][0]["port"] == 9000


# ── VLLMHPASpec ────────────────────────────────────────────────────────────

class TestVLLMHPASpec:
    def test_to_manifest_kind(self):
        hpa = VLLMHPASpec(name="vllm-hpa", deployment_name="vllm")
        m = hpa.to_manifest()
        assert m["kind"] == "HorizontalPodAutoscaler"
        assert m["apiVersion"] == "autoscaling/v2"

    def test_to_manifest_custom_metric(self):
        hpa = VLLMHPASpec(name="vllm-hpa", deployment_name="vllm", target_rps=20.0)
        m = hpa.to_manifest()
        metric = m["spec"]["metrics"][0]
        assert metric["pods"]["metric"]["name"] == "vllm_request_rate"

    def test_min_max_replicas(self):
        hpa = VLLMHPASpec(
            name="hpa", deployment_name="dep", min_replicas=2, max_replicas=8
        )
        m = hpa.to_manifest()
        assert m["spec"]["minReplicas"] == 2
        assert m["spec"]["maxReplicas"] == 8

    def test_invalid_min_gt_max(self):
        with pytest.raises(ValueError, match="min_replicas"):
            VLLMHPASpec(name="hpa", deployment_name="dep", min_replicas=5, max_replicas=3)

    def test_invalid_target_rps(self):
        with pytest.raises(ValueError, match="target_rps"):
            VLLMHPASpec(name="hpa", deployment_name="dep", target_rps=0.0)

    def test_to_dict(self):
        hpa = VLLMHPASpec(name="hpa", deployment_name="dep", target_rps=15.0)
        d = hpa.to_dict()
        assert d["target_rps"] == 15.0


# ── CapacityPlan ───────────────────────────────────────────────────────────

class TestCapacityPlan:
    def test_replicas_needed_basic(self):
        # 10 rps, 5 rps/replica, 1.2 safety → ceil(10*1.2/5) = ceil(2.4) = 3
        plan = CapacityPlan(target_rps=10.0, single_replica_throughput=5.0)
        assert plan.replicas_needed() == 3

    def test_replicas_exact(self):
        # 10 rps, 10 rps/replica, 1.0 safety → ceil(1.0) = 1
        plan = CapacityPlan(
            target_rps=10.0, single_replica_throughput=10.0, safety_factor=1.0
        )
        assert plan.replicas_needed() == 1

    def test_hourly_cost(self):
        plan = CapacityPlan(
            target_rps=10.0,
            single_replica_throughput=5.0,
            gpu_cost_per_hour=3.0,
        )
        # 3 replicas * $3 = $9
        assert plan.hourly_cost_usd() == pytest.approx(9.0)

    def test_invalid_target_rps(self):
        with pytest.raises(ValueError, match="target_rps"):
            CapacityPlan(target_rps=0.0, single_replica_throughput=5.0)

    def test_invalid_throughput(self):
        with pytest.raises(ValueError, match="single_replica_throughput"):
            CapacityPlan(target_rps=10.0, single_replica_throughput=0.0)

    def test_invalid_safety_factor_lt_one(self):
        with pytest.raises(ValueError, match="safety_factor"):
            CapacityPlan(
                target_rps=10.0, single_replica_throughput=5.0, safety_factor=0.9
            )

    def test_to_dict(self):
        plan = CapacityPlan(target_rps=100.0, single_replica_throughput=20.0)
        d = plan.to_dict()
        assert "replicas_needed" in d
        assert "hourly_cost_usd" in d


# ── PodMonitorSpec ─────────────────────────────────────────────────────────

class TestPodMonitorSpec:
    def test_to_manifest_kind(self):
        pm = PodMonitorSpec(name="vllm-monitor")
        m = pm.to_manifest()
        assert m["kind"] == "PodMonitor"
        assert m["apiVersion"] == "monitoring.coreos.com/v1"

    def test_to_manifest_scrape_config(self):
        pm = PodMonitorSpec(name="vllm-monitor", scrape_interval="30s")
        m = pm.to_manifest()
        endpoint = m["spec"]["podMetricsEndpoints"][0]
        assert endpoint["interval"] == "30s"
        assert endpoint["path"] == "/metrics"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            PodMonitorSpec(name="")

    def test_to_dict(self):
        pm = PodMonitorSpec(name="monitor", namespace="prod")
        d = pm.to_dict()
        assert d["namespace"] == "prod"

    def test_namespace_selector(self):
        pm = PodMonitorSpec(name="pm", namespace="ml-prod")
        m = pm.to_manifest()
        assert "ml-prod" in m["spec"]["namespaceSelector"]["matchNames"]
