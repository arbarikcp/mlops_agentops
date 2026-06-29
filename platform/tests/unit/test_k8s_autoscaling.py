"""Tests for infra/k8s_autoscaling.py — HPAMetric, HPASpec, KEDAScaledObject, KueueJobConfig."""
from __future__ import annotations

import pytest

from infra.k8s_autoscaling import HPAMetric, HPASpec, KEDAScaledObject, KueueJobConfig


# ── HPAMetric ──────────────────────────────────────────────────────────────────

class TestHPAMetric:
    def test_cpu_metric(self) -> None:
        m = HPAMetric("cpu", 70)
        d = m.to_dict()
        assert d["resource"]["name"] == "cpu"
        assert d["resource"]["target"]["averageUtilization"] == 70

    def test_memory_metric(self) -> None:
        m = HPAMetric("memory", 80)
        d = m.to_dict()
        assert d["resource"]["name"] == "memory"

    def test_invalid_resource_raises(self) -> None:
        with pytest.raises(ValueError, match="resource"):
            HPAMetric("gpu", 70)

    def test_invalid_utilization_raises(self) -> None:
        with pytest.raises(ValueError, match="target_utilization"):
            HPAMetric("cpu", 110)

    def test_min_utilization(self) -> None:
        m = HPAMetric("cpu", 1)
        assert m.to_dict()["resource"]["target"]["averageUtilization"] == 1


# ── HPASpec ────────────────────────────────────────────────────────────────────

class TestHPASpec:
    def _hpa(self, **kw) -> HPASpec:
        defaults = dict(name="credit-risk-hpa", metrics=[HPAMetric("cpu", 70)])
        return HPASpec(**{**defaults, **kw})

    def test_basic_manifest(self) -> None:
        m = self._hpa().to_manifest()
        assert m["kind"] == "HorizontalPodAutoscaler"
        assert m["apiVersion"] == "autoscaling/v2"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            HPASpec(name="")

    def test_min_replicas_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="min_replicas"):
            HPASpec(name="x", min_replicas=0)

    def test_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_replicas"):
            HPASpec(name="x", min_replicas=10, max_replicas=5)

    def test_deployment_name_defaults_to_name(self) -> None:
        hpa = self._hpa()
        assert hpa.deployment_name == "credit-risk-hpa"

    def test_metrics_in_manifest(self) -> None:
        m = self._hpa(metrics=[HPAMetric("cpu", 70), HPAMetric("memory", 80)]).to_manifest()
        assert len(m["spec"]["metrics"]) == 2

    def test_replicas_in_manifest(self) -> None:
        m = self._hpa(min_replicas=2, max_replicas=15).to_manifest()
        assert m["spec"]["minReplicas"] == 2
        assert m["spec"]["maxReplicas"] == 15


# ── KEDAScaledObject ───────────────────────────────────────────────────────────

class TestKEDAScaledObject:
    def _keda(self, **kw) -> KEDAScaledObject:
        defaults = dict(
            name="batch-scaler",
            trigger_metadata={"queueURL": "https://sqs.aws.com/q", "queueLength": "100"},
        )
        return KEDAScaledObject(**{**defaults, **kw})

    def test_basic_manifest(self) -> None:
        m = self._keda().to_manifest()
        assert m["kind"] == "ScaledObject"
        assert m["apiVersion"] == "keda.sh/v1alpha1"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            KEDAScaledObject(name="")

    def test_invalid_trigger_type_raises(self) -> None:
        with pytest.raises(ValueError, match="trigger_type"):
            KEDAScaledObject(name="x", trigger_type="custom-unsupported")

    def test_min_replicas_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="min_replicas"):
            KEDAScaledObject(name="x", min_replicas=-1)

    def test_scale_to_zero(self) -> None:
        m = self._keda(min_replicas=0).to_manifest()
        assert m["spec"]["minReplicaCount"] == 0

    def test_trigger_metadata_included(self) -> None:
        m = self._keda().to_manifest()
        assert m["spec"]["triggers"][0]["metadata"]["queueLength"] == "100"

    def test_prometheus_trigger(self) -> None:
        m = self._keda(
            trigger_type="prometheus",
            trigger_metadata={"serverAddress": "http://prometheus:9090"},
        ).to_manifest()
        assert m["spec"]["triggers"][0]["type"] == "prometheus"

    def test_deployment_name_defaults(self) -> None:
        k = self._keda()
        assert k.deployment_name == "batch-scaler"


# ── KueueJobConfig ─────────────────────────────────────────────────────────────

class TestKueueJobConfig:
    def _job(self, **kw) -> KueueJobConfig:
        defaults = dict(job_name="credit-risk-train", image="credit-risk-trainer:v1")
        return KueueJobConfig(**{**defaults, **kw})

    def test_basic_manifest(self) -> None:
        m = self._job().to_manifest()
        assert m["kind"] == "Job"
        assert m["apiVersion"] == "batch/v1"

    def test_kueue_label_present(self) -> None:
        m = self._job(queue_name="team-a-queue").to_manifest()
        assert m["metadata"]["labels"]["kueue.x-k8s.io/queue-name"] == "team-a-queue"

    def test_empty_job_name_raises(self) -> None:
        with pytest.raises(ValueError, match="job_name"):
            KueueJobConfig(job_name="")

    def test_negative_gpu_raises(self) -> None:
        with pytest.raises(ValueError, match="gpu_count"):
            KueueJobConfig(job_name="x", gpu_count=-1)

    def test_gpu_in_resources(self) -> None:
        m = self._job(gpu_count=2).to_manifest()
        container = m["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"

    def test_no_gpu_no_gpu_resource(self) -> None:
        m = self._job(gpu_count=0).to_manifest()
        container = m["spec"]["template"]["spec"]["containers"][0]
        assert "nvidia.com/gpu" not in container["resources"]["requests"]

    def test_command_included(self) -> None:
        m = self._job(command=["python", "train.py"]).to_manifest()
        assert m["spec"]["template"]["spec"]["containers"][0]["command"] == ["python", "train.py"]

    def test_restart_policy(self) -> None:
        m = self._job().to_manifest()
        assert m["spec"]["template"]["spec"]["restartPolicy"] == "OnFailure"

    def test_namespace(self) -> None:
        m = self._job(namespace="team-a").to_manifest()
        assert m["metadata"]["namespace"] == "team-a"
