"""Unit tests for platform/llm/llm_serving.py (Day 101)."""

import pytest
from llm.llm_serving import (
    LLMInferenceServiceSpec,
    RayServeDeploymentSpec,
    RayServeGraph,
    RuntimeType,
    ServingBackendAdvisor,
)


class TestLLMInferenceServiceSpec:
    def test_basic_creation(self):
        spec = LLMInferenceServiceSpec(name="llama-svc", model_uri="s3://models/llama")
        assert spec.runtime == RuntimeType.VLLM
        assert spec.min_replicas == 1

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            LLMInferenceServiceSpec(name="", model_uri="s3://x")

    def test_empty_model_uri_raises(self):
        with pytest.raises(ValueError, match="model_uri"):
            LLMInferenceServiceSpec(name="x", model_uri="")

    def test_negative_min_replicas_raises(self):
        with pytest.raises(ValueError, match="min_replicas"):
            LLMInferenceServiceSpec(name="x", model_uri="y", min_replicas=-1)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_replicas"):
            LLMInferenceServiceSpec(name="x", model_uri="y", min_replicas=5, max_replicas=2)

    def test_to_manifest_shape(self):
        spec = LLMInferenceServiceSpec(name="llama-svc", model_uri="s3://models/llama")
        m = spec.to_manifest()
        assert m["apiVersion"] == "serving.kserve.io/v1alpha1"
        assert m["kind"] == "LLMInferenceService"
        assert m["metadata"]["name"] == "llama-svc"
        assert m["spec"]["model"]["uri"] == "s3://models/llama"

    def test_to_dict(self):
        spec = LLMInferenceServiceSpec(name="x", model_uri="y")
        d = spec.to_dict()
        assert d["runtime"] == "vllm"


class TestRayServeDeploymentSpec:
    def test_defaults(self):
        d = RayServeDeploymentSpec(name="router")
        assert d.num_replicas == 2
        assert d.ray_actor_options == {"num_gpus": 1}

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            RayServeDeploymentSpec(name="")

    def test_invalid_num_replicas_raises(self):
        with pytest.raises(ValueError, match="num_replicas"):
            RayServeDeploymentSpec(name="x", num_replicas=0)

    def test_invalid_max_concurrent_raises(self):
        with pytest.raises(ValueError, match="max_concurrent_queries"):
            RayServeDeploymentSpec(name="x", max_concurrent_queries=0)

    def test_to_dict(self):
        d = RayServeDeploymentSpec(name="model-replica")
        assert d.to_dict()["name"] == "model-replica"


class TestRayServeGraph:
    def test_empty_deployments_raises(self):
        with pytest.raises(ValueError, match="deployments"):
            RayServeGraph(name="g", deployments=[])

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            RayServeGraph(name="", deployments=[RayServeDeploymentSpec(name="x")])

    def test_to_manifest(self):
        g = RayServeGraph(
            name="rag-graph",
            deployments=[RayServeDeploymentSpec(name="router"), RayServeDeploymentSpec(name="model")],
        )
        m = g.to_manifest()
        assert m["applications"][0]["name"] == "rag-graph"
        assert len(m["applications"][0]["deployments"]) == 2


class TestServingBackendAdvisor:
    def test_recommend_custom_routing(self):
        assert ServingBackendAdvisor.recommend(False, True) == "ray_serve"

    def test_recommend_default(self):
        assert ServingBackendAdvisor.recommend(True, False) == "kserve_llm_inference_service"

    def test_explain_ray_serve(self):
        reasons = ServingBackendAdvisor.explain("ray_serve")
        assert len(reasons) > 0

    def test_explain_kserve(self):
        reasons = ServingBackendAdvisor.explain("kserve_llm_inference_service")
        assert len(reasons) > 0

    def test_explain_unknown(self):
        assert ServingBackendAdvisor.explain("bogus") == ["Unknown backend"]
