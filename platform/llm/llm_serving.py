"""
llm_serving.py — Serving LLMs: KServe LLMInferenceService / Ray Serve (Day 101)

Covers KServe's LLMInferenceService CRD (purpose-built for LLM serving,
distinct from generic InferenceService) and Ray Serve deployment graphs
for LLMs (composable deployments, per-replica autoscaling). Includes a
simple advisor for choosing between the two backends.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RuntimeType(str, Enum):
    """Supported LLM inference runtimes."""

    VLLM = "vllm"
    TGI = "tgi"
    SGLANG = "sglang"


@dataclass
class LLMInferenceServiceSpec:
    """Spec for a KServe LLMInferenceService CRD instance."""

    name: str
    model_uri: str
    runtime: RuntimeType = RuntimeType.VLLM
    min_replicas: int = 1
    max_replicas: int = 5
    namespace: str = "ml-serving"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.model_uri:
            raise ValueError("model_uri must be non-empty")
        if self.min_replicas < 0:
            raise ValueError("min_replicas must be >= 0")
        if self.max_replicas < self.min_replicas:
            raise ValueError("max_replicas must be >= min_replicas")

    def to_manifest(self) -> dict:
        return {
            "apiVersion": "serving.kserve.io/v1alpha1",
            "kind": "LLMInferenceService",
            "metadata": {
                "name": self.name,
                "namespace": self.namespace,
            },
            "spec": {
                "model": {
                    "uri": self.model_uri,
                    "runtime": self.runtime.value,
                },
                "replicas": {
                    "min": self.min_replicas,
                    "max": self.max_replicas,
                },
            },
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_uri": self.model_uri,
            "runtime": self.runtime.value,
            "min_replicas": self.min_replicas,
            "max_replicas": self.max_replicas,
            "namespace": self.namespace,
        }


@dataclass
class RayServeDeploymentSpec:
    """Spec for a single Ray Serve deployment node."""

    name: str
    num_replicas: int = 2
    ray_actor_options: dict = field(default_factory=lambda: {"num_gpus": 1})
    max_concurrent_queries: int = 10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if self.max_concurrent_queries < 1:
            raise ValueError("max_concurrent_queries must be >= 1")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_replicas": self.num_replicas,
            "ray_actor_options": self.ray_actor_options,
            "max_concurrent_queries": self.max_concurrent_queries,
        }


@dataclass
class RayServeGraph:
    """A composable graph of Ray Serve deployments (e.g. router -> model)."""

    name: str
    deployments: list[RayServeDeploymentSpec]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.deployments:
            raise ValueError("deployments must be non-empty")

    def to_manifest(self) -> dict:
        return {
            "applications": [
                {
                    "name": self.name,
                    "deployments": [d.to_dict() for d in self.deployments],
                }
            ]
        }


class ServingBackendAdvisor:
    """Static advisor for choosing between KServe LLMInferenceService and Ray Serve."""

    @staticmethod
    def recommend(expects_multi_model: bool, needs_custom_routing: bool) -> str:
        if needs_custom_routing:
            return "ray_serve"
        return "kserve_llm_inference_service"

    @staticmethod
    def explain(backend: str) -> list[str]:
        if backend == "ray_serve":
            return [
                "Composable deployment graph (router -> model replicas)",
                "Independent autoscaling per deployment node",
                "Best fit for custom routing logic or multi-model pipelines",
            ]
        if backend == "kserve_llm_inference_service":
            return [
                "K8s-native CRD purpose-built for LLM serving",
                "Wraps vLLM/TGI/SGLANG runtime with built-in autoscaling",
                "Best fit for single-model, standard request/response serving",
            ]
        return ["Unknown backend"]
