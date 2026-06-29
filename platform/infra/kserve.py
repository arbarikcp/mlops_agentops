"""KServe InferenceService and canary traffic config builders.

Days 64–65 — generates KServe InferenceService manifests and canary traffic
split configs as Python dicts. No KServe CRD or cluster required.

Classes:
  InferenceServiceSpec  — InferenceService manifest builder (predictor + optional transformer)
  CanaryConfig          — canary traffic split manager (promote / rollback)

See: docs/phase9/day64_kserve.md, docs/phase9/day65_kserve_canary.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_KSERVE_API = "serving.kserve.io/v1beta1"
_VALID_FORMATS = {"sklearn", "pytorch", "onnx", "tensorflow", "lightgbm", "xgboost"}
_VALID_SCALE_METRICS = {"rps", "concurrency", "cpu", "memory"}


# ── InferenceServiceSpec ──────────────────────────────────────────────────────

@dataclass
class InferenceServiceSpec:
    """Builds a KServe InferenceService manifest.

    Attributes:
        name:               InferenceService name (= model endpoint name).
        namespace:          Target namespace.
        model_format:       Predictor runtime (sklearn/pytorch/onnx/etc.).
        storage_uri:        S3 / PVC URI to the model artifact.
        min_replicas:       Minimum serving replicas (0 = scale-to-zero).
        max_replicas:       Maximum serving replicas.
        scale_target:       Scale trigger threshold (rps/concurrency per pod).
        scale_metric:       Scale metric type.
        cpu_request:        Predictor CPU request.
        memory_request:     Predictor memory request.
        cpu_limit:          Predictor CPU limit.
        memory_limit:       Predictor memory limit.
        has_transformer:    Whether to include a transformer sidecar.
        transformer_image:  Docker image for the transformer container.
    """

    name: str
    namespace: str = "ml-serving"
    model_format: str = "sklearn"
    storage_uri: str = ""
    min_replicas: int = 1
    max_replicas: int = 10
    scale_target: int = 10
    scale_metric: str = "rps"
    cpu_request: str = "500m"
    memory_request: str = "512Mi"
    cpu_limit: str = "2"
    memory_limit: str = "2Gi"
    has_transformer: bool = False
    transformer_image: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("InferenceServiceSpec.name cannot be empty")
        if self.model_format not in _VALID_FORMATS:
            raise ValueError(f"model_format must be one of {_VALID_FORMATS}")
        if self.scale_metric not in _VALID_SCALE_METRICS:
            raise ValueError(f"scale_metric must be one of {_VALID_SCALE_METRICS}")
        if self.min_replicas < 0:
            raise ValueError("min_replicas cannot be negative")
        if self.min_replicas > self.max_replicas:
            raise ValueError("min_replicas cannot exceed max_replicas")

    def _predictor_spec(self) -> dict[str, Any]:
        return {
            "minReplicas": self.min_replicas,
            "maxReplicas": self.max_replicas,
            "scaleTarget": self.scale_target,
            "scaleMetric": self.scale_metric,
            "model": {
                "modelFormat": {"name": self.model_format},
                "storageUri": self.storage_uri,
                "resources": {
                    "requests": {"cpu": self.cpu_request, "memory": self.memory_request},
                    "limits": {"cpu": self.cpu_limit, "memory": self.memory_limit},
                },
            },
        }

    def to_manifest(self) -> dict[str, Any]:
        """Return the InferenceService manifest as a Python dict."""
        spec: dict[str, Any] = {"predictor": self._predictor_spec()}
        if self.has_transformer and self.transformer_image:
            spec["transformer"] = {
                "containers": [{
                    "name": f"{self.name}-transformer",
                    "image": self.transformer_image,
                    "resources": {
                        "requests": {"cpu": "200m", "memory": "256Mi"},
                        "limits": {"cpu": "1", "memory": "512Mi"},
                    },
                    "env": [{"name": "PREDICTOR_HOST", "value": "localhost"}],
                }]
            }
        return {
            "apiVersion": _KSERVE_API,
            "kind": "InferenceService",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": spec,
        }


# ── CanaryConfig ──────────────────────────────────────────────────────────────

@dataclass
class CanaryConfig:
    """Manages KServe canary traffic split between stable and canary models.

    Attributes:
        name:                 InferenceService name.
        namespace:            Target namespace.
        stable_storage_uri:   S3/PVC URI for the stable (current prod) model.
        canary_storage_uri:   S3/PVC URI for the canary (new) model.
        canary_traffic_pct:   Percentage of traffic sent to canary (0–100).
        model_format:         Model format string.
        min_replicas:         Minimum replicas.
        max_replicas:         Maximum replicas.
    """

    name: str
    namespace: str = "ml-serving"
    stable_storage_uri: str = ""
    canary_storage_uri: str = ""
    canary_traffic_pct: int = 10
    model_format: str = "sklearn"
    min_replicas: int = 1
    max_replicas: int = 10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CanaryConfig.name cannot be empty")
        self._validate_pct(self.canary_traffic_pct)

    @staticmethod
    def _validate_pct(pct: int) -> None:
        if not (0 <= pct <= 100):
            raise ValueError(f"canary_traffic_pct must be 0–100; got {pct}")

    def to_manifest(self) -> dict[str, Any]:
        """Return InferenceService manifest with canary traffic split."""
        return {
            "apiVersion": _KSERVE_API,
            "kind": "InferenceService",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "predictor": {
                    "canaryTrafficPercent": self.canary_traffic_pct,
                    "minReplicas": self.min_replicas,
                    "maxReplicas": self.max_replicas,
                    "model": {
                        "modelFormat": {"name": self.model_format},
                        "storageUri": self.canary_storage_uri,
                    },
                }
            },
        }

    def to_patch_dict(self, new_pct: int) -> dict[str, Any]:
        """Return a strategic merge patch dict to update canaryTrafficPercent."""
        self._validate_pct(new_pct)
        return {
            "spec": {
                "predictor": {"canaryTrafficPercent": new_pct}
            }
        }

    def promote(self) -> "CanaryConfig":
        """Return a new CanaryConfig with canary promoted to 100% (full rollout)."""
        return CanaryConfig(
            name=self.name,
            namespace=self.namespace,
            stable_storage_uri=self.canary_storage_uri,  # canary becomes stable
            canary_storage_uri=self.canary_storage_uri,
            canary_traffic_pct=100,
            model_format=self.model_format,
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
        )

    def rollback(self) -> "CanaryConfig":
        """Return a new CanaryConfig rolling back to 0% canary traffic."""
        return CanaryConfig(
            name=self.name,
            namespace=self.namespace,
            stable_storage_uri=self.stable_storage_uri,
            canary_storage_uri=self.canary_storage_uri,
            canary_traffic_pct=0,
            model_format=self.model_format,
            min_replicas=self.min_replicas,
            max_replicas=self.max_replicas,
        )
