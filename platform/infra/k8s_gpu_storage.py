"""K8s storage strategies and GPU workload spec builders.

Days 62–63 — VolumeSpec models three model-storage patterns (emptyDir, PVC,
node-local); GPUWorkloadSpec builds pod specs with node selectors + tolerations
for GPU training jobs.

Classes:
  StorageStrategy  — enum: EMPTY_DIR / PVC / NODE_LOCAL
  VolumeSpec       — model-cache volume + PVC manifest builder
  GPUToleration    — one K8s toleration entry
  GPUWorkloadSpec  — pod spec builder for GPU training pods

See: docs/phase9/day62_k8s_storage.md, docs/phase9/day63_gpu_k8s.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── StorageStrategy ────────────────────────────────────────────────────────────

class StorageStrategy(str, Enum):
    EMPTY_DIR  = "emptyDir"   # download per pod — dev/CI only
    PVC        = "pvc"        # shared PVC downloaded once
    NODE_LOCAL = "nodeLocal"  # first-pod-per-node caches model


# ── VolumeSpec ─────────────────────────────────────────────────────────────────

@dataclass
class VolumeSpec:
    """Model-cache volume specification for K8s pod specs.

    Attributes:
        name:          Volume name referenced in pod spec.
        strategy:      Storage pattern to use.
        mount_path:    Path inside the container.
        pvc_name:      PVC name (only relevant for PVC strategy).
        storage_size:  PVC storage size request (e.g., "5Gi").
        access_modes:  PVC access modes list.
    """

    name: str = "model-volume"
    strategy: StorageStrategy = StorageStrategy.EMPTY_DIR
    mount_path: str = "/model"
    pvc_name: str = "model-cache"
    storage_size: str = "5Gi"
    access_modes: list[str] = field(default_factory=lambda: ["ReadOnlyMany"])

    def to_volume_dict(self) -> dict[str, Any]:
        """Return the volumes[] entry for a pod spec."""
        if self.strategy == StorageStrategy.EMPTY_DIR:
            return {"name": self.name, "emptyDir": {}}
        if self.strategy == StorageStrategy.PVC:
            return {
                "name": self.name,
                "persistentVolumeClaim": {"claimName": self.pvc_name, "readOnly": True},
            }
        # NODE_LOCAL: local PV with host path (first pod per node populates it)
        return {
            "name": self.name,
            "hostPath": {"path": f"/var/ml-model-cache/{self.pvc_name}", "type": "DirectoryOrCreate"},
        }

    def to_volume_mount_dict(self) -> dict[str, Any]:
        """Return the volumeMounts[] entry for a container spec."""
        return {"name": self.name, "mountPath": self.mount_path}

    def to_pvc_manifest(self) -> dict[str, Any]:
        """Return a PVC manifest dict (only meaningful for PVC strategy)."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {"name": self.pvc_name},
            "spec": {
                "accessModes": self.access_modes,
                "resources": {"requests": {"storage": self.storage_size}},
            },
        }


# ── GPUToleration ─────────────────────────────────────────────────────────────

@dataclass
class GPUToleration:
    """One K8s toleration for a GPU-tainted node.

    Attributes:
        key:      Taint key (e.g., "nvidia.com/gpu").
        operator: "Exists" or "Equal".
        effect:   "NoSchedule", "NoExecute", or "PreferNoSchedule".
        value:    Taint value (required if operator="Equal").
    """

    key: str = "nvidia.com/gpu"
    operator: str = "Exists"
    effect: str = "NoSchedule"
    value: str = ""

    def __post_init__(self) -> None:
        if self.operator not in {"Exists", "Equal"}:
            raise ValueError(f"operator must be 'Exists' or 'Equal'; got {self.operator!r}")
        if self.effect not in {"NoSchedule", "NoExecute", "PreferNoSchedule"}:
            raise ValueError(f"invalid effect: {self.effect!r}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "key": self.key,
            "operator": self.operator,
            "effect": self.effect,
        }
        if self.operator == "Equal":
            d["value"] = self.value
        return d


# ── GPUWorkloadSpec ────────────────────────────────────────────────────────────

@dataclass
class GPUWorkloadSpec:
    """Pod spec builder for GPU ML workloads (training or inference).

    Produces the `spec:` section of a Pod/Job manifest with GPU resource
    requests, node selectors, tolerations, and correct resource limits.

    Attributes:
        name:           Container name.
        image:          Docker image (should include CUDA runtime).
        gpu_count:      Number of GPUs to request (and limit).
        cpu_request:    CPU request.
        memory_request: Memory request.
        node_selector:  K8s node selector labels dict.
        tolerations:    List of GPUToleration objects.
        command:        Container command override.
        env:            Environment variable dict.
    """

    name: str
    image: str
    gpu_count: int = 1
    cpu_request: str = "4"
    memory_request: str = "16Gi"
    node_selector: dict[str, str] = field(
        default_factory=lambda: {"node-type": "gpu", "nvidia.com/gpu": "true"}
    )
    tolerations: list[GPUToleration] = field(
        default_factory=lambda: [GPUToleration()]
    )
    command: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GPUWorkloadSpec.name cannot be empty")
        if self.gpu_count < 1:
            raise ValueError("gpu_count must be >= 1")

    def to_pod_spec(self) -> dict[str, Any]:
        """Return the pod spec dict (`.spec` of a Pod or Job template)."""
        container: dict[str, Any] = {
            "name": self.name,
            "image": self.image,
            "resources": {
                "requests": {
                    "cpu": self.cpu_request,
                    "memory": self.memory_request,
                    "nvidia.com/gpu": str(self.gpu_count),
                },
                "limits": {
                    "nvidia.com/gpu": str(self.gpu_count),
                },
            },
        }
        if self.command:
            container["command"] = self.command
        if self.env:
            container["env"] = [{"name": k, "value": v} for k, v in self.env.items()]

        spec: dict[str, Any] = {
            "nodeSelector": self.node_selector,
            "tolerations": [t.to_dict() for t in self.tolerations],
            "containers": [container],
        }
        return spec
