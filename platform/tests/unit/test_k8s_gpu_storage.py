"""Tests for infra/k8s_gpu_storage.py — VolumeSpec, GPUToleration, GPUWorkloadSpec."""
from __future__ import annotations

import pytest

from infra.k8s_gpu_storage import (
    GPUToleration,
    GPUWorkloadSpec,
    StorageStrategy,
    VolumeSpec,
)


# ── VolumeSpec ──────────────────────────────────────────────────────────────────

class TestVolumeSpec:
    def test_empty_dir_volume(self) -> None:
        v = VolumeSpec(strategy=StorageStrategy.EMPTY_DIR)
        d = v.to_volume_dict()
        assert "emptyDir" in d

    def test_pvc_volume(self) -> None:
        v = VolumeSpec(strategy=StorageStrategy.PVC, pvc_name="model-cache")
        d = v.to_volume_dict()
        assert "persistentVolumeClaim" in d
        assert d["persistentVolumeClaim"]["claimName"] == "model-cache"

    def test_node_local_volume(self) -> None:
        v = VolumeSpec(strategy=StorageStrategy.NODE_LOCAL)
        d = v.to_volume_dict()
        assert "hostPath" in d

    def test_volume_mount(self) -> None:
        v = VolumeSpec(mount_path="/model")
        d = v.to_volume_mount_dict()
        assert d["mountPath"] == "/model"

    def test_pvc_manifest(self) -> None:
        v = VolumeSpec(strategy=StorageStrategy.PVC, storage_size="10Gi")
        m = v.to_pvc_manifest()
        assert m["kind"] == "PersistentVolumeClaim"
        assert m["spec"]["resources"]["requests"]["storage"] == "10Gi"

    def test_pvc_manifest_access_modes(self) -> None:
        v = VolumeSpec(access_modes=["ReadOnlyMany"])
        m = v.to_pvc_manifest()
        assert "ReadOnlyMany" in m["spec"]["accessModes"]

    def test_volume_name_in_dict(self) -> None:
        v = VolumeSpec(name="my-volume")
        assert v.to_volume_dict()["name"] == "my-volume"
        assert v.to_volume_mount_dict()["name"] == "my-volume"


# ── GPUToleration ───────────────────────────────────────────────────────────────

class TestGPUToleration:
    def test_exists_operator(self) -> None:
        t = GPUToleration(operator="Exists")
        d = t.to_dict()
        assert d["operator"] == "Exists"
        assert "value" not in d

    def test_equal_operator_includes_value(self) -> None:
        t = GPUToleration(operator="Equal", value="present")
        d = t.to_dict()
        assert d["value"] == "present"

    def test_invalid_operator_raises(self) -> None:
        with pytest.raises(ValueError, match="operator"):
            GPUToleration(operator="BadOp")

    def test_invalid_effect_raises(self) -> None:
        with pytest.raises(ValueError, match="effect"):
            GPUToleration(effect="BadEffect")

    def test_no_execute_effect(self) -> None:
        t = GPUToleration(effect="NoExecute")
        assert t.to_dict()["effect"] == "NoExecute"

    def test_default_key(self) -> None:
        t = GPUToleration()
        assert t.to_dict()["key"] == "nvidia.com/gpu"


# ── GPUWorkloadSpec ─────────────────────────────────────────────────────────────

class TestGPUWorkloadSpec:
    def _spec(self, **kw) -> GPUWorkloadSpec:
        defaults = dict(name="trainer", image="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime")
        return GPUWorkloadSpec(**{**defaults, **kw})

    def test_basic_pod_spec(self) -> None:
        spec = self._spec().to_pod_spec()
        assert "containers" in spec
        assert "nodeSelector" in spec
        assert "tolerations" in spec

    def test_gpu_in_resources(self) -> None:
        spec = self._spec(gpu_count=2).to_pod_spec()
        container = spec["containers"][0]
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "2"
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "2"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            GPUWorkloadSpec(name="", image="img")

    def test_gpu_count_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="gpu_count"):
            GPUWorkloadSpec(name="t", image="img", gpu_count=0)

    def test_node_selector_present(self) -> None:
        spec = self._spec().to_pod_spec()
        assert "node-type" in spec["nodeSelector"]

    def test_tolerations_present(self) -> None:
        spec = self._spec().to_pod_spec()
        assert len(spec["tolerations"]) >= 1

    def test_command_included(self) -> None:
        spec = self._spec(command=["python", "train.py"]).to_pod_spec()
        assert spec["containers"][0]["command"] == ["python", "train.py"]

    def test_env_vars_included(self) -> None:
        spec = self._spec(env={"EPOCHS": "10"}).to_pod_spec()
        env_map = {e["name"]: e["value"] for e in spec["containers"][0]["env"]}
        assert env_map["EPOCHS"] == "10"

    def test_no_command_no_key(self) -> None:
        spec = self._spec().to_pod_spec()
        assert "command" not in spec["containers"][0]
