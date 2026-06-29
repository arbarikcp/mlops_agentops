"""Unit tests for platform/llm/ray_train.py (Day 92)."""

import pytest
from llm.ray_train import (
    CheckpointConfig,
    RayRunConfig,
    RayScalingConfig,
    RayTrainJob,
    ResourceSpec,
)


# ── ResourceSpec ───────────────────────────────────────────────────────────

class TestResourceSpec:
    def test_defaults(self):
        r = ResourceSpec()
        assert r.num_cpus == 2
        assert r.num_gpus == 1
        assert r.memory_gb == 8.0

    def test_custom(self):
        r = ResourceSpec(num_cpus=8, num_gpus=2, memory_gb=32.0)
        assert r.num_cpus == 8

    def test_to_dict(self):
        r = ResourceSpec(num_cpus=4, num_gpus=2, memory_gb=16.0)
        d = r.to_dict()
        assert d == {"num_cpus": 4, "num_gpus": 2, "memory_gb": 16.0}

    def test_invalid_num_cpus(self):
        with pytest.raises(ValueError, match="num_cpus"):
            ResourceSpec(num_cpus=0)

    def test_invalid_memory(self):
        with pytest.raises(ValueError, match="memory_gb"):
            ResourceSpec(memory_gb=-1.0)


# ── RayScalingConfig ───────────────────────────────────────────────────────

class TestRayScalingConfig:
    def test_basic(self):
        cfg = RayScalingConfig(num_workers=4)
        assert cfg.num_workers == 4
        assert cfg.use_gpu is True

    def test_to_dict(self):
        cfg = RayScalingConfig(num_workers=2, use_gpu=False)
        d = cfg.to_dict()
        assert d["num_workers"] == 2
        assert d["use_gpu"] is False
        assert "resources_per_worker" in d

    def test_invalid_workers(self):
        with pytest.raises(ValueError, match="num_workers"):
            RayScalingConfig(num_workers=0)


# ── CheckpointConfig ───────────────────────────────────────────────────────

class TestCheckpointConfig:
    def test_basic(self):
        cfg = CheckpointConfig(checkpoint_dir="/tmp/ckpt")
        assert cfg.num_to_keep == 3

    def test_to_dict(self):
        cfg = CheckpointConfig(checkpoint_dir="s3://bucket/ckpt", num_to_keep=5)
        d = cfg.to_dict()
        assert d["checkpoint_dir"] == "s3://bucket/ckpt"
        assert d["num_to_keep"] == 5

    def test_empty_dir_raises(self):
        with pytest.raises(ValueError, match="checkpoint_dir"):
            CheckpointConfig(checkpoint_dir="")

    def test_invalid_num_to_keep(self):
        with pytest.raises(ValueError, match="num_to_keep"):
            CheckpointConfig(checkpoint_dir="/tmp", num_to_keep=0)


# ── RayRunConfig ───────────────────────────────────────────────────────────

class TestRayRunConfig:
    def test_basic(self):
        cfg = RayRunConfig(name="my-run", storage_path="s3://bucket/runs")
        assert cfg.max_failures == 2

    def test_to_dict_with_checkpoint(self):
        ckpt = CheckpointConfig(checkpoint_dir="/tmp/ckpt")
        cfg = RayRunConfig(name="run-1", storage_path="/tmp", checkpoint=ckpt)
        d = cfg.to_dict()
        assert d["name"] == "run-1"
        assert d["checkpoint"]["checkpoint_dir"] == "/tmp/ckpt"

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            RayRunConfig(name="", storage_path="/tmp")


# ── RayTrainJob ────────────────────────────────────────────────────────────

class TestRayTrainJob:
    def _make_job(self, trainer_type="TorchTrainer"):
        scaling = RayScalingConfig(
            num_workers=4,
            resources_per_worker=ResourceSpec(num_gpus=2),
        )
        run_cfg = RayRunConfig(name="test-job", storage_path="s3://bucket")
        return RayTrainJob(
            name="test-job",
            scaling=scaling,
            run_config=run_cfg,
            trainer_type=trainer_type,
        )

    def test_total_gpus(self):
        job = self._make_job()
        assert job.total_gpus() == 8  # 4 workers * 2 GPUs each

    def test_estimated_cost(self):
        job = self._make_job()
        # 8 GPUs * $3/GPU/hr = $24/hr
        assert job.estimated_cost_per_hour(3.0) == pytest.approx(24.0)

    def test_to_manifest_structure(self):
        job = self._make_job()
        m = job.to_manifest()
        assert m["name"] == "test-job"
        assert m["trainer_type"] == "TorchTrainer"
        assert "scaling_config" in m
        assert "run_config" in m

    def test_huggingface_trainer(self):
        job = self._make_job(trainer_type="HuggingFaceTrainer")
        assert job.trainer_type == "HuggingFaceTrainer"

    def test_invalid_trainer_type(self):
        scaling = RayScalingConfig(num_workers=1)
        run_cfg = RayRunConfig(name="job", storage_path="/tmp")
        with pytest.raises(ValueError, match="trainer_type"):
            RayTrainJob(
                name="job",
                scaling=scaling,
                run_config=run_cfg,
                trainer_type="SparkTrainer",
            )

    def test_empty_name_raises(self):
        scaling = RayScalingConfig(num_workers=1)
        run_cfg = RayRunConfig(name="job", storage_path="/tmp")
        with pytest.raises(ValueError, match="name"):
            RayTrainJob(name="", scaling=scaling, run_config=run_cfg)

    def test_manifest_contains_nested_scaling(self):
        job = self._make_job()
        m = job.to_manifest()
        sc = m["scaling_config"]
        assert sc["num_workers"] == 4
        assert sc["resources_per_worker"]["num_gpus"] == 2
