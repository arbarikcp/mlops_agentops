"""Unit tests for platform/llm/distributed.py (Day 91)."""

import pytest
from llm.distributed import (
    DDPConfig,
    FSDPConfig,
    ParallelismPlan,
    ParallelismStrategy,
    ZeroMemoryEstimator,
    ZeroStage,
)


# ── DDPConfig ──────────────────────────────────────────────────────────────

class TestDDPConfig:
    def test_basic_creation(self):
        cfg = DDPConfig(backend="nccl", world_size=4)
        assert cfg.backend == "nccl"
        assert cfg.world_size == 4
        assert cfg.find_unused_parameters is False

    def test_gloo_backend(self):
        cfg = DDPConfig(backend="gloo", world_size=2)
        assert cfg.backend == "gloo"

    def test_to_dict(self):
        cfg = DDPConfig(backend="nccl", world_size=8, find_unused_parameters=True)
        d = cfg.to_dict()
        assert d["backend"] == "nccl"
        assert d["world_size"] == 8
        assert d["find_unused_parameters"] is True

    def test_invalid_world_size(self):
        with pytest.raises(ValueError, match="world_size"):
            DDPConfig(backend="nccl", world_size=0)

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            DDPConfig(backend="cuda", world_size=2)


# ── FSDPConfig ─────────────────────────────────────────────────────────────

class TestFSDPConfig:
    def test_basic_creation(self):
        cfg = FSDPConfig(zero_stage=ZeroStage.FULL)
        assert cfg.zero_stage == ZeroStage.FULL
        assert cfg.sharding_strategy == "FULL_SHARD"

    def test_shard_grad_op(self):
        cfg = FSDPConfig(zero_stage=ZeroStage.GRADIENT, sharding_strategy="SHARD_GRAD_OP")
        assert cfg.sharding_strategy == "SHARD_GRAD_OP"

    def test_invalid_sharding_strategy(self):
        with pytest.raises(ValueError, match="sharding_strategy"):
            FSDPConfig(zero_stage=ZeroStage.FULL, sharding_strategy="INVALID")

    def test_to_dict(self):
        cfg = FSDPConfig(zero_stage=ZeroStage.OPTIMIZER, cpu_offload=True)
        d = cfg.to_dict()
        assert d["zero_stage"] == 1
        assert d["cpu_offload"] is True
        assert d["sharding_strategy"] == "FULL_SHARD"


# ── ParallelismPlan ────────────────────────────────────────────────────────

class TestParallelismPlan:
    def test_data_parallel_memory(self):
        plan = ParallelismPlan(strategy=ParallelismStrategy.DATA, num_gpus=4)
        # 7B model * 4 bytes = 28GB, split across 4 GPUs = 7GB each
        mem = plan.memory_per_gpu_gb(7.0)
        assert abs(mem - 7.0) < 0.01

    def test_fsdp_full_memory(self):
        fsdp = FSDPConfig(zero_stage=ZeroStage.FULL)
        plan = ParallelismPlan(
            strategy=ParallelismStrategy.MODEL,
            num_gpus=4,
            num_nodes=2,
            fsdp_config=fsdp,
        )
        # 7B * 4 bytes = 28GB / (4 * 2) = 3.5GB each
        mem = plan.memory_per_gpu_gb(7.0)
        assert abs(mem - 3.5) < 0.01

    def test_invalid_num_gpus(self):
        with pytest.raises(ValueError, match="num_gpus"):
            ParallelismPlan(strategy=ParallelismStrategy.DATA, num_gpus=0)

    def test_invalid_num_nodes(self):
        with pytest.raises(ValueError, match="num_nodes"):
            ParallelismPlan(strategy=ParallelismStrategy.DATA, num_gpus=2, num_nodes=0)

    def test_to_dict_with_ddp(self):
        ddp = DDPConfig(backend="nccl", world_size=4)
        plan = ParallelismPlan(
            strategy=ParallelismStrategy.DATA,
            num_gpus=4,
            ddp_config=ddp,
        )
        d = plan.to_dict()
        assert d["strategy"] == "DATA"
        assert d["ddp_config"]["backend"] == "nccl"
        assert d["fsdp_config"] is None

    def test_to_dict_without_configs(self):
        plan = ParallelismPlan(strategy=ParallelismStrategy.TENSOR, num_gpus=8)
        d = plan.to_dict()
        assert d["num_gpus"] == 8
        assert d["ddp_config"] is None


# ── ZeroMemoryEstimator ────────────────────────────────────────────────────

class TestZeroMemoryEstimator:
    def test_stage0_full_replication(self):
        result = ZeroMemoryEstimator.estimate(7.0, ZeroStage.DISABLED, 1)
        # 7B * 2B = 14GB params, 14GB grads, 84GB opt
        assert result["param_gb"] == pytest.approx(14.0, rel=1e-3)
        assert result["grad_gb"] == pytest.approx(14.0, rel=1e-3)
        assert result["optimizer_gb"] == pytest.approx(84.0, rel=1e-3)
        assert result["total_gb"] == pytest.approx(112.0, rel=1e-3)

    def test_stage1_optimizer_sharded(self):
        result = ZeroMemoryEstimator.estimate(7.0, ZeroStage.OPTIMIZER, 8)
        assert result["param_gb"] == pytest.approx(14.0, rel=1e-3)
        assert result["grad_gb"] == pytest.approx(14.0, rel=1e-3)
        assert result["optimizer_gb"] == pytest.approx(84.0 / 8, rel=1e-3)

    def test_stage2_grad_and_opt_sharded(self):
        result = ZeroMemoryEstimator.estimate(7.0, ZeroStage.GRADIENT, 8)
        assert result["param_gb"] == pytest.approx(14.0, rel=1e-3)
        assert result["grad_gb"] == pytest.approx(14.0 / 8, rel=1e-3)
        assert result["optimizer_gb"] == pytest.approx(84.0 / 8, rel=1e-3)

    def test_stage3_all_sharded(self):
        result = ZeroMemoryEstimator.estimate(7.0, ZeroStage.FULL, 8)
        assert result["param_gb"] == pytest.approx(14.0 / 8, rel=1e-3)
        assert result["grad_gb"] == pytest.approx(14.0 / 8, rel=1e-3)
        assert result["optimizer_gb"] == pytest.approx(84.0 / 8, rel=1e-3)

    def test_invalid_world_size(self):
        with pytest.raises(ValueError, match="world_size"):
            ZeroMemoryEstimator.estimate(7.0, ZeroStage.FULL, 0)

    def test_total_is_sum(self):
        result = ZeroMemoryEstimator.estimate(13.0, ZeroStage.GRADIENT, 4)
        expected = result["param_gb"] + result["grad_gb"] + result["optimizer_gb"]
        assert result["total_gb"] == pytest.approx(expected, rel=1e-3)

    def test_stage3_saves_more_than_stage0(self):
        r0 = ZeroMemoryEstimator.estimate(7.0, ZeroStage.DISABLED, 8)
        r3 = ZeroMemoryEstimator.estimate(7.0, ZeroStage.FULL, 8)
        assert r3["total_gb"] < r0["total_gb"]


# ── Enum values ─────────────────────────────────────────────────────────────

def test_parallelism_strategy_values():
    assert ParallelismStrategy.DATA.value == "DATA"
    assert ParallelismStrategy.HYBRID.value == "HYBRID"

def test_zero_stage_values():
    assert ZeroStage.DISABLED.value == 0
    assert ZeroStage.FULL.value == 3
