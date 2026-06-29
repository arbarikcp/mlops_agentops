"""
distributed.py — Distributed Training Parallelism Configuration (Day 91)

Covers data/model/pipeline/tensor parallelism strategies,
DDP/FSDP configurations, and ZeRO memory estimation.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParallelismStrategy(str, Enum):
    """Parallelism strategy for distributed training."""

    DATA = "DATA"
    MODEL = "MODEL"
    PIPELINE = "PIPELINE"
    TENSOR = "TENSOR"
    HYBRID = "HYBRID"


class ZeroStage(int, Enum):
    """ZeRO optimization stages (DeepSpeed / FSDP)."""

    DISABLED = 0
    OPTIMIZER = 1   # shard optimizer states
    GRADIENT = 2    # shard gradients + optimizer states
    FULL = 3        # shard params + gradients + optimizer states


_VALID_SHARDING_STRATEGIES = {"FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"}


@dataclass
class DDPConfig:
    """Configuration for DistributedDataParallel training."""

    backend: str
    world_size: int
    find_unused_parameters: bool = False

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if self.backend not in ("nccl", "gloo", "mpi"):
            raise ValueError(f"backend must be nccl/gloo/mpi, got {self.backend!r}")

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "world_size": self.world_size,
            "find_unused_parameters": self.find_unused_parameters,
        }


@dataclass
class FSDPConfig:
    """Configuration for Fully Sharded Data Parallel training."""

    zero_stage: ZeroStage
    cpu_offload: bool = False
    mixed_precision: bool = True
    sharding_strategy: str = "FULL_SHARD"

    def __post_init__(self) -> None:
        if self.sharding_strategy not in _VALID_SHARDING_STRATEGIES:
            raise ValueError(
                f"sharding_strategy must be one of {_VALID_SHARDING_STRATEGIES}, "
                f"got {self.sharding_strategy!r}"
            )

    def to_dict(self) -> dict:
        return {
            "zero_stage": self.zero_stage.value,
            "cpu_offload": self.cpu_offload,
            "mixed_precision": self.mixed_precision,
            "sharding_strategy": self.sharding_strategy,
        }


@dataclass
class ParallelismPlan:
    """Full parallelism plan for distributed training."""

    strategy: ParallelismStrategy
    num_gpus: int
    num_nodes: int = 1
    ddp_config: DDPConfig | None = None
    fsdp_config: FSDPConfig | None = None

    def __post_init__(self) -> None:
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1, got {self.num_gpus}")
        if self.num_nodes < 1:
            raise ValueError(f"num_nodes must be >= 1, got {self.num_nodes}")

    def memory_per_gpu_gb(self, model_params_b: float) -> float:
        """Estimate memory per GPU in GB for model with model_params_b billion parameters."""
        bytes_per_param = 4  # FP32
        total_bytes = model_params_b * 1e9 * bytes_per_param

        if self.strategy == ParallelismStrategy.DATA:
            return total_bytes / 1e9 / self.num_gpus

        if (
            self.strategy == ParallelismStrategy.MODEL
            and self.fsdp_config is not None
            and self.fsdp_config.zero_stage == ZeroStage.FULL
        ):
            return total_bytes / 1e9 / (self.num_gpus * self.num_nodes)

        # Default for PIPELINE, TENSOR, HYBRID: shard across all GPUs
        return total_bytes / 1e9 / (self.num_gpus * self.num_nodes)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "num_gpus": self.num_gpus,
            "num_nodes": self.num_nodes,
            "ddp_config": self.ddp_config.to_dict() if self.ddp_config else None,
            "fsdp_config": self.fsdp_config.to_dict() if self.fsdp_config else None,
        }


class ZeroMemoryEstimator:
    """Estimates per-GPU memory usage under different ZeRO stages."""

    @staticmethod
    def estimate(
        model_params_b: float,
        stage: ZeroStage,
        world_size: int,
    ) -> dict[str, float]:
        """
        Estimate memory consumption per GPU in GB.

        Parameters
        ----------
        model_params_b : float
            Model size in billions of parameters.
        stage : ZeroStage
            ZeRO optimization stage.
        world_size : int
            Number of GPUs (data-parallel degree).

        Returns
        -------
        dict with param_gb, grad_gb, optimizer_gb, total_gb.
        """
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")

        # FP16 = 2 bytes/param; Adam optimizer = 12 bytes/param (FP32 master + m + v)
        param_bytes = model_params_b * 1e9 * 2
        grad_bytes = model_params_b * 1e9 * 2
        opt_bytes = model_params_b * 1e9 * 12

        if stage == ZeroStage.DISABLED:
            p_gb = param_bytes / 1e9
            g_gb = grad_bytes / 1e9
            o_gb = opt_bytes / 1e9
        elif stage == ZeroStage.OPTIMIZER:
            p_gb = param_bytes / 1e9
            g_gb = grad_bytes / 1e9
            o_gb = opt_bytes / 1e9 / world_size
        elif stage == ZeroStage.GRADIENT:
            p_gb = param_bytes / 1e9
            g_gb = grad_bytes / 1e9 / world_size
            o_gb = opt_bytes / 1e9 / world_size
        else:  # ZeroStage.FULL
            p_gb = param_bytes / 1e9 / world_size
            g_gb = grad_bytes / 1e9 / world_size
            o_gb = opt_bytes / 1e9 / world_size

        return {
            "param_gb": round(p_gb, 4),
            "grad_gb": round(g_gb, 4),
            "optimizer_gb": round(o_gb, 4),
            "total_gb": round(p_gb + g_gb + o_gb, 4),
        }
