"""
ray_train.py — Ray Train Multi-GPU Job Configuration (Day 92)

Covers Ray Train ScalingConfig, RunConfig, CheckpointConfig,
and RayTrainJob for distributed ML without external SDK imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field


_VALID_TRAINER_TYPES = {"TorchTrainer", "HuggingFaceTrainer"}


@dataclass
class ResourceSpec:
    """Per-worker resource specification."""

    num_cpus: int = 2
    num_gpus: int = 1
    memory_gb: float = 8.0

    def __post_init__(self) -> None:
        if self.num_cpus < 1:
            raise ValueError(f"num_cpus must be >= 1, got {self.num_cpus}")
        if self.memory_gb <= 0:
            raise ValueError(f"memory_gb must be > 0, got {self.memory_gb}")

    def to_dict(self) -> dict:
        return {
            "num_cpus": self.num_cpus,
            "num_gpus": self.num_gpus,
            "memory_gb": self.memory_gb,
        }


@dataclass
class RayScalingConfig:
    """Ray Train ScalingConfig equivalent."""

    num_workers: int
    use_gpu: bool = True
    resources_per_worker: ResourceSpec = field(default_factory=ResourceSpec)

    def __post_init__(self) -> None:
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")

    def to_dict(self) -> dict:
        return {
            "num_workers": self.num_workers,
            "use_gpu": self.use_gpu,
            "resources_per_worker": self.resources_per_worker.to_dict(),
        }


@dataclass
class CheckpointConfig:
    """Checkpoint configuration for Ray Train runs."""

    checkpoint_dir: str
    num_to_keep: int = 3
    checkpoint_frequency: int = 1

    def __post_init__(self) -> None:
        if not self.checkpoint_dir:
            raise ValueError("checkpoint_dir must be non-empty")
        if self.num_to_keep < 1:
            raise ValueError(f"num_to_keep must be >= 1, got {self.num_to_keep}")

    def to_dict(self) -> dict:
        return {
            "checkpoint_dir": self.checkpoint_dir,
            "num_to_keep": self.num_to_keep,
            "checkpoint_frequency": self.checkpoint_frequency,
        }


@dataclass
class RayRunConfig:
    """Ray Train RunConfig equivalent."""

    name: str
    storage_path: str
    max_failures: int = 2
    checkpoint: CheckpointConfig | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "storage_path": self.storage_path,
            "max_failures": self.max_failures,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
        }


@dataclass
class RayTrainJob:
    """Top-level Ray Train job specification."""

    name: str
    scaling: RayScalingConfig
    run_config: RayRunConfig
    trainer_type: str = "TorchTrainer"
    mlflow_uri: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.trainer_type not in _VALID_TRAINER_TYPES:
            raise ValueError(
                f"trainer_type must be one of {_VALID_TRAINER_TYPES}, "
                f"got {self.trainer_type!r}"
            )

    def to_manifest(self) -> dict:
        """Return full config dict with all nested configuration."""
        return {
            "name": self.name,
            "trainer_type": self.trainer_type,
            "mlflow_uri": self.mlflow_uri,
            "scaling_config": self.scaling.to_dict(),
            "run_config": self.run_config.to_dict(),
        }

    def total_gpus(self) -> int:
        """Total GPUs across all workers."""
        return self.scaling.num_workers * self.scaling.resources_per_worker.num_gpus

    def estimated_cost_per_hour(self, gpu_cost_usd: float) -> float:
        """Estimated hourly cost based on GPU count and per-GPU price."""
        return self.total_gpus() * gpu_cost_usd
