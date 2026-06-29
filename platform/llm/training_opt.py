"""
training_opt.py — Training Optimization Configuration (Day 93)

Covers mixed precision (FP16/BF16/FP8), gradient checkpointing,
gradient accumulation, optimized data loading, and an advisor
that recommends configs based on model size vs GPU VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PrecisionMode(str, Enum):
    """Floating-point precision modes for training."""

    FP32 = "FP32"
    FP16 = "FP16"
    BF16 = "BF16"
    FP8 = "FP8"


@dataclass
class MixedPrecisionConfig:
    """Mixed precision training configuration."""

    mode: PrecisionMode = PrecisionMode.BF16
    loss_scale: str = "dynamic"
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "loss_scale": self.loss_scale,
            "enabled": self.enabled,
        }


@dataclass
class GradientConfig:
    """Gradient checkpointing and accumulation configuration."""

    checkpointing: bool = False
    accumulation_steps: int = 1
    max_norm: float = 1.0
    clip_enabled: bool = True

    def __post_init__(self) -> None:
        if self.accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {self.accumulation_steps}"
            )
        if self.max_norm <= 0:
            raise ValueError(f"max_norm must be > 0, got {self.max_norm}")

    def effective_batch_size(self, micro_batch: int) -> int:
        """Compute effective batch size = micro_batch * accumulation_steps."""
        return micro_batch * self.accumulation_steps

    def to_dict(self) -> dict:
        return {
            "checkpointing": self.checkpointing,
            "accumulation_steps": self.accumulation_steps,
            "max_norm": self.max_norm,
            "clip_enabled": self.clip_enabled,
        }


@dataclass
class DataLoaderConfig:
    """Optimized DataLoader configuration."""

    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")

    def to_dict(self) -> dict:
        return {
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }


@dataclass
class TrainingOptConfig:
    """Combined training optimization configuration."""

    precision: MixedPrecisionConfig
    gradient: GradientConfig
    dataloader: DataLoaderConfig

    def memory_saving_factor(self) -> float:
        """
        Estimate memory saving factor relative to naive FP32 training.

        1.0 base + 0.5 for gradient checkpointing + 0.25 for non-FP32 precision.
        """
        factor = 1.0
        if self.gradient.checkpointing:
            factor += 0.5
        if self.precision.mode != PrecisionMode.FP32:
            factor += 0.25
        return factor

    def to_dict(self) -> dict:
        return {
            "precision": self.precision.to_dict(),
            "gradient": self.gradient.to_dict(),
            "dataloader": self.dataloader.to_dict(),
        }


class TrainingOptAdvisor:
    """Recommends training optimization configs based on model + GPU specs."""

    @staticmethod
    def recommend(model_params_b: float, gpu_vram_gb: float) -> TrainingOptConfig:
        """
        Recommend a TrainingOptConfig for the given model size and GPU VRAM.

        If model FP32 footprint > GPU VRAM: enable checkpointing + BF16 + accumulation=4.
        Otherwise: BF16 only.
        """
        model_vram_gb = model_params_b * 4  # FP32 = 4 bytes/param

        if model_vram_gb > gpu_vram_gb:
            precision = MixedPrecisionConfig(mode=PrecisionMode.BF16, enabled=True)
            gradient = GradientConfig(
                checkpointing=True,
                accumulation_steps=4,
                max_norm=1.0,
            )
        else:
            precision = MixedPrecisionConfig(mode=PrecisionMode.BF16, enabled=True)
            gradient = GradientConfig(
                checkpointing=False,
                accumulation_steps=1,
                max_norm=1.0,
            )

        return TrainingOptConfig(
            precision=precision,
            gradient=gradient,
            dataloader=DataLoaderConfig(),
        )

    @staticmethod
    def explain(config: TrainingOptConfig) -> list[str]:
        """Return human-readable bullet points explaining the config choices."""
        bullets: list[str] = []

        bullets.append(
            f"Precision mode: {config.precision.mode.value} "
            f"({'enabled' if config.precision.enabled else 'disabled'})"
        )

        if config.gradient.checkpointing:
            bullets.append(
                "Gradient checkpointing enabled — recomputes activations during "
                "backward pass to save ~60% activation memory"
            )

        if config.gradient.accumulation_steps > 1:
            bullets.append(
                f"Gradient accumulation: {config.gradient.accumulation_steps} steps "
                f"— simulates larger batch without extra memory"
            )

        if config.gradient.clip_enabled:
            bullets.append(
                f"Gradient clipping at max_norm={config.gradient.max_norm} "
                "to prevent exploding gradients"
            )

        bullets.append(
            f"DataLoader: {config.dataloader.num_workers} workers, "
            f"prefetch_factor={config.dataloader.prefetch_factor}, "
            f"pin_memory={config.dataloader.pin_memory}"
        )

        bullets.append(
            f"Memory saving factor: {config.memory_saving_factor():.2f}x "
            "vs naive FP32 training"
        )

        return bullets
