"""
quantization.py — Model Quantization for Serving (Day 95)

Covers PTQ/QAT, GPTQ/AWQ, and distillation configuration.
QuantizationAdvisor recommends the optimal scheme based on
model size vs available GPU VRAM.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuantizationType(str, Enum):
    """Weight quantization format."""

    NONE = "NONE"
    INT8 = "INT8"
    INT4 = "INT4"
    GPTQ = "GPTQ"
    AWQ = "AWQ"
    GGUF = "GGUF"


class QuantizationMethod(str, Enum):
    """Training-time quantization method."""

    PTQ = "PTQ"
    QAT = "QAT"
    DISTILLATION = "DISTILLATION"


_VALID_BITS = {4, 8, 16, 32}


@dataclass
class QuantConfig:
    """Quantization configuration."""

    quant_type: QuantizationType
    method: QuantizationMethod = QuantizationMethod.PTQ
    bits: int = 8
    group_size: int = 128
    desc_act: bool = True

    def __post_init__(self) -> None:
        if self.bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {_VALID_BITS}, got {self.bits}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    def compression_ratio(self) -> float:
        """Compression ratio relative to FP16 (16 bits)."""
        return 16.0 / self.bits

    def to_dict(self) -> dict:
        return {
            "quant_type": self.quant_type.value,
            "method": self.method.value,
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "compression_ratio": self.compression_ratio(),
        }


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""

    teacher_model: str
    student_model: str
    temperature: float = 2.0
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if not self.teacher_model:
            raise ValueError("teacher_model must be non-empty")
        if not self.student_model:
            raise ValueError("student_model must be non-empty")
        if not (0 < self.alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")

    def loss_weight_hard(self) -> float:
        """Weight for hard label loss (complement of soft alpha)."""
        return 1.0 - self.alpha

    def to_dict(self) -> dict:
        return {
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "loss_weight_hard": self.loss_weight_hard(),
        }


@dataclass
class QuantizedModelSpec:
    """Specification for a quantized model artifact."""

    base_model: str
    quant_config: QuantConfig
    original_size_gb: float

    def __post_init__(self) -> None:
        if not self.base_model:
            raise ValueError("base_model must be non-empty")
        if self.original_size_gb <= 0:
            raise ValueError(
                f"original_size_gb must be > 0, got {self.original_size_gb}"
            )

    def quantized_size_gb(self) -> float:
        """Model size after quantization."""
        return self.original_size_gb / self.quant_config.compression_ratio()

    def memory_saving_gb(self) -> float:
        """Memory saved by quantization."""
        return self.original_size_gb - self.quantized_size_gb()

    def fits_on_gpu(self, gpu_vram_gb: float) -> bool:
        """Whether the quantized model fits in GPU VRAM."""
        return self.quantized_size_gb() <= gpu_vram_gb

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "quant_config": self.quant_config.to_dict(),
            "original_size_gb": self.original_size_gb,
            "quantized_size_gb": self.quantized_size_gb(),
            "memory_saving_gb": self.memory_saving_gb(),
        }


class QuantizationAdvisor:
    """Recommends optimal quantization based on model vs GPU VRAM."""

    @staticmethod
    def recommend(model_size_gb: float, target_gpu_vram_gb: float) -> QuantConfig:
        """
        Choose quantization scheme:
        - model > 2x GPU VRAM  → AWQ INT4
        - model > GPU VRAM     → INT8
        - otherwise            → NONE
        """
        if model_size_gb > 2 * target_gpu_vram_gb:
            return QuantConfig(
                quant_type=QuantizationType.AWQ,
                method=QuantizationMethod.PTQ,
                bits=4,
                group_size=128,
            )
        if model_size_gb > target_gpu_vram_gb:
            return QuantConfig(
                quant_type=QuantizationType.INT8,
                method=QuantizationMethod.PTQ,
                bits=8,
                group_size=128,
            )
        return QuantConfig(
            quant_type=QuantizationType.NONE,
            method=QuantizationMethod.PTQ,
            bits=16,
            group_size=128,
        )

    @staticmethod
    def explain(spec: QuantizedModelSpec) -> list[str]:
        """Return human-readable bullet points explaining quantization choices."""
        bullets: list[str] = []
        qt = spec.quant_config.quant_type
        bits = spec.quant_config.bits

        bullets.append(
            f"Base model: {spec.base_model} "
            f"({spec.original_size_gb:.1f} GB in FP16)"
        )
        bullets.append(
            f"Quantization: {qt.value} {bits}-bit → "
            f"{spec.quantized_size_gb():.1f} GB "
            f"({spec.quant_config.compression_ratio():.1f}x compression)"
        )
        bullets.append(f"Memory saving: {spec.memory_saving_gb():.1f} GB")

        if qt == QuantizationType.AWQ:
            bullets.append(
                "AWQ preserves 1% salient weights in FP16 to maintain accuracy"
            )
        elif qt == QuantizationType.GPTQ:
            bullets.append(
                "GPTQ minimizes layer-wise reconstruction error using inverse Hessian"
            )
        elif qt == QuantizationType.INT8:
            bullets.append(
                "INT8 PTQ is fast with < 1% accuracy degradation on most tasks"
            )
        elif qt == QuantizationType.NONE:
            bullets.append("No quantization needed — model fits comfortably in VRAM")

        return bullets
