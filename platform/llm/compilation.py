"""
compilation.py — Model Compilation & Runtime Backends (Day 96)

Covers torch.compile, ONNX Runtime, TensorRT-LLM, and Triton
as runtime backends for LLM inference acceleration.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RuntimeBackend(str, Enum):
    """Runtime backend for model compilation and serving."""

    PYTORCH = "PYTORCH"
    ONNX_RUNTIME = "ONNX_RUNTIME"
    TENSORRT_LLM = "TENSORRT_LLM"
    TORCH_COMPILE = "TORCH_COMPILE"
    TRITON = "TRITON"


_VALID_COMPILE_MODES = {"default", "reduce-overhead", "max-autotune"}
_VALID_TRT_DTYPES = {"float16", "bfloat16", "float8"}


@dataclass
class TorchCompileConfig:
    """torch.compile() configuration."""

    backend: str = "inductor"
    mode: str = "default"
    dynamic: bool = False
    fullgraph: bool = False

    def __post_init__(self) -> None:
        if self.mode not in _VALID_COMPILE_MODES:
            raise ValueError(
                f"mode must be one of {_VALID_COMPILE_MODES}, got {self.mode!r}"
            )

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "dynamic": self.dynamic,
            "fullgraph": self.fullgraph,
        }


@dataclass
class ONNXConfig:
    """ONNX export and runtime configuration."""

    opset_version: int = 17
    execution_provider: str = "CUDAExecutionProvider"
    dynamic_axes: dict = field(default_factory=dict)
    fp16: bool = True

    def __post_init__(self) -> None:
        if self.opset_version < 11:
            raise ValueError(
                f"opset_version must be >= 11, got {self.opset_version}"
            )

    def to_dict(self) -> dict:
        return {
            "opset_version": self.opset_version,
            "execution_provider": self.execution_provider,
            "dynamic_axes": self.dynamic_axes,
            "fp16": self.fp16,
        }


@dataclass
class TRTLLMConfig:
    """TensorRT-LLM build configuration."""

    max_batch_size: int = 32
    max_input_len: int = 1024
    max_output_len: int = 512
    dtype: str = "float16"
    tensor_parallel: int = 1

    def __post_init__(self) -> None:
        for name, val in [
            ("max_batch_size", self.max_batch_size),
            ("max_input_len", self.max_input_len),
            ("max_output_len", self.max_output_len),
            ("tensor_parallel", self.tensor_parallel),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")
        if self.dtype not in _VALID_TRT_DTYPES:
            raise ValueError(
                f"dtype must be one of {_VALID_TRT_DTYPES}, got {self.dtype!r}"
            )

    def to_dict(self) -> dict:
        return {
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
            "dtype": self.dtype,
            "tensor_parallel": self.tensor_parallel,
        }


@dataclass
class CompilationSpec:
    """Full compilation specification for a model."""

    model_name: str
    backend: RuntimeBackend
    torch_compile: TorchCompileConfig | None = None
    onnx: ONNXConfig | None = None
    trt_llm: TRTLLMConfig | None = None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")

    def speedup_estimate(self) -> float:
        """
        Estimated inference speedup multiplier over vanilla PyTorch FP32.

        Values reflect published benchmarks:
        - PYTORCH:        1.0x (baseline)
        - TORCH_COMPILE:  1.3x default, 2.0x max-autotune
        - ONNX_RUNTIME:   1.5x
        - TENSORRT_LLM:   5.0x
        - TRITON:         1.2x
        """
        if self.backend == RuntimeBackend.PYTORCH:
            return 1.0
        if self.backend == RuntimeBackend.TORCH_COMPILE:
            if (
                self.torch_compile is not None
                and self.torch_compile.mode == "max-autotune"
            ):
                return 2.0
            return 1.3
        if self.backend == RuntimeBackend.ONNX_RUNTIME:
            return 1.5
        if self.backend == RuntimeBackend.TENSORRT_LLM:
            return 5.0
        if self.backend == RuntimeBackend.TRITON:
            return 1.2
        return 1.0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "backend": self.backend.value,
            "torch_compile": self.torch_compile.to_dict() if self.torch_compile else None,
            "onnx": self.onnx.to_dict() if self.onnx else None,
            "trt_llm": self.trt_llm.to_dict() if self.trt_llm else None,
            "speedup_estimate": self.speedup_estimate(),
        }
