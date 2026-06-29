"""Unit tests for platform/llm/compilation.py (Day 96)."""

import pytest
from llm.compilation import (
    CompilationSpec,
    ONNXConfig,
    RuntimeBackend,
    TRTLLMConfig,
    TorchCompileConfig,
)


# ── TorchCompileConfig ─────────────────────────────────────────────────────

class TestTorchCompileConfig:
    def test_defaults(self):
        cfg = TorchCompileConfig()
        assert cfg.backend == "inductor"
        assert cfg.mode == "default"

    def test_max_autotune(self):
        cfg = TorchCompileConfig(mode="max-autotune")
        assert cfg.mode == "max-autotune"

    def test_reduce_overhead(self):
        cfg = TorchCompileConfig(mode="reduce-overhead")
        assert cfg.mode == "reduce-overhead"

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            TorchCompileConfig(mode="ultra-fast")

    def test_to_dict(self):
        cfg = TorchCompileConfig(mode="max-autotune", dynamic=True)
        d = cfg.to_dict()
        assert d["mode"] == "max-autotune"
        assert d["dynamic"] is True


# ── ONNXConfig ─────────────────────────────────────────────────────────────

class TestONNXConfig:
    def test_defaults(self):
        cfg = ONNXConfig()
        assert cfg.opset_version == 17
        assert cfg.fp16 is True

    def test_invalid_opset(self):
        with pytest.raises(ValueError, match="opset_version"):
            ONNXConfig(opset_version=10)

    def test_valid_opset_11(self):
        cfg = ONNXConfig(opset_version=11)
        assert cfg.opset_version == 11

    def test_to_dict(self):
        cfg = ONNXConfig(opset_version=18, execution_provider="CPUExecutionProvider")
        d = cfg.to_dict()
        assert d["opset_version"] == 18
        assert d["execution_provider"] == "CPUExecutionProvider"


# ── TRTLLMConfig ───────────────────────────────────────────────────────────

class TestTRTLLMConfig:
    def test_defaults(self):
        cfg = TRTLLMConfig()
        assert cfg.max_batch_size == 32
        assert cfg.dtype == "float16"

    def test_bfloat16(self):
        cfg = TRTLLMConfig(dtype="bfloat16")
        assert cfg.dtype == "bfloat16"

    def test_float8(self):
        cfg = TRTLLMConfig(dtype="float8")
        assert cfg.dtype == "float8"

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            TRTLLMConfig(dtype="int8")

    def test_invalid_max_batch_size(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            TRTLLMConfig(max_batch_size=0)

    def test_to_dict(self):
        cfg = TRTLLMConfig(max_batch_size=64, tensor_parallel=4)
        d = cfg.to_dict()
        assert d["max_batch_size"] == 64
        assert d["tensor_parallel"] == 4


# ── CompilationSpec ────────────────────────────────────────────────────────

class TestCompilationSpec:
    def test_pytorch_speedup(self):
        spec = CompilationSpec(model_name="gpt2", backend=RuntimeBackend.PYTORCH)
        assert spec.speedup_estimate() == 1.0

    def test_torch_compile_default_speedup(self):
        spec = CompilationSpec(
            model_name="gpt2",
            backend=RuntimeBackend.TORCH_COMPILE,
            torch_compile=TorchCompileConfig(mode="default"),
        )
        assert spec.speedup_estimate() == pytest.approx(1.3)

    def test_torch_compile_max_autotune_speedup(self):
        spec = CompilationSpec(
            model_name="gpt2",
            backend=RuntimeBackend.TORCH_COMPILE,
            torch_compile=TorchCompileConfig(mode="max-autotune"),
        )
        assert spec.speedup_estimate() == pytest.approx(2.0)

    def test_onnx_speedup(self):
        spec = CompilationSpec(
            model_name="gpt2",
            backend=RuntimeBackend.ONNX_RUNTIME,
            onnx=ONNXConfig(),
        )
        assert spec.speedup_estimate() == pytest.approx(1.5)

    def test_tensorrt_speedup(self):
        spec = CompilationSpec(
            model_name="gpt2",
            backend=RuntimeBackend.TENSORRT_LLM,
            trt_llm=TRTLLMConfig(),
        )
        assert spec.speedup_estimate() == pytest.approx(5.0)

    def test_triton_speedup(self):
        spec = CompilationSpec(
            model_name="gpt2",
            backend=RuntimeBackend.TRITON,
        )
        assert spec.speedup_estimate() == pytest.approx(1.2)

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            CompilationSpec(model_name="", backend=RuntimeBackend.PYTORCH)

    def test_to_dict_structure(self):
        spec = CompilationSpec(
            model_name="llama-7b",
            backend=RuntimeBackend.TENSORRT_LLM,
            trt_llm=TRTLLMConfig(),
        )
        d = spec.to_dict()
        assert d["model_name"] == "llama-7b"
        assert d["backend"] == "TENSORRT_LLM"
        assert d["trt_llm"] is not None
        assert d["onnx"] is None
        assert d["speedup_estimate"] == pytest.approx(5.0)

    def test_torch_compile_no_config_defaults_to_1_3(self):
        # When torch_compile config is None but backend is TORCH_COMPILE
        spec = CompilationSpec(
            model_name="model",
            backend=RuntimeBackend.TORCH_COMPILE,
            torch_compile=None,
        )
        assert spec.speedup_estimate() == pytest.approx(1.3)
