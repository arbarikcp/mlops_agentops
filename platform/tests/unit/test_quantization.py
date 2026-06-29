"""Unit tests for platform/llm/quantization.py (Day 95)."""

import pytest
from llm.quantization import (
    DistillationConfig,
    QuantConfig,
    QuantizationAdvisor,
    QuantizationMethod,
    QuantizationType,
    QuantizedModelSpec,
)


# ── QuantConfig ────────────────────────────────────────────────────────────

class TestQuantConfig:
    def test_compression_ratio_int4(self):
        cfg = QuantConfig(quant_type=QuantizationType.INT4, bits=4)
        assert cfg.compression_ratio() == pytest.approx(4.0)

    def test_compression_ratio_int8(self):
        cfg = QuantConfig(quant_type=QuantizationType.INT8, bits=8)
        assert cfg.compression_ratio() == pytest.approx(2.0)

    def test_compression_ratio_fp16(self):
        cfg = QuantConfig(quant_type=QuantizationType.NONE, bits=16)
        assert cfg.compression_ratio() == pytest.approx(1.0)

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            QuantConfig(quant_type=QuantizationType.INT8, bits=3)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            QuantConfig(quant_type=QuantizationType.AWQ, bits=4, group_size=0)

    def test_to_dict(self):
        cfg = QuantConfig(quant_type=QuantizationType.GPTQ, bits=4, group_size=64)
        d = cfg.to_dict()
        assert d["quant_type"] == "GPTQ"
        assert d["bits"] == 4
        assert d["compression_ratio"] == pytest.approx(4.0)

    def test_awq_type(self):
        cfg = QuantConfig(quant_type=QuantizationType.AWQ, bits=4)
        assert cfg.quant_type == QuantizationType.AWQ


# ── DistillationConfig ─────────────────────────────────────────────────────

class TestDistillationConfig:
    def test_basic(self):
        cfg = DistillationConfig(
            teacher_model="llama-70b",
            student_model="llama-7b",
        )
        assert cfg.alpha == 0.5
        assert cfg.temperature == 2.0

    def test_loss_weight_hard(self):
        cfg = DistillationConfig(
            teacher_model="gpt4",
            student_model="gpt3",
            alpha=0.7,
        )
        assert cfg.loss_weight_hard() == pytest.approx(0.3)

    def test_empty_teacher_raises(self):
        with pytest.raises(ValueError, match="teacher_model"):
            DistillationConfig(teacher_model="", student_model="small")

    def test_empty_student_raises(self):
        with pytest.raises(ValueError, match="student_model"):
            DistillationConfig(teacher_model="big", student_model="")

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            DistillationConfig(teacher_model="big", student_model="small", alpha=0.0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            DistillationConfig(teacher_model="big", student_model="small", temperature=0.0)

    def test_to_dict(self):
        cfg = DistillationConfig(teacher_model="big", student_model="small", alpha=0.6)
        d = cfg.to_dict()
        assert d["teacher_model"] == "big"
        assert d["loss_weight_hard"] == pytest.approx(0.4)


# ── QuantizedModelSpec ─────────────────────────────────────────────────────

class TestQuantizedModelSpec:
    def _make(self, bits=4, original_size_gb=14.0):
        qcfg = QuantConfig(quant_type=QuantizationType.AWQ, bits=bits)
        return QuantizedModelSpec(
            base_model="llama-7b",
            quant_config=qcfg,
            original_size_gb=original_size_gb,
        )

    def test_quantized_size_int4(self):
        spec = self._make(bits=4, original_size_gb=14.0)
        # 14 / 4 = 3.5 GB
        assert spec.quantized_size_gb() == pytest.approx(3.5)

    def test_memory_saving(self):
        spec = self._make(bits=4, original_size_gb=14.0)
        assert spec.memory_saving_gb() == pytest.approx(10.5)

    def test_fits_on_gpu_true(self):
        spec = self._make(bits=4, original_size_gb=14.0)
        assert spec.fits_on_gpu(4.0) is True

    def test_fits_on_gpu_false(self):
        spec = self._make(bits=8, original_size_gb=14.0)
        # 14/2=7 GB, doesn't fit in 6 GB
        assert spec.fits_on_gpu(6.0) is False

    def test_empty_model_raises(self):
        qcfg = QuantConfig(quant_type=QuantizationType.INT8, bits=8)
        with pytest.raises(ValueError, match="base_model"):
            QuantizedModelSpec(base_model="", quant_config=qcfg, original_size_gb=14.0)

    def test_invalid_size(self):
        qcfg = QuantConfig(quant_type=QuantizationType.INT8, bits=8)
        with pytest.raises(ValueError, match="original_size_gb"):
            QuantizedModelSpec(base_model="model", quant_config=qcfg, original_size_gb=0.0)

    def test_to_dict(self):
        spec = self._make()
        d = spec.to_dict()
        assert "quantized_size_gb" in d
        assert "memory_saving_gb" in d


# ── QuantizationAdvisor ────────────────────────────────────────────────────

class TestQuantizationAdvisor:
    def test_awq_when_model_2x_vram(self):
        # 28 GB model, 8 GB VRAM → model > 2 * VRAM
        cfg = QuantizationAdvisor.recommend(28.0, 8.0)
        assert cfg.quant_type == QuantizationType.AWQ
        assert cfg.bits == 4

    def test_int8_when_model_1x_vram(self):
        # 14 GB model, 10 GB VRAM → model > VRAM but < 2 * VRAM
        cfg = QuantizationAdvisor.recommend(14.0, 10.0)
        assert cfg.quant_type == QuantizationType.INT8
        assert cfg.bits == 8

    def test_none_when_fits(self):
        # 7 GB model, 80 GB VRAM
        cfg = QuantizationAdvisor.recommend(7.0, 80.0)
        assert cfg.quant_type == QuantizationType.NONE

    def test_explain_returns_list(self):
        qcfg = QuantConfig(quant_type=QuantizationType.AWQ, bits=4)
        spec = QuantizedModelSpec(base_model="llama-7b", quant_config=qcfg, original_size_gb=14.0)
        bullets = QuantizationAdvisor.explain(spec)
        assert isinstance(bullets, list)
        assert len(bullets) >= 2

    def test_explain_mentions_awq(self):
        qcfg = QuantConfig(quant_type=QuantizationType.AWQ, bits=4)
        spec = QuantizedModelSpec(base_model="llama-7b", quant_config=qcfg, original_size_gb=14.0)
        bullets = QuantizationAdvisor.explain(spec)
        assert any("AWQ" in b or "salient" in b for b in bullets)
