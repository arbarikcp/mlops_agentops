"""Unit tests for platform/llm/training_opt.py (Day 93)."""

import pytest
from llm.training_opt import (
    DataLoaderConfig,
    GradientConfig,
    MixedPrecisionConfig,
    PrecisionMode,
    TrainingOptAdvisor,
    TrainingOptConfig,
)


# ── MixedPrecisionConfig ───────────────────────────────────────────────────

class TestMixedPrecisionConfig:
    def test_defaults(self):
        cfg = MixedPrecisionConfig()
        assert cfg.mode == PrecisionMode.BF16
        assert cfg.enabled is True

    def test_fp16(self):
        cfg = MixedPrecisionConfig(mode=PrecisionMode.FP16)
        assert cfg.mode == PrecisionMode.FP16

    def test_to_dict(self):
        cfg = MixedPrecisionConfig(mode=PrecisionMode.FP8, enabled=False)
        d = cfg.to_dict()
        assert d["mode"] == "FP8"
        assert d["enabled"] is False


# ── GradientConfig ─────────────────────────────────────────────────────────

class TestGradientConfig:
    def test_defaults(self):
        cfg = GradientConfig()
        assert cfg.checkpointing is False
        assert cfg.accumulation_steps == 1
        assert cfg.max_norm == 1.0

    def test_effective_batch_size(self):
        cfg = GradientConfig(accumulation_steps=4)
        assert cfg.effective_batch_size(8) == 32

    def test_effective_batch_size_no_accumulation(self):
        cfg = GradientConfig(accumulation_steps=1)
        assert cfg.effective_batch_size(16) == 16

    def test_invalid_accumulation_steps(self):
        with pytest.raises(ValueError, match="accumulation_steps"):
            GradientConfig(accumulation_steps=0)

    def test_invalid_max_norm(self):
        with pytest.raises(ValueError, match="max_norm"):
            GradientConfig(max_norm=0.0)

    def test_to_dict(self):
        cfg = GradientConfig(checkpointing=True, accumulation_steps=8, max_norm=2.0)
        d = cfg.to_dict()
        assert d["checkpointing"] is True
        assert d["accumulation_steps"] == 8
        assert d["max_norm"] == 2.0


# ── DataLoaderConfig ───────────────────────────────────────────────────────

class TestDataLoaderConfig:
    def test_defaults(self):
        cfg = DataLoaderConfig()
        assert cfg.num_workers == 4
        assert cfg.pin_memory is True

    def test_invalid_num_workers(self):
        with pytest.raises(ValueError, match="num_workers"):
            DataLoaderConfig(num_workers=-1)

    def test_zero_workers_allowed(self):
        cfg = DataLoaderConfig(num_workers=0)
        assert cfg.num_workers == 0

    def test_to_dict(self):
        cfg = DataLoaderConfig(num_workers=8, prefetch_factor=4)
        d = cfg.to_dict()
        assert d["num_workers"] == 8
        assert d["prefetch_factor"] == 4


# ── TrainingOptConfig ──────────────────────────────────────────────────────

class TestTrainingOptConfig:
    def _make_cfg(self, checkpointing=False, mode=PrecisionMode.FP32):
        return TrainingOptConfig(
            precision=MixedPrecisionConfig(mode=mode),
            gradient=GradientConfig(checkpointing=checkpointing),
            dataloader=DataLoaderConfig(),
        )

    def test_memory_factor_fp32_no_ckpt(self):
        cfg = self._make_cfg(checkpointing=False, mode=PrecisionMode.FP32)
        assert cfg.memory_saving_factor() == pytest.approx(1.0)

    def test_memory_factor_bf16_no_ckpt(self):
        cfg = self._make_cfg(checkpointing=False, mode=PrecisionMode.BF16)
        assert cfg.memory_saving_factor() == pytest.approx(1.25)

    def test_memory_factor_bf16_with_ckpt(self):
        cfg = self._make_cfg(checkpointing=True, mode=PrecisionMode.BF16)
        assert cfg.memory_saving_factor() == pytest.approx(1.75)

    def test_memory_factor_fp32_with_ckpt(self):
        cfg = self._make_cfg(checkpointing=True, mode=PrecisionMode.FP32)
        assert cfg.memory_saving_factor() == pytest.approx(1.5)

    def test_to_dict_structure(self):
        cfg = self._make_cfg()
        d = cfg.to_dict()
        assert "precision" in d
        assert "gradient" in d
        assert "dataloader" in d


# ── TrainingOptAdvisor ─────────────────────────────────────────────────────

class TestTrainingOptAdvisor:
    def test_recommends_checkpointing_when_oom(self):
        # 70B model in FP32 = 280GB >> 80GB GPU
        cfg = TrainingOptAdvisor.recommend(70.0, 80.0)
        assert cfg.gradient.checkpointing is True
        assert cfg.gradient.accumulation_steps == 4
        assert cfg.precision.mode == PrecisionMode.BF16

    def test_no_checkpointing_when_fits(self):
        # 1B model in FP32 = 4GB << 80GB GPU
        cfg = TrainingOptAdvisor.recommend(1.0, 80.0)
        assert cfg.gradient.checkpointing is False
        assert cfg.gradient.accumulation_steps == 1
        assert cfg.precision.mode == PrecisionMode.BF16

    def test_explain_returns_list(self):
        cfg = TrainingOptAdvisor.recommend(70.0, 80.0)
        bullets = TrainingOptAdvisor.explain(cfg)
        assert isinstance(bullets, list)
        assert len(bullets) >= 3

    def test_explain_mentions_precision(self):
        cfg = TrainingOptAdvisor.recommend(1.0, 80.0)
        bullets = TrainingOptAdvisor.explain(cfg)
        combined = " ".join(bullets)
        assert "BF16" in combined

    def test_explain_checkpointing_when_enabled(self):
        cfg = TrainingOptAdvisor.recommend(70.0, 80.0)
        bullets = TrainingOptAdvisor.explain(cfg)
        combined = " ".join(bullets)
        assert "checkpointing" in combined.lower()
