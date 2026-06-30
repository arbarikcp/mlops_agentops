"""Unit tests for platform/llm/finetuning_ops.py (Day 105)."""

import pytest
from llm.finetuning_ops import (
    EvalGatedPromotion,
    FineTuneDatasetVersion,
    FineTuneJob,
    FineTuneMethod,
    LoRAFinetuneConfig,
    QLoRAConfig,
)


class TestLoRAFinetuneConfig:
    def test_defaults(self):
        c = LoRAFinetuneConfig()
        assert c.rank == 16
        assert c.scaling_factor() == 2.0

    def test_non_power_of_two_rank_raises(self):
        with pytest.raises(ValueError, match="rank"):
            LoRAFinetuneConfig(rank=15)

    def test_zero_rank_raises(self):
        with pytest.raises(ValueError, match="rank"):
            LoRAFinetuneConfig(rank=0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            LoRAFinetuneConfig(alpha=0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            LoRAFinetuneConfig(dropout=1.0)

    def test_to_dict(self):
        c = LoRAFinetuneConfig(rank=8, alpha=16)
        d = c.to_dict()
        assert d["scaling_factor"] == 2.0


class TestQLoRAConfig:
    def test_basic(self):
        q = QLoRAConfig(lora=LoRAFinetuneConfig())
        assert q.quant_bits == 4

    def test_invalid_quant_bits_raises(self):
        with pytest.raises(ValueError, match="quant_bits"):
            QLoRAConfig(lora=LoRAFinetuneConfig(), quant_bits=2)

    def test_to_dict(self):
        q = QLoRAConfig(lora=LoRAFinetuneConfig())
        d = q.to_dict()
        assert "lora" in d


class TestFineTuneDatasetVersion:
    def test_basic(self):
        ds = FineTuneDatasetVersion(name="sft", version="v1", num_examples=100, source_uri="s3://x")
        assert ds.num_examples == 100

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            FineTuneDatasetVersion(name="", version="v1", num_examples=10, source_uri="s3://x")

    def test_zero_examples_raises(self):
        with pytest.raises(ValueError, match="num_examples"):
            FineTuneDatasetVersion(name="x", version="v1", num_examples=0, source_uri="s3://x")

    def test_empty_source_uri_raises(self):
        with pytest.raises(ValueError, match="source_uri"):
            FineTuneDatasetVersion(name="x", version="v1", num_examples=10, source_uri="")


class TestFineTuneJob:
    def _dataset(self):
        return FineTuneDatasetVersion(name="sft", version="v1", num_examples=100, source_uri="s3://x")

    def test_full_method_no_lora_ok(self):
        job = FineTuneJob(base_model="llama-7b", dataset=self._dataset(), method=FineTuneMethod.FULL)
        assert job.lora_config is None

    def test_lora_method_requires_config(self):
        with pytest.raises(ValueError, match="lora_config"):
            FineTuneJob(base_model="llama-7b", dataset=self._dataset(), method=FineTuneMethod.LORA)

    def test_lora_method_with_config_ok(self):
        job = FineTuneJob(
            base_model="llama-7b",
            dataset=self._dataset(),
            method=FineTuneMethod.LORA,
            lora_config=LoRAFinetuneConfig(),
        )
        assert job.method == FineTuneMethod.LORA

    def test_empty_base_model_raises(self):
        with pytest.raises(ValueError, match="base_model"):
            FineTuneJob(base_model="", dataset=self._dataset(), method=FineTuneMethod.FULL)

    def test_invalid_epochs_raises(self):
        with pytest.raises(ValueError, match="num_epochs"):
            FineTuneJob(base_model="x", dataset=self._dataset(), method=FineTuneMethod.FULL, num_epochs=0)

    def test_invalid_lr_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            FineTuneJob(base_model="x", dataset=self._dataset(), method=FineTuneMethod.FULL, learning_rate=0)

    def test_to_dict(self):
        job = FineTuneJob(base_model="x", dataset=self._dataset(), method=FineTuneMethod.FULL)
        d = job.to_dict()
        assert d["lora_config"] is None


class TestEvalGatedPromotion:
    def test_should_promote_true(self):
        g = EvalGatedPromotion(candidate_score=0.9, baseline_score=0.8)
        assert g.should_promote() is True
        assert g.improvement() == pytest.approx(0.1)

    def test_should_promote_false(self):
        g = EvalGatedPromotion(candidate_score=0.7, baseline_score=0.8)
        assert g.should_promote() is False

    def test_min_improvement_tolerance(self):
        g = EvalGatedPromotion(candidate_score=0.81, baseline_score=0.8, min_improvement=0.05)
        assert g.should_promote() is False

    def test_invalid_candidate_score_raises(self):
        with pytest.raises(ValueError, match="candidate_score"):
            EvalGatedPromotion(candidate_score=1.5, baseline_score=0.5)

    def test_invalid_baseline_score_raises(self):
        with pytest.raises(ValueError, match="baseline_score"):
            EvalGatedPromotion(candidate_score=0.5, baseline_score=-0.1)

    def test_to_dict(self):
        g = EvalGatedPromotion(candidate_score=0.9, baseline_score=0.8)
        d = g.to_dict()
        assert d["should_promote"] is True
