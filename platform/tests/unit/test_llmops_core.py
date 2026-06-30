"""Unit tests for platform/llm/llmops_core.py (Day 100)."""

import pytest
from llm.llmops_core import (
    ArtifactType,
    CostMetric,
    LLMOpsVsMLOps,
    NonDeterminismConfig,
    PromptArtifact,
)


class TestPromptArtifact:
    def test_basic_creation(self):
        pa = PromptArtifact(content="You are a helpful assistant.", version="v1")
        assert pa.version == "v1"
        assert pa.artifact_type == ArtifactType.PROMPT
        assert len(pa.content_hash) == 12

    def test_content_hash_deterministic(self):
        pa1 = PromptArtifact(content="hello", version="v1")
        pa2 = PromptArtifact(content="hello", version="v2")
        assert pa1.content_hash == pa2.content_hash

    def test_content_hash_differs_on_content(self):
        pa1 = PromptArtifact(content="hello", version="v1")
        pa2 = PromptArtifact(content="world", version="v1")
        assert pa1.content_hash != pa2.content_hash

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="content"):
            PromptArtifact(content="", version="v1")

    def test_empty_version_raises(self):
        with pytest.raises(ValueError, match="version"):
            PromptArtifact(content="hi", version="")

    def test_to_dict(self):
        pa = PromptArtifact(content="hi", version="v1", author="bhakti")
        d = pa.to_dict()
        assert d["author"] == "bhakti"
        assert d["artifact_type"] == "prompt"
        assert "content_hash" in d


class TestNonDeterminismConfig:
    def test_deterministic_with_seed(self):
        c = NonDeterminismConfig(temperature=0.0, seed=42)
        assert c.is_deterministic() is True

    def test_non_deterministic_without_seed(self):
        c = NonDeterminismConfig(temperature=0.0)
        assert c.is_deterministic() is False

    def test_non_deterministic_with_temp(self):
        c = NonDeterminismConfig(temperature=0.7, seed=42)
        assert c.is_deterministic() is False

    def test_negative_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            NonDeterminismConfig(temperature=-1)

    def test_invalid_top_p_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            NonDeterminismConfig(temperature=0.5, top_p=0)

    def test_to_dict(self):
        c = NonDeterminismConfig(temperature=0.5, top_p=0.9, seed=1)
        d = c.to_dict()
        assert d["seed"] == 1


class TestCostMetric:
    def test_total_cost_calc(self):
        c = CostMetric(
            prompt_tokens=1000,
            completion_tokens=500,
            price_per_1k_prompt=0.01,
            price_per_1k_completion=0.03,
        )
        assert c.total_cost_usd() == pytest.approx(0.01 + 0.015)

    def test_negative_prompt_tokens_raises(self):
        with pytest.raises(ValueError, match="prompt_tokens"):
            CostMetric(-1, 10, 0.01, 0.01)

    def test_negative_completion_tokens_raises(self):
        with pytest.raises(ValueError, match="completion_tokens"):
            CostMetric(10, -1, 0.01, 0.01)

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            CostMetric(10, 10, -0.01, 0.01)

    def test_to_dict_includes_total(self):
        c = CostMetric(100, 100, 0.01, 0.01)
        d = c.to_dict()
        assert "total_cost_usd" in d


class TestLLMOpsVsMLOps:
    def test_compare_has_all_dimensions(self):
        cmp = LLMOpsVsMLOps.compare()
        for dim in [
            "artifact_versioning",
            "determinism",
            "primary_cost_driver",
            "eval_method",
            "failure_mode",
        ]:
            assert dim in cmp
            assert "mlops" in cmp[dim]
            assert "llmops" in cmp[dim]

    def test_compare_returns_dict(self):
        assert isinstance(LLMOpsVsMLOps.compare(), dict)
