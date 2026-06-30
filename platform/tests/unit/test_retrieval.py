"""Unit tests for platform/llm/retrieval.py (Day 110)."""

import pytest

from llm.index_pipeline import ChunkingStrategy
from llm.retrieval import (
    ChunkExperimentResult,
    HybridRetrievalConfig,
    RerankerConfig,
    RerankResult,
    RetrievalMethod,
    RRFFuser,
)


class TestRetrievalMethod:
    def test_members(self):
        assert RetrievalMethod.BM25.value == "bm25"
        assert RetrievalMethod.VECTOR.value == "vector"
        assert RetrievalMethod.HYBRID.value == "hybrid"


class TestChunkExperimentResult:
    def test_basic(self):
        r = ChunkExperimentResult(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=256,
            mean_relevance_score=0.8,
            mean_chunks_per_doc=4.5,
        )
        assert r.chunk_size == 256

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkExperimentResult(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=0,
                mean_relevance_score=0.5,
                mean_chunks_per_doc=2.0,
            )

    def test_relevance_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="mean_relevance_score"):
            ChunkExperimentResult(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=256,
                mean_relevance_score=1.5,
                mean_chunks_per_doc=2.0,
            )

    def test_negative_relevance_score_raises(self):
        with pytest.raises(ValueError, match="mean_relevance_score"):
            ChunkExperimentResult(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=256,
                mean_relevance_score=-0.1,
                mean_chunks_per_doc=2.0,
            )

    def test_zero_mean_chunks_per_doc_raises(self):
        with pytest.raises(ValueError, match="mean_chunks_per_doc"):
            ChunkExperimentResult(
                strategy=ChunkingStrategy.FIXED_SIZE,
                chunk_size=256,
                mean_relevance_score=0.5,
                mean_chunks_per_doc=0,
            )

    def test_to_dict(self):
        r = ChunkExperimentResult(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=256,
            mean_relevance_score=0.8,
            mean_chunks_per_doc=4.5,
        )
        d = r.to_dict()
        assert d["strategy"] == "recursive"
        assert d["chunk_size"] == 256


class TestHybridRetrievalConfig:
    def test_defaults(self):
        c = HybridRetrievalConfig()
        assert c.bm25_weight == 0.5
        assert c.vector_weight == 0.5

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="bm25_weight"):
            HybridRetrievalConfig(bm25_weight=0.3, vector_weight=0.3)

    def test_weights_within_tolerance_pass(self):
        c = HybridRetrievalConfig(bm25_weight=0.4995, vector_weight=0.5005)
        assert c.bm25_weight == 0.4995

    def test_top_k_initial_zero_raises(self):
        with pytest.raises(ValueError, match="top_k_initial"):
            HybridRetrievalConfig(top_k_initial=0)

    def test_to_dict(self):
        c = HybridRetrievalConfig(bm25_weight=0.6, vector_weight=0.4, top_k_initial=20)
        d = c.to_dict()
        assert d == {"bm25_weight": 0.6, "vector_weight": 0.4, "top_k_initial": 20}


class TestRRFFuser:
    def test_default_k_constant(self):
        f = RRFFuser()
        assert f.k_constant == 60

    def test_k_constant_zero_raises(self):
        with pytest.raises(ValueError, match="k_constant"):
            RRFFuser(k_constant=0)

    def test_negative_k_constant_raises(self):
        with pytest.raises(ValueError, match="k_constant"):
            RRFFuser(k_constant=-5)

    def test_fuse_both_present(self):
        f = RRFFuser(k_constant=60)
        scores = f.fuse({"a": 0}, {"a": 0})
        expected = 1 / 60 + 1 / 60
        assert scores["a"] == pytest.approx(expected)

    def test_fuse_only_in_bm25(self):
        f = RRFFuser(k_constant=60)
        scores = f.fuse({"a": 0}, {})
        assert scores["a"] == pytest.approx(1 / 60)

    def test_fuse_only_in_vector(self):
        f = RRFFuser(k_constant=60)
        scores = f.fuse({}, {"b": 2})
        assert scores["b"] == pytest.approx(1 / 62)

    def test_fuse_union_of_doc_ids(self):
        f = RRFFuser()
        scores = f.fuse({"a": 0, "c": 1}, {"b": 0})
        assert set(scores.keys()) == {"a", "b", "c"}

    def test_fuse_higher_rank_in_both_scores_higher(self):
        f = RRFFuser(k_constant=10)
        scores = f.fuse({"a": 0, "b": 5}, {"a": 1, "b": 6})
        assert scores["a"] > scores["b"]


class TestRerankerConfig:
    def test_basic(self):
        c = RerankerConfig(model_name="cross-encoder/ms-marco")
        assert c.top_k_rerank == 10

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            RerankerConfig(model_name="")

    def test_top_k_rerank_zero_raises(self):
        with pytest.raises(ValueError, match="top_k_rerank"):
            RerankerConfig(model_name="m", top_k_rerank=0)

    def test_to_dict(self):
        c = RerankerConfig(model_name="m", top_k_rerank=5)
        assert c.to_dict() == {"model_name": "m", "top_k_rerank": 5}


class TestRerankResult:
    def test_basic(self):
        r = RerankResult(doc_id="d1", initial_rank=5, rerank_score=0.9)
        assert r.final_rank == 0

    def test_empty_doc_id_raises(self):
        with pytest.raises(ValueError, match="doc_id"):
            RerankResult(doc_id="", initial_rank=0, rerank_score=0.5)

    def test_negative_initial_rank_raises(self):
        with pytest.raises(ValueError, match="initial_rank"):
            RerankResult(doc_id="d1", initial_rank=-1, rerank_score=0.5)

    def test_negative_final_rank_raises(self):
        with pytest.raises(ValueError, match="final_rank"):
            RerankResult(doc_id="d1", initial_rank=0, rerank_score=0.5, final_rank=-1)

    def test_rank_improved_true(self):
        r = RerankResult(doc_id="d1", initial_rank=5, rerank_score=0.9, final_rank=1)
        assert r.rank_improved() is True

    def test_rank_improved_false(self):
        r = RerankResult(doc_id="d1", initial_rank=1, rerank_score=0.9, final_rank=5)
        assert r.rank_improved() is False

    def test_to_dict(self):
        r = RerankResult(doc_id="d1", initial_rank=5, rerank_score=0.9, final_rank=2)
        d = r.to_dict()
        assert d["doc_id"] == "d1"
        assert d["final_rank"] == 2
