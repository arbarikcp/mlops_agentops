"""
retrieval — Day 110: Chunking Experiments + Hybrid Retrieval (BM25 + Vector) + Reranking

Hybrid retrieval fuses sparse (BM25, exact keyword match) and dense (vector,
semantic similarity) search results via Reciprocal Rank Fusion (RRF).
Reranking applies a more expensive cross-encoder-style scorer only to the
top-K candidates from initial retrieval, trading cost for precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from llm.index_pipeline import ChunkingStrategy

__all__ = [
    "ChunkingStrategy",
    "RetrievalMethod",
    "ChunkExperimentResult",
    "HybridRetrievalConfig",
    "RRFFuser",
    "RerankerConfig",
    "RerankResult",
]


class RetrievalMethod(str, Enum):
    """Retrieval strategy used to surface candidate chunks."""

    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


@dataclass
class ChunkExperimentResult:
    """Result of evaluating one chunking strategy/size combination."""

    strategy: ChunkingStrategy
    chunk_size: int
    mean_relevance_score: float
    mean_chunks_per_doc: float

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not (0 <= self.mean_relevance_score <= 1):
            raise ValueError("mean_relevance_score must be in [0, 1]")
        if self.mean_chunks_per_doc <= 0:
            raise ValueError("mean_chunks_per_doc must be > 0")

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "chunk_size": self.chunk_size,
            "mean_relevance_score": self.mean_relevance_score,
            "mean_chunks_per_doc": self.mean_chunks_per_doc,
        }


@dataclass
class HybridRetrievalConfig:
    """Weights for combining BM25 and vector retrieval scores."""

    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    top_k_initial: int = 50

    def __post_init__(self) -> None:
        if abs((self.bm25_weight + self.vector_weight) - 1.0) > 0.001:
            raise ValueError("bm25_weight + vector_weight must approximately equal 1.0")
        if self.top_k_initial < 1:
            raise ValueError("top_k_initial must be >= 1")

    def to_dict(self) -> dict:
        return {
            "bm25_weight": self.bm25_weight,
            "vector_weight": self.vector_weight,
            "top_k_initial": self.top_k_initial,
        }


@dataclass
class RRFFuser:
    """Reciprocal Rank Fusion: combines two ranked lists into one score map."""

    k_constant: int = 60

    def __post_init__(self) -> None:
        if self.k_constant <= 0:
            raise ValueError("k_constant must be > 0")

    def fuse(self, bm25_ranks: dict[str, int], vector_ranks: dict[str, int]) -> dict[str, float]:
        doc_ids = set(bm25_ranks) | set(vector_ranks)
        scores: dict[str, float] = {}
        for doc_id in doc_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (self.k_constant + bm25_ranks[doc_id])
            if doc_id in vector_ranks:
                score += 1.0 / (self.k_constant + vector_ranks[doc_id])
            scores[doc_id] = score
        return scores


@dataclass
class RerankerConfig:
    """Configuration for the cross-encoder reranking stage."""

    model_name: str
    top_k_rerank: int = 10

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.top_k_rerank < 1:
            raise ValueError("top_k_rerank must be >= 1")

    def to_dict(self) -> dict:
        return {"model_name": self.model_name, "top_k_rerank": self.top_k_rerank}


@dataclass
class RerankResult:
    """Result of reranking a single candidate document."""

    doc_id: str
    initial_rank: int
    rerank_score: float
    final_rank: int = 0

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        if self.initial_rank < 0:
            raise ValueError("initial_rank must be >= 0")
        if self.final_rank < 0:
            raise ValueError("final_rank must be >= 0")

    def rank_improved(self) -> bool:
        return self.final_rank < self.initial_rank

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "initial_rank": self.initial_rank,
            "rerank_score": self.rerank_score,
            "final_rank": self.final_rank,
        }
