"""
retrieval_eval — Day 113: Retrieval Failure Taxonomy + Golden Query Set +
Synthetic Query Generation

"RAG isn't working" is not actionable. This module classifies retrieval
failures into a taxonomy (coverage gap vs. ranking gap vs. chunking gap vs.
ambiguous query) and provides the golden-query-set machinery needed to
measure retrieval quality at scale, including synthetic query generation to
grow eval coverage beyond hand-written queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "RetrievalFailureType",
    "GoldenQuery",
    "GoldenQuerySet",
    "RetrievalEvalResult",
    "SyntheticQuerySpec",
    "RetrievalFailureReport",
]


class RetrievalFailureType(str, Enum):
    """Taxonomy of retrieval failure modes."""

    NO_RELEVANT_DOC = "no_relevant_doc"
    RANKING_FAILURE = "ranking_failure"
    CHUNK_GRANULARITY = "chunk_granularity"
    QUERY_AMBIGUITY = "query_ambiguity"


@dataclass
class GoldenQuery:
    """A single curated (query, expected_doc_ids) eval example."""

    query: str
    expected_doc_ids: list[str]
    category: str = "general"

    def __post_init__(self) -> None:
        if not self.query:
            raise ValueError("query must be non-empty")
        if not self.expected_doc_ids:
            raise ValueError("expected_doc_ids must be non-empty")

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "expected_doc_ids": list(self.expected_doc_ids),
            "category": self.category,
        }


@dataclass
class GoldenQuerySet:
    """A named collection of golden queries used as retrieval ground truth."""

    name: str
    queries: list[GoldenQuery]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.queries:
            raise ValueError("queries must be non-empty")

    def size(self) -> int:
        return len(self.queries)

    def to_dict(self) -> dict:
        return {"name": self.name, "queries": [q.to_dict() for q in self.queries]}


@dataclass
class RetrievalEvalResult:
    """The result of running one golden query against the retrieval system."""

    query: GoldenQuery
    retrieved_doc_ids: list[str]
    top_k: int = 10

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

    def recall_at_k(self) -> float:
        top_k_retrieved = set(self.retrieved_doc_ids[: self.top_k])
        expected = set(self.query.expected_doc_ids)
        if not expected:
            return 0.0
        hits = len(expected & top_k_retrieved)
        return hits / len(expected)

    def classify_failure(self) -> RetrievalFailureType | None:
        recall = self.recall_at_k()
        if recall == 1.0:
            return None
        top_k_retrieved = set(self.retrieved_doc_ids[: self.top_k])
        expected = set(self.query.expected_doc_ids)
        overlap = expected & top_k_retrieved
        if not overlap:
            return RetrievalFailureType.NO_RELEVANT_DOC
        return RetrievalFailureType.RANKING_FAILURE

    def to_dict(self) -> dict:
        failure = self.classify_failure()
        return {
            "query": self.query.to_dict(),
            "retrieved_doc_ids": list(self.retrieved_doc_ids),
            "top_k": self.top_k,
            "recall_at_k": self.recall_at_k(),
            "failure_type": failure.value if failure else None,
        }


@dataclass
class SyntheticQuerySpec:
    """Spec for generating synthetic queries from a source document chunk."""

    source_chunk_id: str
    generator_model: str
    num_queries_per_chunk: int = 3

    def __post_init__(self) -> None:
        if not self.source_chunk_id:
            raise ValueError("source_chunk_id must be non-empty")
        if not self.generator_model:
            raise ValueError("generator_model must be non-empty")
        if self.num_queries_per_chunk < 1:
            raise ValueError("num_queries_per_chunk must be >= 1")

    def to_dict(self) -> dict:
        return {
            "source_chunk_id": self.source_chunk_id,
            "generator_model": self.generator_model,
            "num_queries_per_chunk": self.num_queries_per_chunk,
        }


@dataclass
class RetrievalFailureReport:
    """Aggregates retrieval eval results into a failure breakdown."""

    results: list[RetrievalEvalResult]

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("results must be non-empty")

    def mean_recall_at_k(self) -> float:
        return sum(r.recall_at_k() for r in self.results) / len(self.results)

    def failure_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for r in self.results:
            failure = r.classify_failure()
            if failure is None:
                continue
            breakdown[failure.value] = breakdown.get(failure.value, 0) + 1
        return breakdown

    def to_dict(self) -> dict:
        return {
            "mean_recall_at_k": self.mean_recall_at_k(),
            "failure_breakdown": self.failure_breakdown(),
            "num_results": len(self.results),
        }
