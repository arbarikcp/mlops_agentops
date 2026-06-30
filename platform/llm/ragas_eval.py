"""
ragas_eval.py — LLM Eval II: RAGAS metrics (Day 104)

Covers RAGAS-style metrics specifically for RAG pipelines: faithfulness
(answer grounded in retrieved context, no hallucination), context
relevance (retrieved chunks actually relevant to the question), and
answer correctness (semantic similarity + factual overlap vs ground
truth). Separates retrieval failures from generation failures.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RAGEvalExample:
    """A single RAG eval example: question, generated answer, retrieved contexts."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""

    def __post_init__(self) -> None:
        if not self.question:
            raise ValueError("question must be non-empty")
        if not self.answer:
            raise ValueError("answer must be non-empty")
        if not self.contexts:
            raise ValueError("contexts must be non-empty")

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


@dataclass
class FaithfulnessScore:
    """Fraction of answer claims that are supported by retrieved context."""

    example_question: str
    num_claims: int
    num_supported_claims: int

    def __post_init__(self) -> None:
        if self.num_claims < 0:
            raise ValueError("num_claims must be >= 0")
        if not (0 <= self.num_supported_claims <= self.num_claims):
            raise ValueError(
                "num_supported_claims must be in [0, num_claims]"
            )

    def score(self) -> float:
        if self.num_claims == 0:
            return 1.0
        return self.num_supported_claims / self.num_claims

    def to_dict(self) -> dict:
        return {
            "example_question": self.example_question,
            "num_claims": self.num_claims,
            "num_supported_claims": self.num_supported_claims,
            "score": self.score(),
        }


@dataclass
class ContextRelevanceScore:
    """Fraction of retrieved chunks that are actually relevant to the question."""

    example_question: str
    num_chunks: int
    num_relevant_chunks: int

    def __post_init__(self) -> None:
        if self.num_chunks < 0:
            raise ValueError("num_chunks must be >= 0")
        if not (0 <= self.num_relevant_chunks <= self.num_chunks):
            raise ValueError(
                "num_relevant_chunks must be in [0, num_chunks]"
            )

    def score(self) -> float:
        if self.num_chunks == 0:
            return 1.0
        return self.num_relevant_chunks / self.num_chunks

    def to_dict(self) -> dict:
        return {
            "example_question": self.example_question,
            "num_chunks": self.num_chunks,
            "num_relevant_chunks": self.num_relevant_chunks,
            "score": self.score(),
        }


@dataclass
class AnswerCorrectnessScore:
    """Weighted combination of semantic similarity and factual overlap."""

    example_question: str
    semantic_similarity: float
    factual_overlap: float
    weight_semantic: float = 0.5

    def __post_init__(self) -> None:
        if not (0 <= self.semantic_similarity <= 1):
            raise ValueError("semantic_similarity must be in [0, 1]")
        if not (0 <= self.factual_overlap <= 1):
            raise ValueError("factual_overlap must be in [0, 1]")
        if not (0 <= self.weight_semantic <= 1):
            raise ValueError("weight_semantic must be in [0, 1]")

    def score(self) -> float:
        return (
            self.weight_semantic * self.semantic_similarity
            + (1 - self.weight_semantic) * self.factual_overlap
        )

    def to_dict(self) -> dict:
        return {
            "example_question": self.example_question,
            "semantic_similarity": self.semantic_similarity,
            "factual_overlap": self.factual_overlap,
            "weight_semantic": self.weight_semantic,
            "score": self.score(),
        }


@dataclass
class RAGASReport:
    """Aggregate RAGAS report over a RAG eval dataset."""

    dataset_name: str
    faithfulness_scores: list[FaithfulnessScore]
    context_scores: list[ContextRelevanceScore]
    correctness_scores: list[AnswerCorrectnessScore]

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")

    def mean_faithfulness(self) -> float:
        if not self.faithfulness_scores:
            return 0.0
        return sum(s.score() for s in self.faithfulness_scores) / len(
            self.faithfulness_scores
        )

    def mean_context_relevance(self) -> float:
        if not self.context_scores:
            return 0.0
        return sum(s.score() for s in self.context_scores) / len(
            self.context_scores
        )

    def mean_correctness(self) -> float:
        if not self.correctness_scores:
            return 0.0
        return sum(s.score() for s in self.correctness_scores) / len(
            self.correctness_scores
        )

    def overall_score(self) -> float:
        return (
            self.mean_faithfulness()
            + self.mean_context_relevance()
            + self.mean_correctness()
        ) / 3

    def failure_taxonomy(self) -> dict[str, int]:
        return {
            "hallucination": sum(
                1 for s in self.faithfulness_scores if s.score() < 0.7
            ),
            "poor_retrieval": sum(
                1 for s in self.context_scores if s.score() < 0.5
            ),
            "wrong_answer": sum(
                1 for s in self.correctness_scores if s.score() < 0.6
            ),
        }

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "mean_faithfulness": self.mean_faithfulness(),
            "mean_context_relevance": self.mean_context_relevance(),
            "mean_correctness": self.mean_correctness(),
            "overall_score": self.overall_score(),
            "failure_taxonomy": self.failure_taxonomy(),
        }
