"""Unit tests for platform/llm/ragas_eval.py (Day 104)."""

import pytest
from llm.ragas_eval import (
    AnswerCorrectnessScore,
    ContextRelevanceScore,
    FaithfulnessScore,
    RAGEvalExample,
    RAGASReport,
)


class TestRAGEvalExample:
    def test_basic(self):
        e = RAGEvalExample(question="q", answer="a", contexts=["c1"])
        assert e.ground_truth == ""

    def test_empty_question_raises(self):
        with pytest.raises(ValueError, match="question"):
            RAGEvalExample(question="", answer="a", contexts=["c"])

    def test_empty_answer_raises(self):
        with pytest.raises(ValueError, match="answer"):
            RAGEvalExample(question="q", answer="", contexts=["c"])

    def test_empty_contexts_raises(self):
        with pytest.raises(ValueError, match="contexts"):
            RAGEvalExample(question="q", answer="a", contexts=[])

    def test_to_dict(self):
        e = RAGEvalExample(question="q", answer="a", contexts=["c"])
        assert e.to_dict()["question"] == "q"


class TestFaithfulnessScore:
    def test_score_calc(self):
        f = FaithfulnessScore(example_question="q", num_claims=4, num_supported_claims=3)
        assert f.score() == 0.75

    def test_zero_claims_returns_one(self):
        f = FaithfulnessScore(example_question="q", num_claims=0, num_supported_claims=0)
        assert f.score() == 1.0

    def test_negative_claims_raises(self):
        with pytest.raises(ValueError, match="num_claims"):
            FaithfulnessScore(example_question="q", num_claims=-1, num_supported_claims=0)

    def test_supported_exceeds_claims_raises(self):
        with pytest.raises(ValueError, match="num_supported_claims"):
            FaithfulnessScore(example_question="q", num_claims=2, num_supported_claims=5)

    def test_to_dict(self):
        f = FaithfulnessScore(example_question="q", num_claims=2, num_supported_claims=1)
        assert f.to_dict()["score"] == 0.5


class TestContextRelevanceScore:
    def test_score_calc(self):
        c = ContextRelevanceScore(example_question="q", num_chunks=5, num_relevant_chunks=2)
        assert c.score() == 0.4

    def test_zero_chunks_returns_one(self):
        c = ContextRelevanceScore(example_question="q", num_chunks=0, num_relevant_chunks=0)
        assert c.score() == 1.0

    def test_relevant_exceeds_chunks_raises(self):
        with pytest.raises(ValueError):
            ContextRelevanceScore(example_question="q", num_chunks=2, num_relevant_chunks=3)


class TestAnswerCorrectnessScore:
    def test_score_calc(self):
        a = AnswerCorrectnessScore(
            example_question="q", semantic_similarity=0.8, factual_overlap=0.6, weight_semantic=0.5
        )
        assert a.score() == pytest.approx(0.7)

    def test_invalid_semantic_similarity_raises(self):
        with pytest.raises(ValueError, match="semantic_similarity"):
            AnswerCorrectnessScore(example_question="q", semantic_similarity=1.5, factual_overlap=0.5)

    def test_invalid_factual_overlap_raises(self):
        with pytest.raises(ValueError, match="factual_overlap"):
            AnswerCorrectnessScore(example_question="q", semantic_similarity=0.5, factual_overlap=-0.1)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="weight_semantic"):
            AnswerCorrectnessScore(
                example_question="q", semantic_similarity=0.5, factual_overlap=0.5, weight_semantic=1.5
            )


class TestRAGASReport:
    def _make_report(self):
        f_scores = [FaithfulnessScore(example_question="q1", num_claims=4, num_supported_claims=2)]
        c_scores = [ContextRelevanceScore(example_question="q1", num_chunks=4, num_relevant_chunks=1)]
        a_scores = [AnswerCorrectnessScore(example_question="q1", semantic_similarity=0.4, factual_overlap=0.3)]
        return RAGASReport(
            dataset_name="rag-golden",
            faithfulness_scores=f_scores,
            context_scores=c_scores,
            correctness_scores=a_scores,
        )

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="dataset_name"):
            RAGASReport(dataset_name="", faithfulness_scores=[], context_scores=[], correctness_scores=[])

    def test_mean_faithfulness(self):
        report = self._make_report()
        assert report.mean_faithfulness() == 0.5

    def test_overall_score(self):
        report = self._make_report()
        expected = (report.mean_faithfulness() + report.mean_context_relevance() + report.mean_correctness()) / 3
        assert report.overall_score() == pytest.approx(expected)

    def test_failure_taxonomy(self):
        report = self._make_report()
        tax = report.failure_taxonomy()
        assert tax["hallucination"] == 1
        assert tax["poor_retrieval"] == 1
        assert tax["wrong_answer"] == 1

    def test_failure_taxonomy_no_failures(self):
        f_scores = [FaithfulnessScore(example_question="q1", num_claims=4, num_supported_claims=4)]
        c_scores = [ContextRelevanceScore(example_question="q1", num_chunks=4, num_relevant_chunks=4)]
        a_scores = [AnswerCorrectnessScore(example_question="q1", semantic_similarity=0.9, factual_overlap=0.9)]
        report = RAGASReport(
            dataset_name="x", faithfulness_scores=f_scores, context_scores=c_scores, correctness_scores=a_scores
        )
        tax = report.failure_taxonomy()
        assert tax == {"hallucination": 0, "poor_retrieval": 0, "wrong_answer": 0}

    def test_empty_scores_means_zero(self):
        report = RAGASReport(dataset_name="x", faithfulness_scores=[], context_scores=[], correctness_scores=[])
        assert report.mean_faithfulness() == 0.0
        assert report.mean_context_relevance() == 0.0
        assert report.mean_correctness() == 0.0

    def test_to_dict(self):
        report = self._make_report()
        d = report.to_dict()
        assert "failure_taxonomy" in d
