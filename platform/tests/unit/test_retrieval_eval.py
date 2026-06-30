"""Unit tests for platform/llm/retrieval_eval.py (Day 113)."""

import pytest

from llm.retrieval_eval import (
    GoldenQuery,
    GoldenQuerySet,
    RetrievalEvalResult,
    RetrievalFailureReport,
    RetrievalFailureType,
    SyntheticQuerySpec,
)


class TestRetrievalFailureType:
    def test_members(self):
        assert RetrievalFailureType.NO_RELEVANT_DOC.value == "no_relevant_doc"
        assert RetrievalFailureType.RANKING_FAILURE.value == "ranking_failure"
        assert RetrievalFailureType.CHUNK_GRANULARITY.value == "chunk_granularity"
        assert RetrievalFailureType.QUERY_AMBIGUITY.value == "query_ambiguity"


class TestGoldenQuery:
    def test_basic(self):
        q = GoldenQuery(query="what is the refund policy?", expected_doc_ids=["d1"])
        assert q.category == "general"

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            GoldenQuery(query="", expected_doc_ids=["d1"])

    def test_empty_expected_doc_ids_raises(self):
        with pytest.raises(ValueError, match="expected_doc_ids"):
            GoldenQuery(query="q", expected_doc_ids=[])

    def test_to_dict(self):
        q = GoldenQuery(query="q", expected_doc_ids=["d1", "d2"], category="policy")
        d = q.to_dict()
        assert d == {"query": "q", "expected_doc_ids": ["d1", "d2"], "category": "policy"}


class TestGoldenQuerySet:
    def test_basic(self):
        qs = GoldenQuerySet(name="faq", queries=[GoldenQuery(query="q", expected_doc_ids=["d1"])])
        assert qs.size() == 1

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            GoldenQuerySet(name="", queries=[GoldenQuery(query="q", expected_doc_ids=["d1"])])

    def test_empty_queries_raises(self):
        with pytest.raises(ValueError, match="queries"):
            GoldenQuerySet(name="faq", queries=[])

    def test_to_dict(self):
        qs = GoldenQuerySet(name="faq", queries=[GoldenQuery(query="q", expected_doc_ids=["d1"])])
        d = qs.to_dict()
        assert d["name"] == "faq"
        assert len(d["queries"]) == 1


class TestRetrievalEvalResult:
    def test_recall_perfect(self):
        gq = GoldenQuery(query="q", expected_doc_ids=["d1", "d2"])
        r = RetrievalEvalResult(query=gq, retrieved_doc_ids=["d1", "d2", "d3"], top_k=10)
        assert r.recall_at_k() == 1.0
        assert r.classify_failure() is None

    def test_recall_partial_is_ranking_failure(self):
        gq = GoldenQuery(query="q", expected_doc_ids=["d1", "d2"])
        r = RetrievalEvalResult(query=gq, retrieved_doc_ids=["d1", "d3"], top_k=10)
        assert r.recall_at_k() == 0.5
        assert r.classify_failure() == RetrievalFailureType.RANKING_FAILURE

    def test_no_overlap_is_no_relevant_doc(self):
        gq = GoldenQuery(query="q", expected_doc_ids=["d1", "d2"])
        r = RetrievalEvalResult(query=gq, retrieved_doc_ids=["d3", "d4"], top_k=10)
        assert r.recall_at_k() == 0.0
        assert r.classify_failure() == RetrievalFailureType.NO_RELEVANT_DOC

    def test_top_k_truncation_affects_recall(self):
        gq = GoldenQuery(query="q", expected_doc_ids=["d1"])
        r = RetrievalEvalResult(query=gq, retrieved_doc_ids=["d2", "d3", "d1"], top_k=2)
        assert r.recall_at_k() == 0.0
        assert r.classify_failure() == RetrievalFailureType.NO_RELEVANT_DOC

    def test_top_k_zero_raises(self):
        gq = GoldenQuery(query="q", expected_doc_ids=["d1"])
        with pytest.raises(ValueError, match="top_k"):
            RetrievalEvalResult(query=gq, retrieved_doc_ids=["d1"], top_k=0)

    def test_to_dict(self):
        gq = GoldenQuery(query="q", expected_doc_ids=["d1"])
        r = RetrievalEvalResult(query=gq, retrieved_doc_ids=["d1"], top_k=10)
        d = r.to_dict()
        assert d["recall_at_k"] == 1.0
        assert d["failure_type"] is None

    def test_to_dict_with_failure(self):
        gq = GoldenQuery(query="q", expected_doc_ids=["d1"])
        r = RetrievalEvalResult(query=gq, retrieved_doc_ids=["d9"], top_k=10)
        d = r.to_dict()
        assert d["failure_type"] == "no_relevant_doc"


class TestSyntheticQuerySpec:
    def test_basic(self):
        s = SyntheticQuerySpec(source_chunk_id="c1", generator_model="gpt-4")
        assert s.num_queries_per_chunk == 3

    def test_empty_source_chunk_id_raises(self):
        with pytest.raises(ValueError, match="source_chunk_id"):
            SyntheticQuerySpec(source_chunk_id="", generator_model="gpt-4")

    def test_empty_generator_model_raises(self):
        with pytest.raises(ValueError, match="generator_model"):
            SyntheticQuerySpec(source_chunk_id="c1", generator_model="")

    def test_num_queries_zero_raises(self):
        with pytest.raises(ValueError, match="num_queries_per_chunk"):
            SyntheticQuerySpec(source_chunk_id="c1", generator_model="gpt-4", num_queries_per_chunk=0)

    def test_to_dict(self):
        s = SyntheticQuerySpec(source_chunk_id="c1", generator_model="gpt-4", num_queries_per_chunk=5)
        assert s.to_dict() == {
            "source_chunk_id": "c1",
            "generator_model": "gpt-4",
            "num_queries_per_chunk": 5,
        }


class TestRetrievalFailureReport:
    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="results"):
            RetrievalFailureReport(results=[])

    def test_mean_recall_at_k(self):
        gq1 = GoldenQuery(query="q1", expected_doc_ids=["d1"])
        gq2 = GoldenQuery(query="q2", expected_doc_ids=["d2"])
        r1 = RetrievalEvalResult(query=gq1, retrieved_doc_ids=["d1"], top_k=10)
        r2 = RetrievalEvalResult(query=gq2, retrieved_doc_ids=["d9"], top_k=10)
        report = RetrievalFailureReport(results=[r1, r2])
        assert report.mean_recall_at_k() == 0.5

    def test_failure_breakdown_skips_passing(self):
        gq1 = GoldenQuery(query="q1", expected_doc_ids=["d1"])
        gq2 = GoldenQuery(query="q2", expected_doc_ids=["d2"])
        r1 = RetrievalEvalResult(query=gq1, retrieved_doc_ids=["d1"], top_k=10)
        r2 = RetrievalEvalResult(query=gq2, retrieved_doc_ids=["d9"], top_k=10)
        report = RetrievalFailureReport(results=[r1, r2])
        breakdown = report.failure_breakdown()
        assert breakdown == {"no_relevant_doc": 1}

    def test_failure_breakdown_counts_multiple(self):
        gq1 = GoldenQuery(query="q1", expected_doc_ids=["d1", "d2"])
        gq2 = GoldenQuery(query="q2", expected_doc_ids=["d3"])
        gq3 = GoldenQuery(query="q3", expected_doc_ids=["d4"])
        r1 = RetrievalEvalResult(query=gq1, retrieved_doc_ids=["d1"], top_k=10)  # ranking failure
        r2 = RetrievalEvalResult(query=gq2, retrieved_doc_ids=["d9"], top_k=10)  # no relevant doc
        r3 = RetrievalEvalResult(query=gq3, retrieved_doc_ids=["d4"], top_k=10)  # passes
        report = RetrievalFailureReport(results=[r1, r2, r3])
        breakdown = report.failure_breakdown()
        assert breakdown == {"ranking_failure": 1, "no_relevant_doc": 1}

    def test_to_dict(self):
        gq = GoldenQuery(query="q1", expected_doc_ids=["d1"])
        r = RetrievalEvalResult(query=gq, retrieved_doc_ids=["d1"], top_k=10)
        report = RetrievalFailureReport(results=[r])
        d = report.to_dict()
        assert d["num_results"] == 1
        assert d["mean_recall_at_k"] == 1.0
