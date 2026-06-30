"""Unit tests for platform/llm/milestone3_gate.py (Day 114b) — MILESTONE 3 GATE."""

import pytest

from llm.milestone3_gate import M3GateCheckResult, Milestone3Gate, RAGProvenanceRecord


def make_record(**overrides) -> RAGProvenanceRecord:
    defaults = dict(
        answer_id="a1",
        retrieved_chunk_ids=["c1", "c2"],
        prompt_version="prompt-v1",
        embedding_model="text-embedding-3-small",
        index_version_id="index-v1",
        llm_version="gpt-4o",
        eval_score=0.85,
        cost_usd=0.01,
        guardrails_active=True,
    )
    defaults.update(overrides)
    return RAGProvenanceRecord(**defaults)


class TestM3GateCheckResult:
    def test_basic(self):
        c = M3GateCheckResult(check_name="x", passed=True)
        assert c.detail == ""

    def test_empty_check_name_raises(self):
        with pytest.raises(ValueError, match="check_name"):
            M3GateCheckResult(check_name="", passed=True)

    def test_to_dict(self):
        c = M3GateCheckResult(check_name="x", passed=False, detail="oops")
        assert c.to_dict() == {"check_name": "x", "passed": False, "detail": "oops"}


class TestRAGProvenanceRecord:
    def test_basic(self):
        r = make_record()
        assert r.is_fully_provenanced() is True

    def test_empty_answer_id_raises(self):
        with pytest.raises(ValueError, match="answer_id"):
            make_record(answer_id="")

    def test_empty_retrieved_chunk_ids_raises(self):
        with pytest.raises(ValueError, match="retrieved_chunk_ids"):
            make_record(retrieved_chunk_ids=[])

    def test_empty_prompt_version_raises(self):
        with pytest.raises(ValueError, match="prompt_version"):
            make_record(prompt_version="")

    def test_empty_embedding_model_raises(self):
        with pytest.raises(ValueError, match="embedding_model"):
            make_record(embedding_model="")

    def test_empty_index_version_id_raises(self):
        with pytest.raises(ValueError, match="index_version_id"):
            make_record(index_version_id="")

    def test_empty_llm_version_raises(self):
        with pytest.raises(ValueError, match="llm_version"):
            make_record(llm_version="")

    def test_eval_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="eval_score"):
            make_record(eval_score=1.5)

    def test_negative_eval_score_raises(self):
        with pytest.raises(ValueError, match="eval_score"):
            make_record(eval_score=-0.1)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_usd"):
            make_record(cost_usd=-1.0)

    def test_zero_cost_is_valid_construction(self):
        r = make_record(cost_usd=0.0)
        assert r.cost_usd == 0.0

    def test_is_fully_provenanced_false_without_guardrails(self):
        r = make_record(guardrails_active=False)
        assert r.is_fully_provenanced() is False

    def test_to_dict(self):
        r = make_record()
        d = r.to_dict()
        assert d["answer_id"] == "a1"
        assert d["is_fully_provenanced"] is True


class TestMilestone3GateConstruction:
    def test_empty_records_raises(self):
        with pytest.raises(ValueError, match="provenance_records"):
            Milestone3Gate(provenance_records=[])

    def test_min_eval_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="min_eval_score"):
            Milestone3Gate(provenance_records=[make_record()], min_eval_score=1.5)

    def test_basic_construction(self):
        gate = Milestone3Gate(provenance_records=[make_record()])
        assert gate.min_eval_score == 0.7


class TestCheckProvenanceComplete:
    def test_passes_when_all_complete(self):
        gate = Milestone3Gate(provenance_records=[make_record(), make_record(answer_id="a2")])
        result = gate.check_provenance_complete()
        assert result.passed is True

    def test_fails_when_guardrails_inactive(self):
        gate = Milestone3Gate(
            provenance_records=[make_record(answer_id="a1", guardrails_active=False)]
        )
        result = gate.check_provenance_complete()
        assert result.passed is False


class TestCheckEvalThreshold:
    def test_passes_above_threshold(self):
        gate = Milestone3Gate(provenance_records=[make_record(eval_score=0.9)], min_eval_score=0.7)
        assert gate.check_eval_threshold().passed is True

    def test_fails_below_threshold(self):
        gate = Milestone3Gate(provenance_records=[make_record(eval_score=0.5)], min_eval_score=0.7)
        assert gate.check_eval_threshold().passed is False

    def test_boundary_equal_passes(self):
        gate = Milestone3Gate(provenance_records=[make_record(eval_score=0.7)], min_eval_score=0.7)
        assert gate.check_eval_threshold().passed is True


class TestCheckGuardrailsActive:
    def test_passes_when_all_active(self):
        gate = Milestone3Gate(provenance_records=[make_record(guardrails_active=True)])
        assert gate.check_guardrails_active().passed is True

    def test_fails_when_any_inactive(self):
        gate = Milestone3Gate(
            provenance_records=[
                make_record(answer_id="a1", guardrails_active=True),
                make_record(answer_id="a2", guardrails_active=False),
            ]
        )
        assert gate.check_guardrails_active().passed is False


class TestCheckCostTracked:
    def test_passes_when_cost_positive(self):
        gate = Milestone3Gate(provenance_records=[make_record(cost_usd=0.05)])
        assert gate.check_cost_tracked().passed is True

    def test_fails_when_cost_zero(self):
        gate = Milestone3Gate(provenance_records=[make_record(cost_usd=0.0)])
        assert gate.check_cost_tracked().passed is False


class TestRunAllChecksAndGate:
    def test_run_all_checks_returns_four(self):
        gate = Milestone3Gate(provenance_records=[make_record()])
        checks = gate.run_all_checks()
        assert len(checks) == 4
        names = {c.check_name for c in checks}
        assert names == {
            "provenance_complete",
            "eval_threshold",
            "guardrails_active",
            "cost_tracked",
        }

    def test_is_gate_passed_true_for_clean_record(self):
        gate = Milestone3Gate(provenance_records=[make_record()])
        assert gate.is_gate_passed() is True

    def test_is_gate_passed_false_if_any_check_fails(self):
        gate = Milestone3Gate(provenance_records=[make_record(cost_usd=0.0)])
        assert gate.is_gate_passed() is False

    def test_is_gate_passed_false_if_eval_below_threshold(self):
        gate = Milestone3Gate(provenance_records=[make_record(eval_score=0.1)], min_eval_score=0.7)
        assert gate.is_gate_passed() is False

    def test_is_gate_passed_false_if_guardrails_inactive(self):
        gate = Milestone3Gate(provenance_records=[make_record(guardrails_active=False)])
        assert gate.is_gate_passed() is False

    def test_summary_structure_passed(self):
        gate = Milestone3Gate(provenance_records=[make_record()])
        s = gate.summary()
        assert s["gateStatus"] == "PASSED"
        assert len(s["checks"]) == 4
        assert s["total_records"] == 1

    def test_summary_structure_failed(self):
        gate = Milestone3Gate(provenance_records=[make_record(cost_usd=0.0)])
        s = gate.summary()
        assert s["gateStatus"] == "FAILED"


class TestDryRun:
    def test_dry_run_returns_passing_gate(self):
        gate = Milestone3Gate.dry_run()
        assert gate.is_gate_passed() is True

    def test_dry_run_has_three_records(self):
        gate = Milestone3Gate.dry_run()
        assert len(gate.provenance_records) == 3

    def test_dry_run_summary(self):
        gate = Milestone3Gate.dry_run()
        s = gate.summary()
        assert s["gateStatus"] == "PASSED"
        assert s["total_records"] == 3
        assert all(c["passed"] for c in s["checks"])
