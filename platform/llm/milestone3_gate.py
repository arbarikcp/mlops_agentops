"""
milestone3_gate — Day 114b: MILESTONE 3 GATE

The capstone check for Phase 15 / Milestone 3 (Production RAG / LLMOps).

M3 Gate — you pass when: for any answer you can prove "this came from these
retrieved chunks, using this prompt version, this embedding model, this
index version, this LLM version, and this eval score" — with guardrails
active and cost tracked.

This module is intentionally independent of platform/infra/ — it mirrors the
style of infra/milestone2_gate.py's dry_run() pattern without importing it,
keeping llm/ and infra/ decoupled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "M3GateCheckResult",
    "RAGProvenanceRecord",
    "Milestone3Gate",
]


@dataclass
class M3GateCheckResult:
    """The outcome of a single Milestone 3 gate check."""

    check_name: str
    passed: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.check_name:
            raise ValueError("check_name must be non-empty")

    def to_dict(self) -> dict:
        return {"check_name": self.check_name, "passed": self.passed, "detail": self.detail}


@dataclass
class RAGProvenanceRecord:
    """Full provenance for a single RAG answer.

    Proves: which chunks were retrieved, which prompt/embedding/index/LLM
    versions produced the answer, what the eval score was, what it cost,
    and whether guardrails were active.
    """

    answer_id: str
    retrieved_chunk_ids: list[str]
    prompt_version: str
    embedding_model: str
    index_version_id: str
    llm_version: str
    eval_score: float
    cost_usd: float
    guardrails_active: bool

    def __post_init__(self) -> None:
        if not self.answer_id:
            raise ValueError("answer_id must be non-empty")
        if not self.retrieved_chunk_ids:
            raise ValueError("retrieved_chunk_ids must be non-empty")
        if not self.prompt_version:
            raise ValueError("prompt_version must be non-empty")
        if not self.embedding_model:
            raise ValueError("embedding_model must be non-empty")
        if not self.index_version_id:
            raise ValueError("index_version_id must be non-empty")
        if not self.llm_version:
            raise ValueError("llm_version must be non-empty")
        if not (0 <= self.eval_score <= 1):
            raise ValueError("eval_score must be in [0, 1]")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")

    def is_fully_provenanced(self) -> bool:
        return bool(
            self.answer_id
            and self.retrieved_chunk_ids
            and self.prompt_version
            and self.embedding_model
            and self.index_version_id
            and self.llm_version
            and self.guardrails_active
        )

    def to_dict(self) -> dict:
        return {
            "answer_id": self.answer_id,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "prompt_version": self.prompt_version,
            "embedding_model": self.embedding_model,
            "index_version_id": self.index_version_id,
            "llm_version": self.llm_version,
            "eval_score": self.eval_score,
            "cost_usd": self.cost_usd,
            "guardrails_active": self.guardrails_active,
            "is_fully_provenanced": self.is_fully_provenanced(),
        }


@dataclass
class Milestone3Gate:
    """MILESTONE 3 GATE: validates a batch of RAG provenance records.

    Four checks, all must pass for the gate to pass:
      1. Provenance complete (every required field present)
      2. Eval score above threshold
      3. Guardrails active on every record
      4. Cost actually tracked (not defaulted to zero)
    """

    provenance_records: list[RAGProvenanceRecord]
    min_eval_score: float = 0.7

    def __post_init__(self) -> None:
        if not self.provenance_records:
            raise ValueError("provenance_records must be non-empty")
        if not (0 <= self.min_eval_score <= 1):
            raise ValueError("min_eval_score must be in [0, 1]")

    def check_provenance_complete(self) -> M3GateCheckResult:
        passed = all(r.is_fully_provenanced() for r in self.provenance_records)
        n_incomplete = sum(1 for r in self.provenance_records if not r.is_fully_provenanced())
        return M3GateCheckResult(
            check_name="provenance_complete",
            passed=passed,
            detail="all records fully provenanced" if passed else f"{n_incomplete} record(s) incomplete",
        )

    def check_eval_threshold(self) -> M3GateCheckResult:
        passed = all(r.eval_score >= self.min_eval_score for r in self.provenance_records)
        worst = min(r.eval_score for r in self.provenance_records)
        return M3GateCheckResult(
            check_name="eval_threshold",
            passed=passed,
            detail=f"min_eval_score={self.min_eval_score}, worst_observed={worst}",
        )

    def check_guardrails_active(self) -> M3GateCheckResult:
        passed = all(r.guardrails_active for r in self.provenance_records)
        n_inactive = sum(1 for r in self.provenance_records if not r.guardrails_active)
        return M3GateCheckResult(
            check_name="guardrails_active",
            passed=passed,
            detail="guardrails active on all records" if passed else f"{n_inactive} record(s) without guardrails",
        )

    def check_cost_tracked(self) -> M3GateCheckResult:
        passed = all(r.cost_usd > 0 for r in self.provenance_records)
        n_untracked = sum(1 for r in self.provenance_records if r.cost_usd <= 0)
        return M3GateCheckResult(
            check_name="cost_tracked",
            passed=passed,
            detail="cost tracked on all records" if passed else f"{n_untracked} record(s) with untracked cost",
        )

    def run_all_checks(self) -> list[M3GateCheckResult]:
        return [
            self.check_provenance_complete(),
            self.check_eval_threshold(),
            self.check_guardrails_active(),
            self.check_cost_tracked(),
        ]

    def is_gate_passed(self) -> bool:
        return all(c.passed for c in self.run_all_checks())

    def summary(self) -> dict:
        return {
            "gateStatus": "PASSED" if self.is_gate_passed() else "FAILED",
            "checks": [c.to_dict() for c in self.run_all_checks()],
            "total_records": len(self.provenance_records),
        }

    @staticmethod
    def dry_run() -> "Milestone3Gate":
        """Build a synthetic, fully-passing Milestone3Gate for self-test purposes."""
        records = [
            RAGProvenanceRecord(
                answer_id=f"answer-{i}",
                retrieved_chunk_ids=[f"chunk-{i}-a", f"chunk-{i}-b"],
                prompt_version="prompt-v3",
                embedding_model="text-embedding-3-small",
                index_version_id="index-v7",
                llm_version="gpt-4o-2026-01",
                eval_score=0.85,
                cost_usd=0.0042,
                guardrails_active=True,
            )
            for i in range(1, 4)
        ]
        return Milestone3Gate(provenance_records=records, min_eval_score=0.7)
