"""
rag_guardrails — Day 114a: RAG Guardrails + Eval by Document Slice/Source/Type

A model that has been instruction-tuned to be helpful will often follow
instructions found anywhere in its context window — including instructions
planted inside a retrieved document ("ignore previous instructions...").
This is prompt injection via RAG context, OWASP LLM01. Guardrails must scan
both the retrieved context (before constructing the prompt) and the
generated output (before returning it to the user).

Slice-level eval breaks down quality scores by document source/type so a
global average can't hide a category that silently regressed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "OWASPLLMRisk",
    "InjectionPattern",
    "PromptInjectionScanner",
    "SourceTrustGate",
    "RAGGuardrailReport",
    "SliceEvalKey",
    "SliceEvalResult",
    "SliceEvalReport",
]

_VALID_SEVERITIES = {"low", "medium", "high"}


class OWASPLLMRisk(str, Enum):
    """Relevant OWASP LLM Top 10 risk categories for RAG systems."""

    PROMPT_INJECTION = "LLM01"
    SENSITIVE_INFO_DISCLOSURE = "LLM02"
    SUPPLY_CHAIN_VULN = "LLM05"
    INSECURE_OUTPUT_HANDLING = "LLM08"


@dataclass
class InjectionPattern:
    """A single known prompt-injection pattern to scan for."""

    pattern: str
    risk: OWASPLLMRisk = OWASPLLMRisk.PROMPT_INJECTION
    severity: str = "high"

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("pattern must be non-empty")
        if self.severity not in _VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {_VALID_SEVERITIES}")

    def to_dict(self) -> dict:
        return {"pattern": self.pattern, "risk": self.risk.value, "severity": self.severity}


@dataclass
class PromptInjectionScanner:
    """Scans text for known prompt-injection patterns (case-insensitive)."""

    patterns: list[InjectionPattern] = field(
        default_factory=lambda: [
            InjectionPattern(p)
            for p in [
                "ignore previous instructions",
                "disregard the above",
                "you are now",
                "system prompt:",
            ]
        ]
    )

    def scan(self, text: str) -> list[InjectionPattern]:
        lowered = text.lower()
        return [p for p in self.patterns if p.pattern.lower() in lowered]

    def is_safe(self, text: str) -> bool:
        return len(self.scan(text)) == 0


@dataclass
class SourceTrustGate:
    """Gate that requires a minimum source trust score before using a document."""

    min_trust_score: float = 0.5

    def __post_init__(self) -> None:
        if not (0 <= self.min_trust_score <= 1):
            raise ValueError("min_trust_score must be in [0, 1]")

    def passes(self, source_trust_score: float) -> bool:
        return source_trust_score >= self.min_trust_score

    def to_dict(self) -> dict:
        return {"min_trust_score": self.min_trust_score}


@dataclass
class RAGGuardrailReport:
    """Combined report of context + output guardrail scans for one query."""

    query: str
    context_scan_results: list[InjectionPattern]
    output_scan_results: list[InjectionPattern]
    source_trust_passed: bool

    def __post_init__(self) -> None:
        if not self.query:
            raise ValueError("query must be non-empty")

    def is_safe(self) -> bool:
        return (
            len(self.context_scan_results) == 0
            and len(self.output_scan_results) == 0
            and self.source_trust_passed
        )

    def risks_detected(self) -> list[str]:
        risks = {p.risk.value for p in self.context_scan_results}
        risks |= {p.risk.value for p in self.output_scan_results}
        return list(risks)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "context_scan_results": [p.to_dict() for p in self.context_scan_results],
            "output_scan_results": [p.to_dict() for p in self.output_scan_results],
            "source_trust_passed": self.source_trust_passed,
            "is_safe": self.is_safe(),
            "risks_detected": self.risks_detected(),
        }


@dataclass
class SliceEvalKey:
    """Identifies an eval slice (e.g. doc_source=legal)."""

    slice_field: str
    slice_value: str

    def __post_init__(self) -> None:
        if not self.slice_field:
            raise ValueError("slice_field must be non-empty")
        if not self.slice_value:
            raise ValueError("slice_value must be non-empty")

    def to_dict(self) -> dict:
        return {"slice_field": self.slice_field, "slice_value": self.slice_value}


@dataclass
class SliceEvalResult:
    """Eval result for a single slice."""

    slice_key: SliceEvalKey
    mean_score: float
    num_examples: int

    def __post_init__(self) -> None:
        if not (0 <= self.mean_score <= 1):
            raise ValueError("mean_score must be in [0, 1]")
        if self.num_examples <= 0:
            raise ValueError("num_examples must be > 0")

    def to_dict(self) -> dict:
        return {
            "slice_key": self.slice_key.to_dict(),
            "mean_score": self.mean_score,
            "num_examples": self.num_examples,
        }


@dataclass
class SliceEvalReport:
    """Aggregates per-slice eval results and flags failing slices."""

    results: list[SliceEvalResult]
    global_threshold: float = 0.7

    def __post_init__(self) -> None:
        if not self.results:
            raise ValueError("results must be non-empty")
        if not (0 <= self.global_threshold <= 1):
            raise ValueError("global_threshold must be in [0, 1]")

    def failing_slices(self) -> list[SliceEvalResult]:
        return [r for r in self.results if r.mean_score < self.global_threshold]

    def worst_slice(self) -> SliceEvalResult:
        return min(self.results, key=lambda r: r.mean_score)

    def to_dict(self) -> dict:
        return {
            "results": [r.to_dict() for r in self.results],
            "global_threshold": self.global_threshold,
            "failing_slices": [r.to_dict() for r in self.failing_slices()],
            "worst_slice": self.worst_slice().to_dict(),
        }
