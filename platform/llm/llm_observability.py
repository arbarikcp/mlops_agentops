"""
llm_observability.py — LLM Observability: OTel GenAI semantic conventions,
Langfuse/Phoenix/LangSmith comparison (Day 106)

Covers OpenTelemetry GenAI semantic conventions (gen_ai.* span attributes
extending canonical distributed traces) and a structured comparison of
observability platforms (Langfuse, Phoenix, LangSmith).
No external SDK imports — pure Python dataclasses (no opentelemetry import;
this module models the OTel GenAI conventions, it doesn't depend on the SDK).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class GenAISpanKind(str, Enum):
    """Kinds of spans in a GenAI request trace."""

    LLM_CALL = "llm_call"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    TOOL_CALL = "tool_call"
    AGENT_STEP = "agent_step"


@dataclass
class GenAISpanAttributes:
    """OTel GenAI semantic-convention attributes for a single span."""

    gen_ai_system: str
    gen_ai_request_model: str
    gen_ai_usage_prompt_tokens: int = 0
    gen_ai_usage_completion_tokens: int = 0
    gen_ai_response_finish_reason: str = ""

    def __post_init__(self) -> None:
        if not self.gen_ai_system:
            raise ValueError("gen_ai_system must be non-empty")
        if not self.gen_ai_request_model:
            raise ValueError("gen_ai_request_model must be non-empty")
        if self.gen_ai_usage_prompt_tokens < 0:
            raise ValueError("gen_ai_usage_prompt_tokens must be >= 0")
        if self.gen_ai_usage_completion_tokens < 0:
            raise ValueError("gen_ai_usage_completion_tokens must be >= 0")

    def to_otel_attrs(self) -> dict:
        return {
            "gen_ai.system": self.gen_ai_system,
            "gen_ai.request.model": self.gen_ai_request_model,
            "gen_ai.usage.prompt_tokens": self.gen_ai_usage_prompt_tokens,
            "gen_ai.usage.completion_tokens": self.gen_ai_usage_completion_tokens,
            "gen_ai.response.finish_reason": self.gen_ai_response_finish_reason,
        }

    def to_dict(self) -> dict:
        return {
            "gen_ai_system": self.gen_ai_system,
            "gen_ai_request_model": self.gen_ai_request_model,
            "gen_ai_usage_prompt_tokens": self.gen_ai_usage_prompt_tokens,
            "gen_ai_usage_completion_tokens": self.gen_ai_usage_completion_tokens,
            "gen_ai_response_finish_reason": self.gen_ai_response_finish_reason,
        }


@dataclass
class GenAISpan:
    """A single span within a GenAI request trace."""

    span_id: str
    name: str
    kind: GenAISpanKind
    attributes: GenAISpanAttributes
    parent_span_id: str = ""
    duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.span_id:
            raise ValueError("span_id must be non-empty")
        if not self.name:
            raise ValueError("name must be non-empty")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must be >= 0")

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind.value,
            "attributes": self.attributes.to_dict(),
            "parent_span_id": self.parent_span_id,
            "duration_ms": self.duration_ms,
        }


@dataclass
class GenAITrace:
    """A full request trace composed of nested GenAI spans."""

    trace_id: str
    spans: list[GenAISpan] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trace_id:
            raise ValueError("trace_id must be non-empty")

    def add_span(self, span: GenAISpan) -> None:
        self.spans.append(span)

    def total_duration_ms(self) -> float:
        return sum(
            s.duration_ms for s in self.spans if not s.parent_span_id
        )

    def total_cost_tokens(self) -> int:
        return sum(
            s.attributes.gen_ai_usage_prompt_tokens
            + s.attributes.gen_ai_usage_completion_tokens
            for s in self.spans
            if s.kind == GenAISpanKind.LLM_CALL
        )

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "spans": [s.to_dict() for s in self.spans],
            "total_duration_ms": self.total_duration_ms(),
            "total_cost_tokens": self.total_cost_tokens(),
        }


class ObservabilityPlatform(str, Enum):
    """LLM observability platform options."""

    LANGFUSE = "langfuse"
    PHOENIX = "phoenix"
    LANGSMITH = "langsmith"


class ObservabilityComparison:
    """Static comparison of LLM observability platforms."""

    @staticmethod
    def compare() -> dict[str, dict]:
        return {
            ObservabilityPlatform.LANGFUSE.value: {
                "self_hosted": True,
                "cost_tracking": "per-trace token + cost breakdown, open-source",
                "eval_integration": "built-in eval scores + custom scorers",
                "otel_native": True,
            },
            ObservabilityPlatform.PHOENIX.value: {
                "self_hosted": True,
                "cost_tracking": "per-span token usage, embedding-aware analysis",
                "eval_integration": "Arize eval library, embedding drift detection",
                "otel_native": True,
            },
            ObservabilityPlatform.LANGSMITH.value: {
                "self_hosted": False,
                "cost_tracking": "per-run cost in managed dashboard",
                "eval_integration": "LangChain-native eval datasets + annotation queues",
                "otel_native": False,
            },
        }
