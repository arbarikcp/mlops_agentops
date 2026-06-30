"""Unit tests for platform/llm/llm_observability.py (Day 106)."""

import pytest
from llm.llm_observability import (
    GenAISpan,
    GenAISpanAttributes,
    GenAISpanKind,
    GenAITrace,
    ObservabilityComparison,
    ObservabilityPlatform,
)


class TestGenAISpanAttributes:
    def test_basic(self):
        a = GenAISpanAttributes(gen_ai_system="openai", gen_ai_request_model="gpt-4")
        assert a.gen_ai_usage_prompt_tokens == 0

    def test_empty_system_raises(self):
        with pytest.raises(ValueError, match="gen_ai_system"):
            GenAISpanAttributes(gen_ai_system="", gen_ai_request_model="gpt-4")

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="gen_ai_request_model"):
            GenAISpanAttributes(gen_ai_system="openai", gen_ai_request_model="")

    def test_negative_prompt_tokens_raises(self):
        with pytest.raises(ValueError, match="gen_ai_usage_prompt_tokens"):
            GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y", gen_ai_usage_prompt_tokens=-1)

    def test_negative_completion_tokens_raises(self):
        with pytest.raises(ValueError, match="gen_ai_usage_completion_tokens"):
            GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y", gen_ai_usage_completion_tokens=-1)

    def test_to_otel_attrs_dotted_keys(self):
        a = GenAISpanAttributes(gen_ai_system="openai", gen_ai_request_model="gpt-4")
        attrs = a.to_otel_attrs()
        assert "gen_ai.system" in attrs
        assert "gen_ai.request.model" in attrs


class TestGenAISpan:
    def test_basic(self):
        attrs = GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y")
        s = GenAISpan(span_id="s1", name="llm-call", kind=GenAISpanKind.LLM_CALL, attributes=attrs)
        assert s.duration_ms == 0.0

    def test_empty_span_id_raises(self):
        attrs = GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y")
        with pytest.raises(ValueError, match="span_id"):
            GenAISpan(span_id="", name="x", kind=GenAISpanKind.LLM_CALL, attributes=attrs)

    def test_negative_duration_raises(self):
        attrs = GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y")
        with pytest.raises(ValueError, match="duration_ms"):
            GenAISpan(span_id="s1", name="x", kind=GenAISpanKind.LLM_CALL, attributes=attrs, duration_ms=-1)

    def test_to_dict(self):
        attrs = GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y")
        s = GenAISpan(span_id="s1", name="x", kind=GenAISpanKind.RETRIEVAL, attributes=attrs)
        assert s.to_dict()["kind"] == "retrieval"


class TestGenAITrace:
    def test_empty_trace_id_raises(self):
        with pytest.raises(ValueError, match="trace_id"):
            GenAITrace(trace_id="")

    def test_add_span(self):
        t = GenAITrace(trace_id="t1")
        attrs = GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y")
        s = GenAISpan(span_id="s1", name="x", kind=GenAISpanKind.LLM_CALL, attributes=attrs)
        t.add_span(s)
        assert len(t.spans) == 1

    def test_total_duration_only_root_spans(self):
        t = GenAITrace(trace_id="t1")
        attrs = GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y")
        root = GenAISpan(span_id="s1", name="root", kind=GenAISpanKind.AGENT_STEP, attributes=attrs, duration_ms=100)
        child = GenAISpan(
            span_id="s2", name="child", kind=GenAISpanKind.LLM_CALL, attributes=attrs,
            parent_span_id="s1", duration_ms=50,
        )
        t.add_span(root)
        t.add_span(child)
        assert t.total_duration_ms() == 100

    def test_total_cost_tokens_only_llm_calls(self):
        t = GenAITrace(trace_id="t1")
        llm_attrs = GenAISpanAttributes(
            gen_ai_system="x", gen_ai_request_model="y",
            gen_ai_usage_prompt_tokens=100, gen_ai_usage_completion_tokens=50,
        )
        retrieval_attrs = GenAISpanAttributes(gen_ai_system="x", gen_ai_request_model="y")
        t.add_span(GenAISpan(span_id="s1", name="llm", kind=GenAISpanKind.LLM_CALL, attributes=llm_attrs))
        t.add_span(GenAISpan(span_id="s2", name="ret", kind=GenAISpanKind.RETRIEVAL, attributes=retrieval_attrs))
        assert t.total_cost_tokens() == 150

    def test_to_dict(self):
        t = GenAITrace(trace_id="t1")
        d = t.to_dict()
        assert d["trace_id"] == "t1"


class TestObservabilityComparison:
    def test_compare_has_all_platforms(self):
        cmp = ObservabilityComparison.compare()
        for p in [ObservabilityPlatform.LANGFUSE, ObservabilityPlatform.PHOENIX, ObservabilityPlatform.LANGSMITH]:
            assert p.value in cmp

    def test_compare_fields(self):
        cmp = ObservabilityComparison.compare()
        for platform_info in cmp.values():
            assert "self_hosted" in platform_info
            assert "cost_tracking" in platform_info
            assert "eval_integration" in platform_info
            assert "otel_native" in platform_info

    def test_langsmith_not_self_hosted(self):
        cmp = ObservabilityComparison.compare()
        assert cmp["langsmith"]["self_hosted"] is False
