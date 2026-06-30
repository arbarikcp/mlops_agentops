"""Unit tests for platform/llm/prompt_registry.py (Day 102)."""

import pytest
from llm.prompt_registry import (
    PromptABTest,
    PromptRegistry,
    PromptStatus,
    PromptVersion,
)


class TestPromptVersion:
    def test_basic_creation(self):
        pv = PromptVersion(name="greeting", version="v1", template="Hello {name}", variables=["name"])
        assert pv.status == PromptStatus.DRAFT

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            PromptVersion(name="", version="v1", template="x")

    def test_empty_version_raises(self):
        with pytest.raises(ValueError, match="version"):
            PromptVersion(name="x", version="", template="y")

    def test_empty_template_raises(self):
        with pytest.raises(ValueError, match="template"):
            PromptVersion(name="x", version="v1", template="")

    def test_render_success(self):
        pv = PromptVersion(name="g", version="v1", template="Hello {name}", variables=["name"])
        assert pv.render(name="Bhakti") == "Hello Bhakti"

    def test_render_missing_var_raises(self):
        pv = PromptVersion(name="g", version="v1", template="Hello {name}", variables=["name"])
        with pytest.raises(KeyError):
            pv.render()

    def test_to_dict(self):
        pv = PromptVersion(name="g", version="v1", template="hi")
        assert pv.to_dict()["status"] == "draft"


class TestPromptRegistry:
    def test_register_and_get(self):
        reg = PromptRegistry()
        pv = PromptVersion(name="g", version="v1", template="hi")
        reg.register(pv)
        assert reg.get("g", "v1") is pv

    def test_get_missing_raises(self):
        reg = PromptRegistry()
        with pytest.raises(KeyError):
            reg.get("missing", "v1")

    def test_get_production_none(self):
        reg = PromptRegistry()
        reg.register(PromptVersion(name="g", version="v1", template="hi"))
        assert reg.get_production("g") is None

    def test_get_production_returns_highest(self):
        reg = PromptRegistry()
        reg.register(PromptVersion(name="g", version="v1", template="a", status=PromptStatus.PRODUCTION))
        reg.register(PromptVersion(name="g", version="v2", template="b", status=PromptStatus.PRODUCTION))
        prod = reg.get_production("g")
        assert prod.version == "v2"

    def test_promote(self):
        reg = PromptRegistry()
        reg.register(PromptVersion(name="g", version="v1", template="hi"))
        reg.promote("g", "v1", PromptStatus.PRODUCTION)
        assert reg.get("g", "v1").status == PromptStatus.PRODUCTION

    def test_history(self):
        reg = PromptRegistry()
        reg.register(PromptVersion(name="g", version="v1", template="hi"))
        reg.register(PromptVersion(name="g", version="v2", template="bye"))
        assert len(reg.history("g")) == 2

    def test_history_empty_for_unknown(self):
        reg = PromptRegistry()
        assert reg.history("nope") == []


class TestPromptABTest:
    def test_invalid_split_raises(self):
        a = PromptVersion(name="g", version="v1", template="a")
        b = PromptVersion(name="g", version="v2", template="b")
        with pytest.raises(ValueError, match="traffic_split_b"):
            PromptABTest(name="t", variant_a=a, variant_b=b, traffic_split_b=1.5)

    def test_empty_name_raises(self):
        a = PromptVersion(name="g", version="v1", template="a")
        b = PromptVersion(name="g", version="v2", template="b")
        with pytest.raises(ValueError, match="name"):
            PromptABTest(name="", variant_a=a, variant_b=b)

    def test_assign_variant_deterministic(self):
        a = PromptVersion(name="g", version="v1", template="a")
        b = PromptVersion(name="g", version="v2", template="b")
        test = PromptABTest(name="t", variant_a=a, variant_b=b, traffic_split_b=0.5)
        v1 = test.assign_variant("user-123")
        v2 = test.assign_variant("user-123")
        assert v1 is v2

    def test_assign_variant_is_a_or_b(self):
        a = PromptVersion(name="g", version="v1", template="a")
        b = PromptVersion(name="g", version="v2", template="b")
        test = PromptABTest(name="t", variant_a=a, variant_b=b, traffic_split_b=0.5)
        result = test.assign_variant("req-1")
        assert result is a or result is b

    def test_to_dict(self):
        a = PromptVersion(name="g", version="v1", template="a")
        b = PromptVersion(name="g", version="v2", template="b")
        test = PromptABTest(name="t", variant_a=a, variant_b=b)
        d = test.to_dict()
        assert d["traffic_split_b"] == 0.5
