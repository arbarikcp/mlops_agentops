"""Unit tests for platform/llm/retrieval_security.py (Day 111)."""

import pytest

from llm.retrieval_security import (
    ACLEnforcer,
    DocumentACL,
    MetadataFilter,
    PoisoningDetector,
    RAGThreatType,
    TenantRetrievalRequest,
)


class TestRAGThreatType:
    def test_members(self):
        assert RAGThreatType.INDEX_POISONING.value == "index_poisoning"
        assert RAGThreatType.DATA_EXFILTRATION.value == "data_exfiltration"
        assert RAGThreatType.CROSS_TENANT_LEAK.value == "cross_tenant_leak"
        assert RAGThreatType.PROMPT_INJECTION_VIA_CONTEXT.value == "prompt_injection_via_context"


class TestDocumentACL:
    def test_defaults(self):
        acl = DocumentACL(doc_id="d1", tenant_id="t1")
        assert acl.allowed_roles == ["viewer"]
        assert acl.source_trust_score == 1.0

    def test_empty_doc_id_raises(self):
        with pytest.raises(ValueError, match="doc_id"):
            DocumentACL(doc_id="", tenant_id="t1")

    def test_empty_tenant_id_raises(self):
        with pytest.raises(ValueError, match="tenant_id"):
            DocumentACL(doc_id="d1", tenant_id="")

    def test_empty_allowed_roles_raises(self):
        with pytest.raises(ValueError, match="allowed_roles"):
            DocumentACL(doc_id="d1", tenant_id="t1", allowed_roles=[])

    def test_trust_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="source_trust_score"):
            DocumentACL(doc_id="d1", tenant_id="t1", source_trust_score=1.5)

    def test_negative_trust_score_raises(self):
        with pytest.raises(ValueError, match="source_trust_score"):
            DocumentACL(doc_id="d1", tenant_id="t1", source_trust_score=-0.1)

    def test_to_dict(self):
        acl = DocumentACL(doc_id="d1", tenant_id="t1", allowed_roles=["admin"], source_trust_score=0.9)
        d = acl.to_dict()
        assert d["allowed_roles"] == ["admin"]


class TestMetadataFilter:
    def test_basic(self):
        f = MetadataFilter(field_name="doc_type", operator="eq", value="policy")
        assert f.operator == "eq"

    def test_empty_field_name_raises(self):
        with pytest.raises(ValueError, match="field_name"):
            MetadataFilter(field_name="", operator="eq", value=1)

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="operator"):
            MetadataFilter(field_name="f", operator="neq", value=1)

    @pytest.mark.parametrize("op", ["eq", "in", "gte", "lte"])
    def test_valid_operators(self, op):
        f = MetadataFilter(field_name="f", operator=op, value=1)
        assert f.operator == op

    def test_to_dict(self):
        f = MetadataFilter(field_name="f", operator="in", value=[1, 2])
        assert f.to_dict() == {"field_name": "f", "operator": "in", "value": [1, 2]}


class TestTenantRetrievalRequest:
    def test_basic(self):
        r = TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="hello")
        assert r.filters == []

    def test_empty_tenant_id_raises(self):
        with pytest.raises(ValueError, match="tenant_id"):
            TenantRetrievalRequest(tenant_id="", requester_role="viewer", query="hello")

    def test_empty_requester_role_raises(self):
        with pytest.raises(ValueError, match="requester_role"):
            TenantRetrievalRequest(tenant_id="t1", requester_role="", query="hello")

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="")

    def test_mandatory_filter(self):
        r = TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="hello")
        mf = r.mandatory_filter()
        assert mf.field_name == "tenant_id"
        assert mf.operator == "eq"
        assert mf.value == "t1"

    def test_mandatory_filter_ignores_user_filters(self):
        evil_filter = MetadataFilter(field_name="tenant_id", operator="eq", value="other_tenant")
        r = TenantRetrievalRequest(
            tenant_id="t1", requester_role="viewer", query="hello", filters=[evil_filter]
        )
        assert r.mandatory_filter().value == "t1"

    def test_to_dict(self):
        r = TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="hello")
        d = r.to_dict()
        assert d["tenant_id"] == "t1"
        assert d["filters"] == []


class TestACLEnforcer:
    def test_register_and_accessible(self):
        enforcer = ACLEnforcer()
        enforcer.register(DocumentACL(doc_id="d1", tenant_id="t1", allowed_roles=["viewer"]))
        req = TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="q")
        assert enforcer.is_accessible("d1", req) is True

    def test_not_registered_inaccessible(self):
        enforcer = ACLEnforcer()
        req = TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="q")
        assert enforcer.is_accessible("unknown", req) is False

    def test_cross_tenant_inaccessible(self):
        enforcer = ACLEnforcer()
        enforcer.register(DocumentACL(doc_id="d1", tenant_id="t1", allowed_roles=["viewer"]))
        req = TenantRetrievalRequest(tenant_id="t2", requester_role="viewer", query="q")
        assert enforcer.is_accessible("d1", req) is False

    def test_wrong_role_inaccessible(self):
        enforcer = ACLEnforcer()
        enforcer.register(DocumentACL(doc_id="d1", tenant_id="t1", allowed_roles=["admin"]))
        req = TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="q")
        assert enforcer.is_accessible("d1", req) is False

    def test_filter_results_preserves_order_and_excludes_inaccessible(self):
        enforcer = ACLEnforcer()
        enforcer.register(DocumentACL(doc_id="d1", tenant_id="t1", allowed_roles=["viewer"]))
        enforcer.register(DocumentACL(doc_id="d2", tenant_id="t2", allowed_roles=["viewer"]))
        enforcer.register(DocumentACL(doc_id="d3", tenant_id="t1", allowed_roles=["viewer"]))
        req = TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="q")
        result = enforcer.filter_results(["d1", "d2", "d3", "d4"], req)
        assert result == ["d1", "d3"]

    def test_filter_results_empty_input(self):
        enforcer = ACLEnforcer()
        req = TenantRetrievalRequest(tenant_id="t1", requester_role="viewer", query="q")
        assert enforcer.filter_results([], req) == []


class TestPoisoningDetector:
    def test_defaults(self):
        d = PoisoningDetector()
        assert d.min_trust_score == 0.3

    def test_min_trust_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="min_trust_score"):
            PoisoningDetector(min_trust_score=1.5)

    def test_max_frequency_zero_raises(self):
        with pytest.raises(ValueError, match="max_retrieval_frequency_per_hour"):
            PoisoningDetector(max_retrieval_frequency_per_hour=0)

    def test_low_trust_is_suspicious(self):
        d = PoisoningDetector(min_trust_score=0.5)
        acl = DocumentACL(doc_id="d1", tenant_id="t1", source_trust_score=0.2)
        assert d.is_suspicious(acl, retrieval_count_last_hour=5) is True

    def test_high_frequency_is_suspicious(self):
        d = PoisoningDetector(max_retrieval_frequency_per_hour=100)
        acl = DocumentACL(doc_id="d1", tenant_id="t1", source_trust_score=0.9)
        assert d.is_suspicious(acl, retrieval_count_last_hour=200) is True

    def test_normal_doc_not_suspicious(self):
        d = PoisoningDetector()
        acl = DocumentACL(doc_id="d1", tenant_id="t1", source_trust_score=0.9)
        assert d.is_suspicious(acl, retrieval_count_last_hour=5) is False

    def test_to_dict(self):
        d = PoisoningDetector(min_trust_score=0.4, max_retrieval_frequency_per_hour=500)
        assert d.to_dict() == {
            "min_trust_score": 0.4,
            "max_retrieval_frequency_per_hour": 500,
        }
