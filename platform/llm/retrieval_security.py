"""
retrieval_security — Day 111: Metadata Filtering + Multi-Tenant Retrieval
Security + Document ACL Propagation

A shared vector index across tenants without enforced filtering is a data
breach waiting to happen — semantic similarity doesn't respect access
boundaries. This module makes tenant isolation a *mandatory* pre-filter
(not an optional metadata filter) and provides ACL enforcement plus basic
index-poisoning detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "RAGThreatType",
    "DocumentACL",
    "MetadataFilter",
    "TenantRetrievalRequest",
    "ACLEnforcer",
    "PoisoningDetector",
]

_VALID_OPERATORS = {"eq", "in", "gte", "lte"}


class RAGThreatType(str, Enum):
    """Threat categories specific to RAG retrieval pipelines."""

    INDEX_POISONING = "index_poisoning"
    DATA_EXFILTRATION = "data_exfiltration"
    CROSS_TENANT_LEAK = "cross_tenant_leak"
    PROMPT_INJECTION_VIA_CONTEXT = "prompt_injection_via_context"


@dataclass
class DocumentACL:
    """Access control metadata attached to a document/chunk at ingestion time."""

    doc_id: str
    tenant_id: str
    allowed_roles: list[str] = field(default_factory=lambda: ["viewer"])
    source_trust_score: float = 1.0

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if not self.allowed_roles:
            raise ValueError("allowed_roles must be non-empty")
        if not (0 <= self.source_trust_score <= 1):
            raise ValueError("source_trust_score must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "tenant_id": self.tenant_id,
            "allowed_roles": list(self.allowed_roles),
            "source_trust_score": self.source_trust_score,
        }


@dataclass
class MetadataFilter:
    """A single metadata predicate applied at retrieval time."""

    field_name: str
    operator: str
    value: object

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("field_name must be non-empty")
        if self.operator not in _VALID_OPERATORS:
            raise ValueError(f"operator must be one of {_VALID_OPERATORS}")

    def to_dict(self) -> dict:
        return {"field_name": self.field_name, "operator": self.operator, "value": self.value}


@dataclass
class TenantRetrievalRequest:
    """A retrieval request scoped to a specific tenant and requester role."""

    tenant_id: str
    requester_role: str
    query: str
    filters: list[MetadataFilter] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if not self.requester_role:
            raise ValueError("requester_role must be non-empty")
        if not self.query:
            raise ValueError("query must be non-empty")

    def mandatory_filter(self) -> MetadataFilter:
        return MetadataFilter(field_name="tenant_id", operator="eq", value=self.tenant_id)

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "requester_role": self.requester_role,
            "query": self.query,
            "filters": [f.to_dict() for f in self.filters],
        }


@dataclass
class ACLEnforcer:
    """Enforces document ACLs against retrieval requests."""

    acls: dict[str, DocumentACL] = field(default_factory=dict)

    def register(self, acl: DocumentACL) -> None:
        self.acls[acl.doc_id] = acl

    def is_accessible(self, doc_id: str, request: TenantRetrievalRequest) -> bool:
        if doc_id not in self.acls:
            return False
        acl = self.acls[doc_id]
        return acl.tenant_id == request.tenant_id and request.requester_role in acl.allowed_roles

    def filter_results(self, doc_ids: list[str], request: TenantRetrievalRequest) -> list[str]:
        return [doc_id for doc_id in doc_ids if self.is_accessible(doc_id, request)]


@dataclass
class PoisoningDetector:
    """Flags documents that look like index-poisoning candidates."""

    min_trust_score: float = 0.3
    max_retrieval_frequency_per_hour: int = 1000

    def __post_init__(self) -> None:
        if not (0 <= self.min_trust_score <= 1):
            raise ValueError("min_trust_score must be in [0, 1]")
        if self.max_retrieval_frequency_per_hour <= 0:
            raise ValueError("max_retrieval_frequency_per_hour must be > 0")

    def is_suspicious(self, acl: DocumentACL, retrieval_count_last_hour: int) -> bool:
        return (
            acl.source_trust_score < self.min_trust_score
            or retrieval_count_last_hour > self.max_retrieval_frequency_per_hour
        )

    def to_dict(self) -> dict:
        return {
            "min_trust_score": self.min_trust_score,
            "max_retrieval_frequency_per_hour": self.max_retrieval_frequency_per_hour,
        }
