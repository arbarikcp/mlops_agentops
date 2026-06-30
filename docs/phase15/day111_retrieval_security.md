# Day 111 — Metadata Filtering + Multi-Tenant Retrieval Security + Document ACL Propagation

**Phase 15: RAG Production Operations | Module:** `platform/llm/retrieval_security.py`

## WHY

Vector similarity has no concept of "who is allowed to see this." If tenant
A's confidential contract and tenant B's query happen to land near each
other in embedding space, a naively-shared index will happily retrieve A's
document for B's query — a cross-tenant data leak that looks like a normal
retrieval result, not an obvious bug.

Two related threats this module defends against:

- **Cross-tenant leak / data exfiltration** — access control set at
  ingestion time (`DocumentACL`) must propagate all the way through to
  retrieval-time filtering. If it doesn't, "shared index, per-tenant
  metadata tag" is security theater.
- **Index poisoning** — an attacker contributes a document designed to be
  retrieved for unrelated queries (e.g. stuffed with generic high-frequency
  terms) and to inject malicious instructions into the LLM's context. This
  is a supply-chain-style attack on the corpus itself.

## HOW

- Every document/chunk carries a `DocumentACL` at ingestion: which tenant it
  belongs to, which roles may view it, and a `source_trust_score` reflecting
  how trusted its origin is.
- Every retrieval request is a `TenantRetrievalRequest` carrying the
  requester's `tenant_id` and `requester_role`. Its `mandatory_filter()`
  **always** returns a `tenant_id == self.tenant_id` filter — this is
  computed from the request's own identity, not from anything the caller
  supplies, so a malicious or buggy caller cannot override it by passing a
  different `tenant_id` filter in `filters`.
- `ACLEnforcer` is the actual gate: `is_accessible` requires both tenant
  match AND role membership; `filter_results` runs every candidate through
  that gate before chunks reach the LLM prompt.
- `PoisoningDetector` flags documents whose `source_trust_score` is below a
  floor, or whose retrieval frequency spikes abnormally (a sign of
  prompt-stuffing or SEO-style manipulation targeting the retriever).

## Class Diagram

```mermaid
classDiagram
    class RAGThreatType {
        <<enumeration>>
        INDEX_POISONING
        DATA_EXFILTRATION
        CROSS_TENANT_LEAK
        PROMPT_INJECTION_VIA_CONTEXT
    }

    class DocumentACL {
        +str doc_id
        +str tenant_id
        +list~str~ allowed_roles
        +float source_trust_score
        +to_dict() dict
    }

    class MetadataFilter {
        +str field_name
        +str operator
        +object value
        +to_dict() dict
    }

    class TenantRetrievalRequest {
        +str tenant_id
        +str requester_role
        +str query
        +list~MetadataFilter~ filters
        +mandatory_filter() MetadataFilter
        +to_dict() dict
    }

    class ACLEnforcer {
        +dict~str,DocumentACL~ acls
        +register(acl) None
        +is_accessible(doc_id, request) bool
        +filter_results(doc_ids, request) list
    }

    class PoisoningDetector {
        +float min_trust_score
        +int max_retrieval_frequency_per_hour
        +is_suspicious(acl, retrieval_count_last_hour) bool
        +to_dict() dict
    }

    TenantRetrievalRequest --> MetadataFilter : mandatory_filter()
    ACLEnforcer --> "many" DocumentACL
    ACLEnforcer ..> TenantRetrievalRequest : evaluates
    PoisoningDetector ..> DocumentACL : inspects trust score
```

## Sequence Diagram — Mandatory Tenant Filter + ACL Enforcement

```mermaid
sequenceDiagram
    participant User as Tenant B User
    participant API as Retrieval API
    participant Req as TenantRetrievalRequest
    participant Index as Vector/BM25 Index
    participant ACL as ACLEnforcer

    User->>API: query (tenant_id="B", role="viewer")
    API->>Req: construct TenantRetrievalRequest
    Req-->>API: mandatory_filter() -> tenant_id == "B"
    Note over API: mandatory filter is ALWAYS applied,<br/>regardless of any user-supplied filters
    API->>Index: search(query, filters=[mandatory_filter, ...user_filters])
    Index-->>API: candidate doc_ids (pre-filtered to tenant B where possible)
    API->>ACL: filter_results(candidate doc_ids, request)
    loop each doc_id
        ACL->>ACL: is_accessible? (tenant match AND role in allowed_roles)
    end
    ACL-->>API: accessible doc_ids only
    API-->>User: final retrieved chunks (no tenant A leakage)
```

## Key Design Points

- The mandatory filter is enforced **in two layers**: ideally pushed down
  into the index query itself (cheap), and always re-checked by
  `ACLEnforcer.filter_results` as defense-in-depth (catches index
  bugs/misconfiguration).
- `is_accessible` defaults to `False` for unregistered `doc_id`s — fail
  closed, not fail open.
- `PoisoningDetector.is_suspicious` is intentionally a simple OR of two
  independent signals (low trust OR abnormal frequency) so either alone is
  enough to flag for review.
