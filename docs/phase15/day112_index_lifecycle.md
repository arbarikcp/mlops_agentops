# Day 112 — Stale-Document Removal + Embedding-Model Migration + RAG Cache Invalidation

**Phase 15: RAG Production Operations | Module:** `platform/llm/index_lifecycle.py`

## WHY

Indexes are not "build once, query forever" artifacts. Three forms of decay
threaten correctness over time:

- **Stale documents** — a deprecated policy or outdated price sheet stays
  retrievable forever unless something explicitly sweeps it out. The result
  is confidently wrong answers sourced from content that's no longer true.
- **Embedding-model migration** — this is the highest-risk lifecycle event.
  Vectors from two different embedding models are **not comparable** — they
  live in different, unrelated vector spaces, even if dimensionality
  happens to match. Mixing old and new embeddings in the same similarity
  search silently produces garbage rankings with no error, no exception,
  just bad retrieval that looks plausible.
- **Stale cache** — a cached `query -> answer` pair served after the
  underlying document or index changed silently serves an outdated answer
  with full confidence.

## HOW

- **Stale docs**: `StaleDocPolicy` is a pure TTL check —
  `last_updated_days_ago > ttl_days`. A sweep job iterates documents,
  evaluates this policy, and purges anything stale from the index.
- **Embedding migration**: `EmbeddingMigrationPlan` models the migration as
  an explicit state machine: `pending → building → validating → ready →
  (switch)`, with `rolled_back` as an escape hatch. The new index is always
  built **fully, in parallel**, as a brand-new `IndexVersion` (Day 109) — it
  is never partially migrated in place. `is_safe_to_switch()` only returns
  `True` once status is `"ready"` AND a `new_index_version` actually exists,
  guarding against switching to an empty/unbuilt target.
- **Cache invalidation**: `CacheKey.key()` bakes the `index_version_id`
  (and `prompt_version`) directly into the cache key string. When an index
  is rebuilt and a new version is promoted, all old cache entries become
  unreachable by construction — no separate invalidation logic needed for
  the *new* version's queries. `RAGCache.invalidate_by_index_version`
  additionally provides an explicit sweep to reclaim memory from now-dead
  entries tied to retired index versions.

## Class Diagram

```mermaid
classDiagram
    class StaleDocPolicy {
        +int ttl_days
        +str doc_id
        +int last_updated_days_ago
        +is_stale() bool
        +to_dict() dict
    }

    class EmbeddingMigrationPlan {
        +str old_model
        +str new_model
        +str old_index_version
        +str new_index_version
        +str status
        +is_safe_to_switch() bool
        +to_dict() dict
    }

    class CacheKey {
        +str query_hash
        +str index_version_id
        +str prompt_version
        +key() str
        +to_dict() dict
    }

    class RAGCache {
        +dict~str,str~ entries
        +put(key, answer) None
        +get(key) str
        +invalidate_by_index_version(index_version_id) int
        +to_dict() dict
    }

    RAGCache --> CacheKey : keyed by
    CacheKey ..> EmbeddingMigrationPlan : index_version_id ties to migration target
```

## Sequence Diagram — Safe Embedding Migration with Cache Invalidation

```mermaid
sequenceDiagram
    participant Eng as ML Engineer
    participant Plan as EmbeddingMigrationPlan
    participant Pipe as Index Build Pipeline
    participant Reg as IndexRegistry (Day 109)
    participant Cache as RAGCache
    participant Serve as Retrieval Service

    Eng->>Plan: create(old_model, new_model, old_index_version="v1")
    Note over Plan: status = "pending"
    Eng->>Plan: status = "building"
    Plan->>Pipe: build full new index with new_model (shadow build)
    Pipe-->>Reg: register_version(v2)
    Eng->>Plan: new_index_version = "v2", status = "validating"
    Eng->>Eng: run eval suite against v2 (Day 113)
    alt eval passes threshold
        Eng->>Plan: status = "ready"
        Plan-->>Eng: is_safe_to_switch() == True
        Eng->>Reg: set_alias("production", "v2")
        Eng->>Cache: invalidate_by_index_version("v1")
        Cache-->>Eng: removed N stale entries
        Serve->>Reg: resolve("production") -> v2
    else eval fails
        Eng->>Plan: status = "rolled_back"
        Note over Serve: production alias stays on v1,<br/>no partial migration ever happened
    end
```

## Key Design Points

- `EmbeddingMigrationPlan` validates `old_model != new_model` — a
  "migration" to the same model is a no-op and likely a config bug, so it's
  rejected at construction.
- `is_safe_to_switch()` is a conjunction, not an OR — both `status ==
  "ready"` and a populated `new_index_version` are required, so an
  incomplete plan can never be mistaken for a safe one.
- `invalidate_by_index_version` checks for the delimited substring
  `f":{index_version_id}:"`, not a bare substring match, so `"v1"` does not
  accidentally match `"v10"` — this was deliberately tested.
