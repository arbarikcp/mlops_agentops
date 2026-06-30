# Day 109 — Index Build Pipeline + Versioning + Rollback

**Phase 15: RAG Production Operations | Module:** `platform/llm/index_pipeline.py`

## WHY

A vector index is the load-bearing artifact of a RAG system, exactly the way a
trained model checkpoint is the load-bearing artifact of a classical ML
system. It deserves the same production guarantees:

- **Reproducibility** — given an index version, you must be able to say
  exactly what documents, chunking strategy, and embedding model produced it.
- **Immutability** — once built, a version never changes in place. Mutating
  an index silently (re-embedding chunks, dropping documents) breaks the
  ability to reason about why retrieval quality changed.
- **Instant rollback** — if a re-index introduces bad chunks, a broken
  embedding model, or corrupted source documents, you need to revert
  production traffic to the last known-good version in seconds, not by
  re-running the whole pipeline.

Without versioning, "the index" is a mutable blob nobody can audit. A bad
re-index becomes a multi-hour incident instead of a one-line pointer flip.

## HOW

Every build of an index produces an `IndexVersion`: an immutable record
identified by a `version_id` and a `content_hash` derived from the build
configuration (embedding model, chunking strategy, chunk count). The actual
"production index" that the serving layer queries is never referenced
directly — it's referenced through an `IndexAlias`, a named pointer (e.g.
`"production"`) that can be repointed to any registered version.

- **Build** → produces a new immutable `IndexVersion`, registered in the
  `IndexRegistry`.
- **Promote** → `IndexRegistry.set_alias("production", new_version_id)`
  repoints the alias, pushing the previous version onto `history`.
- **Rollback** → `IndexAlias.rollback()` pops the last entry off `history`
  and makes it current again — no rebuild required.

This mirrors model registry patterns (MLflow Model Registry stages, SageMaker
Model Package versions) applied to retrieval indexes.

## Class Diagram

```mermaid
classDiagram
    class ChunkingStrategy {
        <<enumeration>>
        FIXED_SIZE
        SENTENCE
        SEMANTIC
        RECURSIVE
    }

    class IndexBuildConfig {
        +str source_uri
        +str embedding_model
        +ChunkingStrategy chunking_strategy
        +int chunk_size
        +int chunk_overlap
        +to_dict() dict
    }

    class IndexVersion {
        +str version_id
        +IndexBuildConfig build_config
        +int num_documents
        +int num_chunks
        +str created_at
        +str content_hash
        +to_dict() dict
    }

    class IndexAlias {
        +str alias_name
        +str current_version_id
        +list~str~ history
        +repoint(new_version_id) None
        +rollback() str
        +to_dict() dict
    }

    class IndexRegistry {
        +dict~str,IndexVersion~ versions
        +dict~str,IndexAlias~ aliases
        +register_version(v) None
        +get_version(version_id) IndexVersion
        +set_alias(alias_name, version_id) None
        +resolve(alias_name) IndexVersion
    }

    IndexVersion --> IndexBuildConfig
    IndexRegistry --> "many" IndexVersion
    IndexRegistry --> "many" IndexAlias
    IndexAlias --> IndexVersion : current_version_id (by id)
```

## Sequence Diagram — Build, Promote, and Rollback

```mermaid
sequenceDiagram
    participant Eng as ML Engineer
    participant Pipe as Index Build Pipeline
    participant Reg as IndexRegistry
    participant Alias as IndexAlias("production")
    participant Serve as Retrieval Service

    Eng->>Pipe: trigger build (new docs / new embedding model)
    Pipe->>Pipe: chunk + embed + write vectors
    Pipe->>Reg: register_version(IndexVersion v2)
    Eng->>Reg: set_alias("production", "v2")
    Reg->>Alias: repoint("v2")
    Note over Alias: history = [v1]<br/>current = v2
    Serve->>Reg: resolve("production")
    Reg-->>Serve: IndexVersion v2

    Note over Eng,Serve: Quality regression detected on v2!

    Eng->>Alias: rollback()
    Alias-->>Eng: "v1"
    Note over Alias: history = []<br/>current = v1
    Serve->>Reg: resolve("production")
    Reg-->>Serve: IndexVersion v1 (restored)
```

## Key Design Points

- `content_hash` is computed automatically in `__post_init__` from
  `embedding_model:chunking_strategy:num_chunks` via SHA-256 (truncated to 12
  hex chars) — this gives a cheap fingerprint for spotting accidental
  duplicate builds, without requiring a full corpus hash.
- `IndexAlias.history` is a simple stack: `repoint` pushes, `rollback` pops.
  Multiple rollbacks walk back through prior promotions.
- `IndexRegistry.set_alias` is idempotent for "first use" (creates the alias)
  vs. "subsequent use" (repoints) — callers never need to branch on whether
  the alias already exists.
