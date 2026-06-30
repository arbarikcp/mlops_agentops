"""
index_lifecycle — Day 112: Stale-Document Removal + Embedding-Model Migration
+ RAG Cache Invalidation

Three lifecycle concerns that all share a theme: state goes stale and must
be explicitly invalidated, never silently mixed with fresh state.

- Stale documents must be swept out of the index (TTL-based or explicit).
- Embedding-model migrations must never mix old/new vector spaces in one
  similarity search — migrate via a full parallel build + atomic switch.
- RAG response caches must invalidate automatically whenever the underlying
  index version changes, by baking the index version into the cache key.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "StaleDocPolicy",
    "EmbeddingMigrationPlan",
    "CacheKey",
    "RAGCache",
]

_VALID_MIGRATION_STATUSES = {"pending", "building", "validating", "ready", "rolled_back"}


@dataclass
class StaleDocPolicy:
    """TTL-based staleness policy for a single document."""

    ttl_days: int
    doc_id: str
    last_updated_days_ago: int

    def __post_init__(self) -> None:
        if self.ttl_days <= 0:
            raise ValueError("ttl_days must be > 0")
        if not self.doc_id:
            raise ValueError("doc_id must be non-empty")
        if self.last_updated_days_ago < 0:
            raise ValueError("last_updated_days_ago must be >= 0")

    def is_stale(self) -> bool:
        return self.last_updated_days_ago > self.ttl_days

    def to_dict(self) -> dict:
        return {
            "ttl_days": self.ttl_days,
            "doc_id": self.doc_id,
            "last_updated_days_ago": self.last_updated_days_ago,
        }


@dataclass
class EmbeddingMigrationPlan:
    """Tracks the state of migrating from one embedding model to another."""

    old_model: str
    new_model: str
    old_index_version: str
    new_index_version: str = ""
    status: str = "pending"

    def __post_init__(self) -> None:
        if not self.old_model:
            raise ValueError("old_model must be non-empty")
        if not self.new_model:
            raise ValueError("new_model must be non-empty")
        if not self.old_index_version:
            raise ValueError("old_index_version must be non-empty")
        if self.old_model == self.new_model:
            raise ValueError("old_model must differ from new_model")
        if self.status not in _VALID_MIGRATION_STATUSES:
            raise ValueError(f"status must be one of {_VALID_MIGRATION_STATUSES}")

    def is_safe_to_switch(self) -> bool:
        return self.status == "ready" and self.new_index_version != ""

    def to_dict(self) -> dict:
        return {
            "old_model": self.old_model,
            "new_model": self.new_model,
            "old_index_version": self.old_index_version,
            "new_index_version": self.new_index_version,
            "status": self.status,
        }


@dataclass
class CacheKey:
    """A cache key that ties a cached answer to its full provenance context."""

    query_hash: str
    index_version_id: str
    prompt_version: str

    def __post_init__(self) -> None:
        if not self.query_hash:
            raise ValueError("query_hash must be non-empty")
        if not self.index_version_id:
            raise ValueError("index_version_id must be non-empty")
        if not self.prompt_version:
            raise ValueError("prompt_version must be non-empty")

    def key(self) -> str:
        return f"{self.query_hash}:{self.index_version_id}:{self.prompt_version}"

    def to_dict(self) -> dict:
        return {
            "query_hash": self.query_hash,
            "index_version_id": self.index_version_id,
            "prompt_version": self.prompt_version,
        }


@dataclass
class RAGCache:
    """A simple cache of query+context -> answer, invalidatable by index version."""

    entries: dict[str, str] = field(default_factory=dict)

    def put(self, key: CacheKey, answer: str) -> None:
        self.entries[key.key()] = answer

    def get(self, key: CacheKey) -> str | None:
        return self.entries.get(key.key())

    def invalidate_by_index_version(self, index_version_id: str) -> int:
        marker = f":{index_version_id}:"
        to_remove = [k for k in self.entries if marker in k]
        for k in to_remove:
            del self.entries[k]
        return len(to_remove)

    def to_dict(self) -> dict:
        return {"num_entries": len(self.entries)}
