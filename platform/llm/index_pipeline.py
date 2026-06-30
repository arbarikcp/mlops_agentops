"""
index_pipeline — Day 109: Index Build Pipeline + Versioning + Rollback

A vector index is a production artifact, just like a trained model. This module
treats index builds as immutable, content-addressed versions with an alias
(pointer) layer for safe promotion and instant rollback.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class ChunkingStrategy(str, Enum):
    """Supported document chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"


@dataclass
class IndexBuildConfig:
    """Configuration describing how a vector index should be built."""

    source_uri: str
    embedding_model: str
    chunking_strategy: ChunkingStrategy
    chunk_size: int = 512
    chunk_overlap: int = 50

    def __post_init__(self) -> None:
        if not self.source_uri:
            raise ValueError("source_uri must be non-empty")
        if not self.embedding_model:
            raise ValueError("embedding_model must be non-empty")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if not (0 <= self.chunk_overlap < self.chunk_size):
            raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    def to_dict(self) -> dict:
        return {
            "source_uri": self.source_uri,
            "embedding_model": self.embedding_model,
            "chunking_strategy": self.chunking_strategy.value,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


@dataclass
class IndexVersion:
    """An immutable, content-addressed index build artifact."""

    version_id: str
    build_config: IndexBuildConfig
    num_documents: int
    num_chunks: int
    created_at: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.version_id:
            raise ValueError("version_id must be non-empty")
        if self.num_documents < 0:
            raise ValueError("num_documents must be >= 0")
        if self.num_chunks < 0:
            raise ValueError("num_chunks must be >= 0")
        if not self.content_hash:
            raw = f"{self.build_config.embedding_model}:{self.build_config.chunking_strategy}:{self.num_chunks}"
            self.content_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "build_config": self.build_config.to_dict(),
            "num_documents": self.num_documents,
            "num_chunks": self.num_chunks,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
        }


@dataclass
class IndexAlias:
    """A mutable pointer (e.g. 'production') to a specific immutable IndexVersion."""

    alias_name: str
    current_version_id: str
    history: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.alias_name:
            raise ValueError("alias_name must be non-empty")

    def repoint(self, new_version_id: str) -> None:
        self.history.append(self.current_version_id)
        self.current_version_id = new_version_id

    def rollback(self) -> str:
        if not self.history:
            raise ValueError("no history to roll back to")
        previous = self.history.pop()
        self.current_version_id = previous
        return previous

    def to_dict(self) -> dict:
        return {
            "alias_name": self.alias_name,
            "current_version_id": self.current_version_id,
            "history": list(self.history),
        }


@dataclass
class IndexRegistry:
    """Registry tracking all known index versions and aliases."""

    versions: dict[str, IndexVersion] = field(default_factory=dict)
    aliases: dict[str, IndexAlias] = field(default_factory=dict)

    def register_version(self, v: IndexVersion) -> None:
        self.versions[v.version_id] = v

    def get_version(self, version_id: str) -> IndexVersion:
        if version_id not in self.versions:
            raise KeyError(f"unknown index version: {version_id}")
        return self.versions[version_id]

    def set_alias(self, alias_name: str, version_id: str) -> None:
        if alias_name in self.aliases:
            self.aliases[alias_name].repoint(version_id)
        else:
            self.aliases[alias_name] = IndexAlias(alias_name=alias_name, current_version_id=version_id)

    def resolve(self, alias_name: str) -> IndexVersion:
        if alias_name not in self.aliases:
            raise KeyError(f"unknown alias: {alias_name}")
        return self.get_version(self.aliases[alias_name].current_version_id)
