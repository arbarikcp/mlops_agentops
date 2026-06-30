"""Unit tests for platform/llm/index_pipeline.py (Day 109)."""

import pytest

from llm.index_pipeline import (
    ChunkingStrategy,
    IndexAlias,
    IndexBuildConfig,
    IndexRegistry,
    IndexVersion,
)


def make_config(**overrides):
    defaults = dict(
        source_uri="s3://bucket/docs",
        embedding_model="text-embedding-3-small",
        chunking_strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=512,
        chunk_overlap=50,
    )
    defaults.update(overrides)
    return IndexBuildConfig(**defaults)


class TestChunkingStrategy:
    def test_members(self):
        assert ChunkingStrategy.FIXED_SIZE.value == "fixed_size"
        assert ChunkingStrategy.SENTENCE.value == "sentence"
        assert ChunkingStrategy.SEMANTIC.value == "semantic"
        assert ChunkingStrategy.RECURSIVE.value == "recursive"


class TestIndexBuildConfig:
    def test_basic(self):
        c = make_config()
        assert c.chunk_size == 512

    def test_empty_source_uri_raises(self):
        with pytest.raises(ValueError, match="source_uri"):
            make_config(source_uri="")

    def test_empty_embedding_model_raises(self):
        with pytest.raises(ValueError, match="embedding_model"):
            make_config(embedding_model="")

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            make_config(chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ValueError, match="chunk_size"):
            make_config(chunk_size=-10)

    def test_chunk_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            make_config(chunk_size=100, chunk_overlap=100)

    def test_negative_chunk_overlap_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            make_config(chunk_overlap=-1)

    def test_to_dict(self):
        c = make_config()
        d = c.to_dict()
        assert d["chunking_strategy"] == "fixed_size"
        assert d["source_uri"] == "s3://bucket/docs"
        assert d["chunk_size"] == 512
        assert d["chunk_overlap"] == 50


class TestIndexVersion:
    def test_basic_and_content_hash_computed(self):
        v = IndexVersion(
            version_id="v1",
            build_config=make_config(),
            num_documents=10,
            num_chunks=100,
            created_at="2026-01-01T00:00:00Z",
        )
        assert len(v.content_hash) == 12

    def test_explicit_content_hash_preserved(self):
        v = IndexVersion(
            version_id="v1",
            build_config=make_config(),
            num_documents=10,
            num_chunks=100,
            created_at="now",
            content_hash="deadbeefcafe",
        )
        assert v.content_hash == "deadbeefcafe"

    def test_same_inputs_produce_same_hash(self):
        v1 = IndexVersion(
            version_id="v1",
            build_config=make_config(),
            num_documents=10,
            num_chunks=100,
            created_at="now",
        )
        v2 = IndexVersion(
            version_id="v2",
            build_config=make_config(),
            num_documents=999,
            num_chunks=100,
            created_at="later",
        )
        assert v1.content_hash == v2.content_hash

    def test_empty_version_id_raises(self):
        with pytest.raises(ValueError, match="version_id"):
            IndexVersion(
                version_id="",
                build_config=make_config(),
                num_documents=1,
                num_chunks=1,
                created_at="now",
            )

    def test_negative_num_documents_raises(self):
        with pytest.raises(ValueError, match="num_documents"):
            IndexVersion(
                version_id="v1",
                build_config=make_config(),
                num_documents=-1,
                num_chunks=1,
                created_at="now",
            )

    def test_negative_num_chunks_raises(self):
        with pytest.raises(ValueError, match="num_chunks"):
            IndexVersion(
                version_id="v1",
                build_config=make_config(),
                num_documents=1,
                num_chunks=-1,
                created_at="now",
            )

    def test_to_dict(self):
        v = IndexVersion(
            version_id="v1",
            build_config=make_config(),
            num_documents=1,
            num_chunks=1,
            created_at="now",
        )
        d = v.to_dict()
        assert d["version_id"] == "v1"
        assert isinstance(d["build_config"], dict)


class TestIndexAlias:
    def test_basic(self):
        a = IndexAlias(alias_name="production", current_version_id="v1")
        assert a.history == []

    def test_empty_alias_name_raises(self):
        with pytest.raises(ValueError, match="alias_name"):
            IndexAlias(alias_name="", current_version_id="v1")

    def test_repoint(self):
        a = IndexAlias(alias_name="production", current_version_id="v1")
        a.repoint("v2")
        assert a.current_version_id == "v2"
        assert a.history == ["v1"]

    def test_rollback(self):
        a = IndexAlias(alias_name="production", current_version_id="v1")
        a.repoint("v2")
        rolled_to = a.rollback()
        assert rolled_to == "v1"
        assert a.current_version_id == "v1"
        assert a.history == []

    def test_rollback_empty_history_raises(self):
        a = IndexAlias(alias_name="production", current_version_id="v1")
        with pytest.raises(ValueError):
            a.rollback()

    def test_multiple_repoints_and_rollback(self):
        a = IndexAlias(alias_name="production", current_version_id="v1")
        a.repoint("v2")
        a.repoint("v3")
        assert a.history == ["v1", "v2"]
        assert a.rollback() == "v2"
        assert a.current_version_id == "v2"

    def test_to_dict(self):
        a = IndexAlias(alias_name="production", current_version_id="v1")
        d = a.to_dict()
        assert d == {"alias_name": "production", "current_version_id": "v1", "history": []}


class TestIndexRegistry:
    def _version(self, vid="v1"):
        return IndexVersion(
            version_id=vid,
            build_config=make_config(),
            num_documents=1,
            num_chunks=1,
            created_at="now",
        )

    def test_register_and_get(self):
        reg = IndexRegistry()
        v = self._version()
        reg.register_version(v)
        assert reg.get_version("v1") is v

    def test_get_unknown_raises_keyerror(self):
        reg = IndexRegistry()
        with pytest.raises(KeyError):
            reg.get_version("nope")

    def test_set_alias_creates_new(self):
        reg = IndexRegistry()
        reg.register_version(self._version("v1"))
        reg.set_alias("production", "v1")
        assert reg.aliases["production"].current_version_id == "v1"

    def test_set_alias_repoints_existing(self):
        reg = IndexRegistry()
        reg.register_version(self._version("v1"))
        reg.register_version(self._version("v2"))
        reg.set_alias("production", "v1")
        reg.set_alias("production", "v2")
        assert reg.aliases["production"].current_version_id == "v2"
        assert reg.aliases["production"].history == ["v1"]

    def test_resolve(self):
        reg = IndexRegistry()
        reg.register_version(self._version("v1"))
        reg.set_alias("production", "v1")
        resolved = reg.resolve("production")
        assert resolved.version_id == "v1"

    def test_resolve_unknown_alias_raises(self):
        reg = IndexRegistry()
        with pytest.raises(KeyError):
            reg.resolve("nope")

    def test_rollback_via_registry_alias(self):
        reg = IndexRegistry()
        reg.register_version(self._version("v1"))
        reg.register_version(self._version("v2"))
        reg.set_alias("production", "v1")
        reg.set_alias("production", "v2")
        rolled_to = reg.aliases["production"].rollback()
        assert rolled_to == "v1"
        assert reg.resolve("production").version_id == "v1"
