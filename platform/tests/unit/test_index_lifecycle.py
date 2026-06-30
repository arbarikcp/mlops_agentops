"""Unit tests for platform/llm/index_lifecycle.py (Day 112)."""

import pytest

from llm.index_lifecycle import CacheKey, EmbeddingMigrationPlan, RAGCache, StaleDocPolicy


class TestStaleDocPolicy:
    def test_basic(self):
        p = StaleDocPolicy(ttl_days=30, doc_id="d1", last_updated_days_ago=10)
        assert p.is_stale() is False

    def test_is_stale_true(self):
        p = StaleDocPolicy(ttl_days=30, doc_id="d1", last_updated_days_ago=31)
        assert p.is_stale() is True

    def test_is_stale_boundary_equal_not_stale(self):
        p = StaleDocPolicy(ttl_days=30, doc_id="d1", last_updated_days_ago=30)
        assert p.is_stale() is False

    def test_ttl_days_zero_raises(self):
        with pytest.raises(ValueError, match="ttl_days"):
            StaleDocPolicy(ttl_days=0, doc_id="d1", last_updated_days_ago=1)

    def test_negative_ttl_raises(self):
        with pytest.raises(ValueError, match="ttl_days"):
            StaleDocPolicy(ttl_days=-1, doc_id="d1", last_updated_days_ago=1)

    def test_empty_doc_id_raises(self):
        with pytest.raises(ValueError, match="doc_id"):
            StaleDocPolicy(ttl_days=30, doc_id="", last_updated_days_ago=1)

    def test_negative_last_updated_raises(self):
        with pytest.raises(ValueError, match="last_updated_days_ago"):
            StaleDocPolicy(ttl_days=30, doc_id="d1", last_updated_days_ago=-1)

    def test_to_dict(self):
        p = StaleDocPolicy(ttl_days=30, doc_id="d1", last_updated_days_ago=10)
        assert p.to_dict() == {"ttl_days": 30, "doc_id": "d1", "last_updated_days_ago": 10}


class TestEmbeddingMigrationPlan:
    def test_basic(self):
        m = EmbeddingMigrationPlan(old_model="ada-002", new_model="3-small", old_index_version="v1")
        assert m.status == "pending"
        assert m.is_safe_to_switch() is False

    def test_empty_old_model_raises(self):
        with pytest.raises(ValueError, match="old_model"):
            EmbeddingMigrationPlan(old_model="", new_model="new", old_index_version="v1")

    def test_empty_new_model_raises(self):
        with pytest.raises(ValueError, match="new_model"):
            EmbeddingMigrationPlan(old_model="old", new_model="", old_index_version="v1")

    def test_empty_old_index_version_raises(self):
        with pytest.raises(ValueError, match="old_index_version"):
            EmbeddingMigrationPlan(old_model="old", new_model="new", old_index_version="")

    def test_same_model_raises(self):
        with pytest.raises(ValueError, match="differ"):
            EmbeddingMigrationPlan(old_model="same", new_model="same", old_index_version="v1")

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="status"):
            EmbeddingMigrationPlan(
                old_model="old", new_model="new", old_index_version="v1", status="done"
            )

    @pytest.mark.parametrize(
        "status", ["pending", "building", "validating", "ready", "rolled_back"]
    )
    def test_valid_statuses(self, status):
        m = EmbeddingMigrationPlan(
            old_model="old", new_model="new", old_index_version="v1", status=status
        )
        assert m.status == status

    def test_is_safe_to_switch_requires_ready_and_new_version(self):
        m = EmbeddingMigrationPlan(
            old_model="old",
            new_model="new",
            old_index_version="v1",
            new_index_version="v2",
            status="ready",
        )
        assert m.is_safe_to_switch() is True

    def test_is_safe_to_switch_false_if_ready_but_no_new_version(self):
        m = EmbeddingMigrationPlan(
            old_model="old", new_model="new", old_index_version="v1", status="ready"
        )
        assert m.is_safe_to_switch() is False

    def test_is_safe_to_switch_false_if_not_ready(self):
        m = EmbeddingMigrationPlan(
            old_model="old",
            new_model="new",
            old_index_version="v1",
            new_index_version="v2",
            status="validating",
        )
        assert m.is_safe_to_switch() is False

    def test_to_dict(self):
        m = EmbeddingMigrationPlan(old_model="old", new_model="new", old_index_version="v1")
        d = m.to_dict()
        assert d["status"] == "pending"
        assert d["new_index_version"] == ""


class TestCacheKey:
    def test_key_format(self):
        k = CacheKey(query_hash="abc", index_version_id="v1", prompt_version="p1")
        assert k.key() == "abc:v1:p1"

    def test_empty_query_hash_raises(self):
        with pytest.raises(ValueError, match="query_hash"):
            CacheKey(query_hash="", index_version_id="v1", prompt_version="p1")

    def test_empty_index_version_id_raises(self):
        with pytest.raises(ValueError, match="index_version_id"):
            CacheKey(query_hash="abc", index_version_id="", prompt_version="p1")

    def test_empty_prompt_version_raises(self):
        with pytest.raises(ValueError, match="prompt_version"):
            CacheKey(query_hash="abc", index_version_id="v1", prompt_version="")

    def test_to_dict(self):
        k = CacheKey(query_hash="abc", index_version_id="v1", prompt_version="p1")
        assert k.to_dict() == {"query_hash": "abc", "index_version_id": "v1", "prompt_version": "p1"}


class TestRAGCache:
    def test_put_and_get(self):
        cache = RAGCache()
        key = CacheKey(query_hash="abc", index_version_id="v1", prompt_version="p1")
        cache.put(key, "the answer")
        assert cache.get(key) == "the answer"

    def test_get_missing_returns_none(self):
        cache = RAGCache()
        key = CacheKey(query_hash="abc", index_version_id="v1", prompt_version="p1")
        assert cache.get(key) is None

    def test_invalidate_by_index_version(self):
        cache = RAGCache()
        k1 = CacheKey(query_hash="q1", index_version_id="v1", prompt_version="p1")
        k2 = CacheKey(query_hash="q2", index_version_id="v1", prompt_version="p2")
        k3 = CacheKey(query_hash="q3", index_version_id="v2", prompt_version="p1")
        cache.put(k1, "a1")
        cache.put(k2, "a2")
        cache.put(k3, "a3")
        removed = cache.invalidate_by_index_version("v1")
        assert removed == 2
        assert cache.get(k1) is None
        assert cache.get(k2) is None
        assert cache.get(k3) == "a3"

    def test_invalidate_no_matches_returns_zero(self):
        cache = RAGCache()
        k1 = CacheKey(query_hash="q1", index_version_id="v1", prompt_version="p1")
        cache.put(k1, "a1")
        removed = cache.invalidate_by_index_version("v999")
        assert removed == 0
        assert cache.get(k1) == "a1"

    def test_invalidate_does_not_false_positive_on_substring(self):
        # v1 should not match v10 due to the ":v1:" delimiter check
        cache = RAGCache()
        k1 = CacheKey(query_hash="q1", index_version_id="v10", prompt_version="p1")
        cache.put(k1, "a1")
        removed = cache.invalidate_by_index_version("v1")
        assert removed == 0
        assert cache.get(k1) == "a1"

    def test_to_dict(self):
        cache = RAGCache()
        k1 = CacheKey(query_hash="q1", index_version_id="v1", prompt_version="p1")
        cache.put(k1, "a1")
        assert cache.to_dict() == {"num_entries": 1}
