"""Unit tests for platform/llm/inference_opt.py (Day 94)."""

import pytest
from llm.inference_opt import (
    AttentionType,
    BatchingStrategy,
    ContinuousBatchingConfig,
    InferenceOptConfig,
    KVCacheConfig,
    PagedAttentionConfig,
)


# ── KVCacheConfig ──────────────────────────────────────────────────────────

class TestKVCacheConfig:
    def _make(self, **kwargs):
        defaults = dict(num_layers=32, num_heads=32, head_dim=128)
        defaults.update(kwargs)
        return KVCacheConfig(**defaults)

    def test_cache_size_basic(self):
        # 2 * 32 * 32 * 128 * 4096 * 1 * 2 bytes = 2,147,483,648 B = ~2.147 GB
        cfg = self._make()
        size = cfg.cache_size_gb(1)
        expected = (2 * 32 * 32 * 128 * 4096 * 1 * 2) / 1e9
        assert size == pytest.approx(expected, rel=1e-4)

    def test_cache_size_scales_with_batch(self):
        cfg = self._make()
        assert cfg.cache_size_gb(4) == pytest.approx(4 * cfg.cache_size_gb(1))

    def test_invalid_num_layers(self):
        with pytest.raises(ValueError, match="num_layers"):
            KVCacheConfig(num_layers=0, num_heads=32, head_dim=128)

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError, match="num_heads"):
            KVCacheConfig(num_layers=32, num_heads=0, head_dim=128)

    def test_invalid_batch_size(self):
        cfg = self._make()
        with pytest.raises(ValueError, match="batch_size"):
            cfg.cache_size_gb(0)

    def test_to_dict(self):
        cfg = self._make()
        d = cfg.to_dict()
        assert d["num_layers"] == 32
        assert d["max_seq_len"] == 4096


# ── PagedAttentionConfig ───────────────────────────────────────────────────

class TestPagedAttentionConfig:
    def test_max_concurrent_seqs(self):
        kv = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128, max_seq_len=2048)
        paged = PagedAttentionConfig(block_size=16, max_num_blocks=4096)
        # total_tokens = 4096 * 16 = 65536; seqs = 65536 // 2048 = 32
        assert paged.max_concurrent_seqs(kv) == 32

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            PagedAttentionConfig(block_size=0)

    def test_invalid_gpu_util(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            PagedAttentionConfig(gpu_memory_utilization=0.0)

    def test_gpu_util_exactly_one(self):
        cfg = PagedAttentionConfig(gpu_memory_utilization=1.0)
        assert cfg.gpu_memory_utilization == 1.0

    def test_to_dict(self):
        cfg = PagedAttentionConfig(block_size=32, max_num_blocks=2048)
        d = cfg.to_dict()
        assert d["block_size"] == 32
        assert d["max_num_blocks"] == 2048


# ── ContinuousBatchingConfig ───────────────────────────────────────────────

class TestContinuousBatchingConfig:
    def test_defaults(self):
        cfg = ContinuousBatchingConfig()
        assert cfg.max_num_seqs == 256

    def test_invalid_max_num_seqs(self):
        with pytest.raises(ValueError, match="max_num_seqs"):
            ContinuousBatchingConfig(max_num_seqs=0)

    def test_to_dict(self):
        cfg = ContinuousBatchingConfig(max_num_seqs=512, max_paddings=128)
        d = cfg.to_dict()
        assert d["max_num_seqs"] == 512


# ── InferenceOptConfig ─────────────────────────────────────────────────────

class TestInferenceOptConfig:
    def _make_kv(self):
        return KVCacheConfig(num_layers=32, num_heads=32, head_dim=128)

    def test_static_throughput_multiplier(self):
        cfg = InferenceOptConfig(
            attention=AttentionType.STANDARD,
            batching=BatchingStrategy.STATIC,
            kv_cache=self._make_kv(),
        )
        assert cfg.throughput_multiplier() == 1.0

    def test_dynamic_throughput_multiplier(self):
        cfg = InferenceOptConfig(
            attention=AttentionType.FLASH_ATTENTION,
            batching=BatchingStrategy.DYNAMIC,
            kv_cache=self._make_kv(),
        )
        assert cfg.throughput_multiplier() == 5.0

    def test_continuous_throughput_multiplier(self):
        cfg = InferenceOptConfig(
            attention=AttentionType.PAGED_ATTENTION,
            batching=BatchingStrategy.CONTINUOUS,
            kv_cache=self._make_kv(),
        )
        assert cfg.throughput_multiplier() == 15.0

    def test_to_dict_structure(self):
        cfg = InferenceOptConfig(
            attention=AttentionType.PAGED_ATTENTION,
            batching=BatchingStrategy.CONTINUOUS,
            kv_cache=self._make_kv(),
            paged=PagedAttentionConfig(),
            continuous=ContinuousBatchingConfig(),
        )
        d = cfg.to_dict()
        assert d["attention"] == "PAGED_ATTENTION"
        assert d["batching"] == "CONTINUOUS"
        assert d["paged"] is not None
        assert d["continuous"] is not None
        assert d["throughput_multiplier"] == 15.0

    def test_to_dict_optional_none(self):
        cfg = InferenceOptConfig(
            attention=AttentionType.STANDARD,
            batching=BatchingStrategy.STATIC,
            kv_cache=self._make_kv(),
        )
        d = cfg.to_dict()
        assert d["paged"] is None
        assert d["continuous"] is None
