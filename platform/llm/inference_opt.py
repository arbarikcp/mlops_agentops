"""
inference_opt.py — Inference Optimization Configuration (Day 94)

Covers KV cache sizing, PagedAttention, and continuous batching
for high-throughput LLM inference.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AttentionType(str, Enum):
    """Attention implementation for LLM inference."""

    STANDARD = "STANDARD"
    FLASH_ATTENTION = "FLASH_ATTENTION"
    PAGED_ATTENTION = "PAGED_ATTENTION"


class BatchingStrategy(str, Enum):
    """Request batching strategy for LLM serving."""

    STATIC = "STATIC"
    DYNAMIC = "DYNAMIC"
    CONTINUOUS = "CONTINUOUS"


@dataclass
class KVCacheConfig:
    """KV cache memory configuration."""

    num_layers: int
    num_heads: int
    head_dim: int
    dtype_bytes: int = 2          # FP16 = 2 bytes
    max_seq_len: int = 4096

    def __post_init__(self) -> None:
        for name, val in [
            ("num_layers", self.num_layers),
            ("num_heads", self.num_heads),
            ("head_dim", self.head_dim),
            ("dtype_bytes", self.dtype_bytes),
            ("max_seq_len", self.max_seq_len),
        ]:
            if val <= 0:
                raise ValueError(f"{name} must be > 0, got {val}")

    def cache_size_gb(self, batch_size: int) -> float:
        """KV cache size in GB for given batch size.

        Formula: 2 * num_layers * num_heads * head_dim * max_seq_len * batch_size * dtype_bytes
        (factor 2 for K and V).
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        total_bytes = (
            2
            * self.num_layers
            * self.num_heads
            * self.head_dim
            * self.max_seq_len
            * batch_size
            * self.dtype_bytes
        )
        return total_bytes / 1e9

    def to_dict(self) -> dict:
        return {
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "dtype_bytes": self.dtype_bytes,
            "max_seq_len": self.max_seq_len,
        }


@dataclass
class PagedAttentionConfig:
    """PagedAttention memory management configuration (vLLM-style)."""

    block_size: int = 16
    max_num_blocks: int = 4096
    gpu_memory_utilization: float = 0.9

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if not (0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )

    def max_concurrent_seqs(self, kv_cfg: KVCacheConfig) -> int:
        """Maximum concurrent sequences given a KV cache config."""
        total_tokens = self.max_num_blocks * self.block_size
        return total_tokens // kv_cfg.max_seq_len

    def to_dict(self) -> dict:
        return {
            "block_size": self.block_size,
            "max_num_blocks": self.max_num_blocks,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }


@dataclass
class ContinuousBatchingConfig:
    """Continuous batching scheduler configuration."""

    max_num_seqs: int = 256
    max_paddings: int = 256
    scheduler_delay_factor: float = 0.0

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {self.max_num_seqs}")

    def to_dict(self) -> dict:
        return {
            "max_num_seqs": self.max_num_seqs,
            "max_paddings": self.max_paddings,
            "scheduler_delay_factor": self.scheduler_delay_factor,
        }


@dataclass
class InferenceOptConfig:
    """Combined inference optimization configuration."""

    attention: AttentionType
    batching: BatchingStrategy
    kv_cache: KVCacheConfig
    paged: PagedAttentionConfig | None = None
    continuous: ContinuousBatchingConfig | None = None

    def throughput_multiplier(self) -> float:
        """Estimated throughput multiplier relative to single-request naive inference."""
        if self.batching == BatchingStrategy.STATIC:
            return 1.0
        if self.batching == BatchingStrategy.DYNAMIC:
            return 5.0
        return 15.0  # CONTINUOUS

    def to_dict(self) -> dict:
        return {
            "attention": self.attention.value,
            "batching": self.batching.value,
            "kv_cache": self.kv_cache.to_dict(),
            "paged": self.paged.to_dict() if self.paged else None,
            "continuous": self.continuous.to_dict() if self.continuous else None,
            "throughput_multiplier": self.throughput_multiplier(),
        }
