"""
llm_gateway.py — LLM Gateway Architecture: Model Routing, Quota
Enforcement, Semantic Caching, Cost Governance (Day 108)

Covers a gateway sitting between application and multiple LLM
providers/models: complexity-based model routing (cheap -> expensive
escalation), per-tenant quota enforcement, semantic caching of
similar queries, and aggregate cost governance with budget alerts.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ModelTier(str, Enum):
    """Cost/capability tiers for routed models."""

    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"


@dataclass
class ModelRoute:
    """A candidate route: a model tier with a complexity ceiling and price."""

    tier: ModelTier
    model_name: str
    cost_per_1k_tokens: float
    max_complexity_score: float = 1.0

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty")
        if self.cost_per_1k_tokens < 0:
            raise ValueError("cost_per_1k_tokens must be >= 0")
        if not (0 <= self.max_complexity_score <= 1):
            raise ValueError("max_complexity_score must be in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "model_name": self.model_name,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "max_complexity_score": self.max_complexity_score,
        }


@dataclass
class ModelRouter:
    """Routes a query to the cheapest model whose complexity ceiling fits."""

    routes: list[ModelRoute]

    def __post_init__(self) -> None:
        if not self.routes:
            raise ValueError("routes must be non-empty")

    def route(self, complexity_score: float) -> ModelRoute:
        candidates = sorted(self.routes, key=lambda r: r.max_complexity_score)
        for r in candidates:
            if r.max_complexity_score >= complexity_score:
                return r
        raise ValueError(
            f"No route found for complexity_score={complexity_score}"
        )

    def to_dict(self) -> dict:
        return {"routes": [r.to_dict() for r in self.routes]}


@dataclass
class QuotaConfig:
    """Per-key daily quota configuration."""

    key_id: str
    max_requests_per_day: int
    max_tokens_per_day: int

    def __post_init__(self) -> None:
        if not self.key_id:
            raise ValueError("key_id must be non-empty")
        if self.max_requests_per_day <= 0:
            raise ValueError("max_requests_per_day must be > 0")
        if self.max_tokens_per_day <= 0:
            raise ValueError("max_tokens_per_day must be > 0")

    def to_dict(self) -> dict:
        return {
            "key_id": self.key_id,
            "max_requests_per_day": self.max_requests_per_day,
            "max_tokens_per_day": self.max_tokens_per_day,
        }


@dataclass
class QuotaEnforcer:
    """Tracks per-key usage against registered quotas."""

    quotas: dict[str, QuotaConfig] = field(default_factory=dict)
    usage: dict[str, dict] = field(default_factory=dict)

    def register(self, quota: QuotaConfig) -> None:
        self.quotas[quota.key_id] = quota
        self.usage.setdefault(quota.key_id, {"requests": 0, "tokens": 0})

    def record_usage(self, key_id: str, tokens: int) -> None:
        if key_id not in self.quotas:
            raise KeyError(f"key_id {key_id!r} not registered")
        self.usage[key_id]["requests"] += 1
        self.usage[key_id]["tokens"] += tokens

    def is_over_quota(self, key_id: str) -> bool:
        if key_id not in self.quotas:
            raise KeyError(f"key_id {key_id!r} not registered")
        quota = self.quotas[key_id]
        used = self.usage.get(key_id, {"requests": 0, "tokens": 0})
        return (
            used["requests"] > quota.max_requests_per_day
            or used["tokens"] > quota.max_tokens_per_day
        )


@dataclass
class SemanticCacheEntry:
    """A cached (query_hash -> response) entry with a similarity threshold."""

    query_hash: str
    response: str
    embedding_similarity_threshold: float = 0.95

    def __post_init__(self) -> None:
        if not self.query_hash:
            raise ValueError("query_hash must be non-empty")
        if not self.response:
            raise ValueError("response must be non-empty")
        if not (0 < self.embedding_similarity_threshold <= 1):
            raise ValueError(
                "embedding_similarity_threshold must be in (0, 1]"
            )

    def to_dict(self) -> dict:
        return {
            "query_hash": self.query_hash,
            "response": self.response,
            "embedding_similarity_threshold": self.embedding_similarity_threshold,
        }


@dataclass
class SemanticCache:
    """Cache that returns a stored response if similarity exceeds threshold."""

    entries: dict[str, SemanticCacheEntry] = field(default_factory=dict)
    default_threshold: float = 0.95

    def put(self, query_hash: str, response: str) -> None:
        self.entries[query_hash] = SemanticCacheEntry(
            query_hash=query_hash,
            response=response,
            embedding_similarity_threshold=self.default_threshold,
        )

    def get(self, query_hash: str, similarity: float) -> str | None:
        entry = self.entries.get(query_hash)
        if entry is None:
            return None
        if similarity >= entry.embedding_similarity_threshold:
            return entry.response
        return None

    def hit_rate(self, hits: int, total: int) -> float:
        return hits / total if total else 0.0


@dataclass
class CostGovernor:
    """Tracks daily spend against a budget and fires threshold alerts."""

    daily_budget_usd: float
    spent_today_usd: float = 0.0
    alert_thresholds: list[float] = field(
        default_factory=lambda: [0.5, 0.8, 1.0]
    )

    def __post_init__(self) -> None:
        if self.daily_budget_usd <= 0:
            raise ValueError("daily_budget_usd must be > 0")
        for t in self.alert_thresholds:
            if not (0 < t <= 2):
                raise ValueError("alert_thresholds must be in (0, 2]")

    def record_spend(self, amount_usd: float) -> None:
        if amount_usd < 0:
            raise ValueError("amount_usd must be >= 0")
        self.spent_today_usd += amount_usd

    def budget_utilization(self) -> float:
        return self.spent_today_usd / self.daily_budget_usd

    def triggered_alerts(self) -> list[float]:
        util = self.budget_utilization()
        return [t for t in self.alert_thresholds if t <= util]

    def is_over_budget(self) -> bool:
        return self.budget_utilization() > 1.0

    def to_dict(self) -> dict:
        return {
            "daily_budget_usd": self.daily_budget_usd,
            "spent_today_usd": self.spent_today_usd,
            "alert_thresholds": self.alert_thresholds,
            "budget_utilization": self.budget_utilization(),
            "triggered_alerts": self.triggered_alerts(),
            "is_over_budget": self.is_over_budget(),
        }
