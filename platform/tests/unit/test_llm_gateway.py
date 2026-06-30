"""Unit tests for platform/llm/llm_gateway.py (Day 108)."""

import pytest
from llm.llm_gateway import (
    CostGovernor,
    ModelRoute,
    ModelRouter,
    ModelTier,
    QuotaConfig,
    QuotaEnforcer,
    SemanticCache,
    SemanticCacheEntry,
)


class TestModelRoute:
    def test_basic(self):
        r = ModelRoute(tier=ModelTier.CHEAP, model_name="gpt-3.5", cost_per_1k_tokens=0.001)
        assert r.max_complexity_score == 1.0

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            ModelRoute(tier=ModelTier.CHEAP, model_name="", cost_per_1k_tokens=0.01)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_per_1k_tokens"):
            ModelRoute(tier=ModelTier.CHEAP, model_name="x", cost_per_1k_tokens=-1)

    def test_invalid_complexity_raises(self):
        with pytest.raises(ValueError, match="max_complexity_score"):
            ModelRoute(tier=ModelTier.CHEAP, model_name="x", cost_per_1k_tokens=0.1, max_complexity_score=1.5)


class TestModelRouter:
    def test_empty_routes_raises(self):
        with pytest.raises(ValueError, match="routes"):
            ModelRouter(routes=[])

    def test_route_picks_cheapest_fitting(self):
        cheap = ModelRoute(tier=ModelTier.CHEAP, model_name="cheap-m", cost_per_1k_tokens=0.001, max_complexity_score=0.3)
        standard = ModelRoute(tier=ModelTier.STANDARD, model_name="std-m", cost_per_1k_tokens=0.01, max_complexity_score=0.7)
        premium = ModelRoute(tier=ModelTier.PREMIUM, model_name="prem-m", cost_per_1k_tokens=0.05, max_complexity_score=1.0)
        router = ModelRouter(routes=[premium, cheap, standard])
        chosen = router.route(0.5)
        assert chosen.model_name == "std-m"

    def test_route_picks_cheap_for_low_complexity(self):
        cheap = ModelRoute(tier=ModelTier.CHEAP, model_name="cheap-m", cost_per_1k_tokens=0.001, max_complexity_score=0.3)
        premium = ModelRoute(tier=ModelTier.PREMIUM, model_name="prem-m", cost_per_1k_tokens=0.05, max_complexity_score=1.0)
        router = ModelRouter(routes=[premium, cheap])
        assert router.route(0.1).model_name == "cheap-m"

    def test_route_raises_when_no_fit(self):
        cheap = ModelRoute(tier=ModelTier.CHEAP, model_name="cheap-m", cost_per_1k_tokens=0.001, max_complexity_score=0.3)
        router = ModelRouter(routes=[cheap])
        with pytest.raises(ValueError):
            router.route(0.9)

    def test_to_dict(self):
        cheap = ModelRoute(tier=ModelTier.CHEAP, model_name="cheap-m", cost_per_1k_tokens=0.001)
        router = ModelRouter(routes=[cheap])
        assert "routes" in router.to_dict()


class TestQuotaConfig:
    def test_basic(self):
        q = QuotaConfig(key_id="k1", max_requests_per_day=1000, max_tokens_per_day=100000)
        assert q.key_id == "k1"

    def test_empty_key_raises(self):
        with pytest.raises(ValueError, match="key_id"):
            QuotaConfig(key_id="", max_requests_per_day=1, max_tokens_per_day=1)

    def test_invalid_max_requests_raises(self):
        with pytest.raises(ValueError, match="max_requests_per_day"):
            QuotaConfig(key_id="k1", max_requests_per_day=0, max_tokens_per_day=1)

    def test_invalid_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens_per_day"):
            QuotaConfig(key_id="k1", max_requests_per_day=1, max_tokens_per_day=0)


class TestQuotaEnforcer:
    def test_record_usage_and_check(self):
        enforcer = QuotaEnforcer()
        enforcer.register(QuotaConfig(key_id="k1", max_requests_per_day=2, max_tokens_per_day=1000))
        enforcer.record_usage("k1", 100)
        assert enforcer.is_over_quota("k1") is False
        enforcer.record_usage("k1", 100)
        enforcer.record_usage("k1", 100)
        assert enforcer.is_over_quota("k1") is True

    def test_record_usage_unregistered_raises(self):
        enforcer = QuotaEnforcer()
        with pytest.raises(KeyError):
            enforcer.record_usage("missing", 10)

    def test_is_over_quota_unregistered_raises(self):
        enforcer = QuotaEnforcer()
        with pytest.raises(KeyError):
            enforcer.is_over_quota("missing")

    def test_over_token_quota(self):
        enforcer = QuotaEnforcer()
        enforcer.register(QuotaConfig(key_id="k1", max_requests_per_day=100, max_tokens_per_day=50))
        enforcer.record_usage("k1", 100)
        assert enforcer.is_over_quota("k1") is True


class TestSemanticCacheEntry:
    def test_basic(self):
        e = SemanticCacheEntry(query_hash="h1", response="hi")
        assert e.embedding_similarity_threshold == 0.95

    def test_empty_query_hash_raises(self):
        with pytest.raises(ValueError, match="query_hash"):
            SemanticCacheEntry(query_hash="", response="x")

    def test_empty_response_raises(self):
        with pytest.raises(ValueError, match="response"):
            SemanticCacheEntry(query_hash="h1", response="")

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError, match="embedding_similarity_threshold"):
            SemanticCacheEntry(query_hash="h1", response="x", embedding_similarity_threshold=1.5)


class TestSemanticCache:
    def test_put_and_get_hit(self):
        cache = SemanticCache()
        cache.put("h1", "cached response")
        assert cache.get("h1", 0.99) == "cached response"

    def test_get_below_threshold_miss(self):
        cache = SemanticCache()
        cache.put("h1", "cached response")
        assert cache.get("h1", 0.5) is None

    def test_get_unknown_hash_returns_none(self):
        cache = SemanticCache()
        assert cache.get("missing", 1.0) is None

    def test_hit_rate(self):
        cache = SemanticCache()
        assert cache.hit_rate(5, 10) == 0.5

    def test_hit_rate_zero_total(self):
        cache = SemanticCache()
        assert cache.hit_rate(0, 0) == 0.0


class TestCostGovernor:
    def test_record_spend_and_utilization(self):
        gov = CostGovernor(daily_budget_usd=100)
        gov.record_spend(50)
        assert gov.budget_utilization() == 0.5

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError, match="daily_budget_usd"):
            CostGovernor(daily_budget_usd=0)

    def test_negative_spend_raises(self):
        gov = CostGovernor(daily_budget_usd=100)
        with pytest.raises(ValueError, match="amount_usd"):
            gov.record_spend(-1)

    def test_triggered_alerts(self):
        gov = CostGovernor(daily_budget_usd=100)
        gov.record_spend(85)
        alerts = gov.triggered_alerts()
        assert 0.5 in alerts
        assert 0.8 in alerts
        assert 1.0 not in alerts

    def test_is_over_budget(self):
        gov = CostGovernor(daily_budget_usd=100)
        gov.record_spend(150)
        assert gov.is_over_budget() is True

    def test_invalid_alert_thresholds_raises(self):
        with pytest.raises(ValueError, match="alert_thresholds"):
            CostGovernor(daily_budget_usd=100, alert_thresholds=[3.0])

    def test_to_dict(self):
        gov = CostGovernor(daily_budget_usd=100)
        d = gov.to_dict()
        assert "is_over_budget" in d
