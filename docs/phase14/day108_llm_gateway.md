# Day 108 — LLM Gateway Architecture: Model Routing, Quota Enforcement, Semantic Caching, Cost Governance

## WHY

Without a centralizing gateway, every team in an organization calls LLM provider APIs directly — there's no shared cost visibility, no quota enforcement preventing one team from exhausting a shared budget, and no de-duplication of semantically identical requests (huge waste for FAQ-style or templated traffic). A gateway sits between applications and the LLM provider(s) and owns four concerns:

1. **Model routing** — send cheap/simple queries to a cheap model and only escalate to an expensive model when complexity demands it.
2. **Quota enforcement** — cap usage per tenant/API-key so no single caller can blow the shared budget or rate limit.
3. **Semantic caching** — detect and reuse responses for queries that are semantically (not just textually) identical to a prior request.
4. **Cost governance** — aggregate spend across all callers and trigger alerts at configurable budget thresholds before the bill surprises anyone.

---

## HOW

`ModelRouter.route(complexity_score)` sorts its `ModelRoute`s by `max_complexity_score` ascending and returns the first (cheapest) route whose ceiling is still `>=` the query's complexity — i.e., escalate only as far as necessary, never further. `QuotaEnforcer` tracks per-key `usage` (`requests`, `tokens`) against a registered `QuotaConfig`; `is_over_quota()` checks both dimensions, since a key could exhaust its token budget well before its request-count budget or vice versa.

`SemanticCache` stores `(query_hash -> SemanticCacheEntry)` pairs; `get(query_hash, similarity)` returns the cached response only if the provided similarity score (computed upstream by an embedding comparison) meets the entry's own `embedding_similarity_threshold` — different cached entries can demand different confidence levels. `CostGovernor` accumulates `spent_today_usd` and reports which `alert_thresholds` (e.g. 50%, 80%, 100%) have been crossed via `triggered_alerts()`, with `is_over_budget()` as the hard stop signal.

---

## Class Diagram

```mermaid
classDiagram
    class ModelTier {
        <<enumeration>>
        CHEAP
        STANDARD
        PREMIUM
    }

    class ModelRoute {
        +ModelTier tier
        +str model_name
        +float cost_per_1k_tokens
        +float max_complexity_score
        +__post_init__()
        +to_dict() dict
    }

    class ModelRouter {
        +list~ModelRoute~ routes
        +__post_init__()
        +route(complexity_score) ModelRoute
        +to_dict() dict
    }

    class QuotaConfig {
        +str key_id
        +int max_requests_per_day
        +int max_tokens_per_day
        +__post_init__()
        +to_dict() dict
    }

    class QuotaEnforcer {
        +dict~str,QuotaConfig~ quotas
        +dict~str,dict~ usage
        +register(quota)
        +record_usage(key_id, tokens)
        +is_over_quota(key_id) bool
    }

    class SemanticCacheEntry {
        +str query_hash
        +str response
        +float embedding_similarity_threshold
        +__post_init__()
        +to_dict() dict
    }

    class SemanticCache {
        +dict~str,SemanticCacheEntry~ entries
        +float default_threshold
        +put(query_hash, response)
        +get(query_hash, similarity) str
        +hit_rate(hits, total) float
    }

    class CostGovernor {
        +float daily_budget_usd
        +float spent_today_usd
        +list~float~ alert_thresholds
        +__post_init__()
        +record_spend(amount_usd)
        +budget_utilization() float
        +triggered_alerts() list~float~
        +is_over_budget() bool
        +to_dict() dict
    }

    ModelRouter --> ModelRoute
    ModelRoute --> ModelTier
    QuotaEnforcer --> QuotaConfig
    SemanticCache --> SemanticCacheEntry
```

---

## Sequence Diagram — A Request Through the Gateway

```mermaid
sequenceDiagram
    participant Caller as Application (API key)
    participant GW as LLM Gateway
    participant Cache as SemanticCache
    participant Quota as QuotaEnforcer
    participant Router as ModelRouter
    participant LLM as Routed Model
    participant Gov as CostGovernor

    Caller->>GW: request(query, api_key)
    GW->>Quota: is_over_quota(api_key)
    alt over quota
        Quota-->>Caller: 429 Quota Exceeded
    else within quota
        GW->>Cache: get(query_hash, similarity)
        alt cache hit
            Cache-->>Caller: cached response (no LLM call, no cost)
        else cache miss
            GW->>Router: route(complexity_score)
            Router-->>GW: cheapest fitting ModelRoute
            GW->>LLM: generate(query) on routed model
            LLM-->>GW: response + token usage
            GW->>Cache: put(query_hash, response)
            GW->>Quota: record_usage(api_key, tokens)
            GW->>Gov: record_spend(cost)
            Gov->>Gov: triggered_alerts()
            alt is_over_budget()
                Gov-->>GW: halt further requests / alert finance
            end
            GW-->>Caller: response
        end
    end
```

---

## Key Takeaways

1. `ModelRouter.route()` always picks the **cheapest** route that still fits the complexity ceiling — escalation is monotonic and cost-minimizing.
2. `QuotaEnforcer.is_over_quota()` checks both request count and token count — either dimension can independently trip the limit.
3. `SemanticCache` keys are query hashes, but the threshold for a hit is per-entry (`embedding_similarity_threshold`), allowing different confidence bars for different cached answers.
4. `CostGovernor.triggered_alerts()` returns every threshold crossed, not just the highest — lets you distinguish "approaching budget" (50%/80%) from "over budget" (100%+) alerts.
5. A gateway turns four previously invisible problems (routing cost, quota abuse, redundant calls, runaway spend) into explicit, testable, governed logic.
