# Day 35 — Model Validation Gate: Thresholds, Champion/Challenger, Auto-Promote

## Why Model Validation is a Gate

A data gate (Day 34) prevents bad data from reaching the training step.
A **model gate** prevents a bad model from reaching the serving endpoint.

Without a model gate:
- A model trained on drifted data gets promoted automatically
- A model that is worse than the current champion gets deployed
- Fairness violations (slice AUC gap > threshold) reach production silently

With a model gate:
- Every candidate model passes the same criteria as the current champion
- Champion/challenger comparison is explicit and audited
- Promotion is a DAG step, not an implicit side-effect

---

## Metric Hierarchy

Not all metrics are equal. The model gate enforces a priority order:

```mermaid
graph TD
    M1["1. Data contract\n(must pass before training)"]
    M2["2. Minimum AUC threshold\n(absolute floor)"]
    M3["3. Calibration ECE\n(probability reliability)"]
    M4["4. Slice gap\n(fairness — AUC gap across demographics)"]
    M5["5. Champion comparison\n(challenger must beat champion)"]
    M6["6. Cost at threshold\n(business cost ≤ current champion)"]

    M1 --> M2 --> M3 --> M4 --> M5 --> M6
    M6 -->|ALL PASS| PROMOTE["✅ Promote to champion"]
    M2 & M3 & M4 & M5 & M6 -->|ANY FAIL| REJECT["❌ Reject + alert"]
```

---

## Champion / Challenger Framework

```
Champion:   current production model (the baseline)
Challenger: newly trained candidate model

Rules:
  1. Challenger must pass ALL hard gates (AUC threshold, calibration, fairness)
  2. Challenger must beat champion on the primary metric (AUC) by at least delta
  3. If champion doesn't exist yet, ANY model passing hard gates is promoted
  4. If challenger loses, champion stays; challenger is logged to registry as "rejected"
```

### Delta threshold

The "delta" prevents churning models for negligible improvements:

```
AUC champion = 0.780
AUC challenger = 0.783
delta = 0.005 (minimum improvement required)

challenger AUC gain = 0.003 < 0.005 → REJECT (not better enough)
```

Why delta matters: deploying a new model has cost (rollout risk, cache invalidation, retraining feature store). Marginal improvement doesn't justify this cost.

---

## Champion / Challenger Sequence

```mermaid
sequenceDiagram
    participant Pipeline
    participant Gate as ModelGate
    participant Registry as ModelRegistry
    participant Alert

    Pipeline->>Gate: evaluate(challenger_model, X_test, y_test)

    Gate->>Gate: check AUC >= 0.75
    Gate->>Gate: check ECE <= 0.05
    Gate->>Gate: check slice gap <= 0.10

    alt Hard gates failed
        Gate->>Alert: alert(reason, metrics)
        Gate-->>Pipeline: ModelGateReport(passed=False)
        Note over Pipeline: STOP — no promotion
    end

    Gate->>Registry: get_champion_metrics()
    Registry-->>Gate: champion AUC = 0.780

    Gate->>Gate: challenger_auc - champion_auc >= delta (0.005)?

    alt Challenger wins
        Gate->>Registry: promote(challenger, version="v3")
        Gate->>Registry: archive(champion, version="v2", status="previous_stable")
        Gate-->>Pipeline: ModelGateReport(passed=True, promoted=True)
    else Champion retained
        Gate->>Registry: register(challenger, status="rejected")
        Gate-->>Pipeline: ModelGateReport(passed=False, reason="delta not met")
    end
```

---

## Auto-Promote Logic

Auto-promote is conditional, not unconditional:

```python
def auto_promote(challenger, champion):
    # Hard gates: must pass regardless of champion
    assert challenger.auc >= AUC_THRESHOLD
    assert challenger.ece <= ECE_THRESHOLD
    assert challenger.slice_gap <= SLICE_GAP_THRESHOLD

    # Champion comparison: challenger must be better
    if champion is None:
        return promote(challenger)   # first model ever

    delta = challenger.auc - champion.auc
    if delta >= DELTA_THRESHOLD:
        return promote(challenger)
    else:
        return reject(challenger, reason=f"delta={delta:.4f} < {DELTA_THRESHOLD}")
```

---

## Model Gate Class Diagram

```mermaid
classDiagram
    class ModelMetrics {
        +auc: float
        +ece: float
        +brier: float
        +slice_auc_gap: float
        +cost_at_threshold: float
        +n_test: int
        +model_version: str
        +to_dict() dict
    }

    class GateThresholds {
        +min_auc: float
        +max_ece: float
        +max_slice_gap: float
        +champion_delta: float
        +max_cost: float | None
        +from_env() GateThresholds
    }

    class ModelGateReport {
        +passed: bool
        +promoted: bool
        +challenger_metrics: ModelMetrics
        +champion_metrics: ModelMetrics | None
        +gate_failures: list~str~
        +promotion_reason: str | None
        +rejection_reason: str | None
    }

    class ChampionRegistry {
        +current_champion: ModelMetrics | None
        +history: list~ModelMetrics~
        +get_champion() ModelMetrics | None
        +promote(metrics, model) None
        +reject(metrics, reason) None
    }

    class ModelGate {
        +thresholds: GateThresholds
        +registry: ChampionRegistry
        +evaluate(model, X_test, y_test) ModelGateReport
        +_hard_gates(metrics) list~str~
        +_champion_comparison(metrics) list~str~
    }

    ModelGate --> GateThresholds
    ModelGate --> ChampionRegistry
    ModelGate --> ModelGateReport
    ModelGateReport --> ModelMetrics
    ChampionRegistry --> ModelMetrics
```

---

## Rollback Strategy

When a promoted model misbehaves in production:

```
champion_model (v3) → shows AUC regression in production monitoring
         │
         ▼
ModelGate.rollback()
         │
         ├── previous_stable = ChampionRegistry.history[-2]  (v2)
         ├── promote(v2) as champion
         └── tag v3 as "rolled_back"

Target: rollback in < 7 minutes (K8s `kubectl rollout undo`)
```

---

## Key Invariants

1. **Hard gates are absolute** — AUC threshold is never relaxed for champion comparison.
2. **Champion comparison requires delta** — prevents churning models for noise.
3. **No champion means any passing model is promoted** — first run bootstraps cleanly.
4. **Rejection is logged as a versioned artifact** — "rejected" models are auditable.
5. **Rollback is always available** — `previous_stable` is always stored in the registry.
