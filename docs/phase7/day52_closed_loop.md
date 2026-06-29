# Day 52 — Closed-Loop Learning System (8 Steps)

## What Is a Closed Loop?

An open-loop ML system trains once and serves forever. A closed-loop system continuously
updates itself using feedback from real-world outcomes.

The loop has **8 discrete steps** that must run in order:

```mermaid
flowchart LR
    A[1 PREDICT] --> B[2 DECIDE]
    B --> C[3 LOG]
    C --> D[4 AWAIT_OUTCOME]
    D --> E[5 JOIN_LABEL]
    E --> F[6 RECOMPUTE]
    F --> G[7 TRIGGER]
    G --> H[8 APPROVE]
    H -->|deploy new model| A
```

---

## The 8 Steps

| Step | Class / Method | What happens |
|---|---|---|
| **PREDICT** | `ModelServingNode` | Model scores entity, returns probability |
| **DECIDE** | `DecisionNode` | Score → approve / review / decline via thresholds |
| **LOG** | `PredictionLogger.log()` | Entry written to JSONL with feature snapshot |
| **AWAIT_OUTCOME** | Time passes (30–180 days) | Outcome (default/no-default) arrives from external system |
| **JOIN_LABEL** | `GroundTruthJoiner.join()` | Predictions matched to confirmed outcomes |
| **RECOMPUTE** | `MetricRecomputer.recompute()` | AUC + approval rate recomputed on new labeled set |
| **TRIGGER** | `RetrainDecider.decide()` | Fire retrain if both batch size AND delta thresholds met |
| **APPROVE** | `LoopApprover.approve()` | Human or automated gate; updates baseline on approval |

---

## The Loop's Relationship to Phase 6

Phase 6 (Day 44) built the label feedback components:
- `GroundTruthJoiner` (Step 5)
- `MetricRecomputer` (Step 6)
- `RetrainDecider` + `LabelFeedbackLoop` (Step 7)

Day 52 wraps them into an **orchestrated ClosedLoop** that also handles:
- **Steps 1–3** via `ModelServingNode` + `PredictionLogger`
- **Step 8** — `LoopApprover` (human-in-the-loop gate)
- **State machine** — tracks which step the loop is currently in

---

## State Machine

```mermaid
stateDiagram-v2
    [*] --> PREDICT
    PREDICT --> DECIDE : score computed
    DECIDE --> LOG : decision made
    LOG --> AWAIT_OUTCOME : entry persisted
    AWAIT_OUTCOME --> JOIN_LABEL : outcomes arrived
    JOIN_LABEL --> RECOMPUTE : labels joined
    RECOMPUTE --> TRIGGER : metrics updated
    TRIGGER --> APPROVE : trigger fired
    TRIGGER --> AWAIT_OUTCOME : not enough data yet
    APPROVE --> PREDICT : new model deployed
    APPROVE --> AWAIT_OUTCOME : human rejected
```

---

## LoopApprover: Two Modes

| Mode | Description | When to use |
|---|---|---|
| **AUTO** | Approves automatically if AUC improved | Mature pipelines with stable data |
| **HUMAN** | Returns PENDING — waits for external signal | Regulated environments (e.g., credit) |
| **BLOCK** | Always rejects (for testing) | CI pipeline dry-run |

---

## Closed Loop Class Diagram

```mermaid
classDiagram
    class ApprovalMode {
        <<enumeration>>
        AUTO
        HUMAN
        BLOCK
    }

    class ApprovalResult {
        +approved: bool
        +mode: ApprovalMode
        +reason: str
        +baseline_updated: bool
    }

    class LoopApprover {
        +mode: ApprovalMode
        +min_auc_improvement: float
        +approve(trigger, current_metrics, baseline_metrics) ApprovalResult
    }

    class ClosedLoopState {
        +current_step: LoopPhase
        +n_predictions: int
        +n_labeled: int
        +last_auc: float
        +last_trigger_ts: datetime
    }

    class ClosedLoop {
        +logger: PredictionLogger
        +joiner: GroundTruthJoiner
        +recomputer: MetricRecomputer
        +decider: RetrainDecider
        +approver: LoopApprover
        +state: ClosedLoopState
        +serve_and_log(entity_key, score, decision, features) PredictionLogEntry
        +tick(outcomes) LoopResult
        +get_state() ClosedLoopState
    }

    ClosedLoop --> LoopApprover
    ClosedLoop --> ClosedLoopState
    ClosedLoop --> ApprovalResult
    LoopApprover --> ApprovalMode
    LoopApprover --> ApprovalResult
```

---

## Monitoring Integration

The closed loop emits metrics to `MLMetricsCollector` at each tick:

```python
metrics.record_auc(result.current_metrics["auc"])
metrics.record_approval_rate(result.current_metrics["approval_rate"])
```

This means Grafana always shows the current loop state without additional polling.
