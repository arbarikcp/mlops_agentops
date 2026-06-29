# Day 55 — Testing ML: Unit, Data, Behavioral, and Training Smoke Tests

## The ML Testing Landscape

```mermaid
graph LR
    A[ML Testing] --> B[Unit Tests]
    A --> C[Data Contract Tests]
    A --> D[Behavioral Tests]
    A --> E[Training Smoke Tests]

    B --> B1["Pure functions\n(transforms, features)\nno I/O, fast"]
    C --> C1["Schema\nStats distribution\nLabel contract\nno model needed"]
    D --> D1["Invariants the model\nmust satisfy\n(fairness, monotonicity\nrobustness)"]
    E --> E1["100-row train\nconvergence check\nAUC guard\nreproducibility"]
```

---

## 1 — Unit Tests for ML Transforms

ML transforms are pure functions — given same input, same output. Test them like any pure function.

### What to test

| Transform | Test | Example |
|---|---|---|
| Bucketization | Bucket boundaries correct | `age=35` → `bucket="31-40"` |
| Log transform | Handles zero / negative | `log1p(0)=0`, `log1p(-1)` raises |
| One-hot encoder | Unknown category → zeros | `cat="unseen"` → `[0,0,0,0]` |
| Scaler | Fitted range not exceeded | `z-score` stays reasonable |
| PIT join | No future leakage | feature timestamp < prediction timestamp |

---

## 2 — Data Contract Tests

A **data contract** is a machine-readable spec for a dataset: column types, ranges, null rates, label distribution.

```mermaid
graph TD
    A[Incoming data batch] --> B[Schema check]
    B -->|fail| Z1[Block pipeline ❌]
    B -->|pass| C[Stats check PSI/KS]
    C -->|drift > 0.20| Z2[Alert + log ⚠️]
    C -->|ok| D[Label contract]
    D -->|label rate out of bounds| Z3[Alert ⚠️]
    D -->|ok| E[Data CI passed ✅]
```

### Schema contract

```python
EXPECTED_SCHEMA = {
    "age":         {"dtype": float, "min": 18, "max": 100, "null_rate": 0.0},
    "income":      {"dtype": float, "min": 0,  "max": None, "null_rate": 0.05},
    "loan_amount": {"dtype": float, "min": 100, "max": 500_000, "null_rate": 0.0},
    "default":     {"dtype": int,  "values": [0, 1], "null_rate": 0.0},
}
```

### Label contract

```python
LABEL_CONTRACT = {
    "column": "default",
    "min_positive_rate": 0.05,   # at least 5% defaults
    "max_positive_rate": 0.40,   # at most 40% defaults
}
```

---

## 3 — Behavioral Tests

Behavioral tests check **properties the model must satisfy** regardless of the dataset:

| Behavioral invariant | Failure means | Example assertion |
|---|---|---|
| Monotonicity | Higher risk feature → higher score | `score(income=0) > score(income=100k)` |
| Robustness | Small input noise → small score change | `Δscore < 0.05` for `Δinput 1%` |
| Directional | Increasing bad signal → increasing score | More derogatory marks → higher default score |
| Invariance | Protected attribute swap → same score | Score unchanged when `gender=M ↔ F` |
| Minimum confidence | Score not stuck at 0.5 | `stdev(scores) > 0.05` over test set |

```mermaid
graph LR
    subgraph "Behavioral Test Suite"
        M[Monotonicity]
        R[Robustness]
        D[Directional]
        I[Invariance]
        C[Confidence]
    end
    M --> Out[All must pass before merge]
    R --> Out
    D --> Out
    I --> Out
    C --> Out
```

---

## 4 — Training Smoke Tests

Goal: confirm training code is runnable and produces a valid artifact in **<5 seconds** using 100 synthetic rows.

```mermaid
sequenceDiagram
    participant CI
    participant Smoke as SmokeTrainer
    participant Model

    CI->>Smoke: run(n_rows=100, max_iter=10, seed=42)
    Smoke->>Smoke: generate synthetic data
    Smoke->>Model: fit(X_train, y_train)
    Model-->>Smoke: trained coefficients
    Smoke->>Smoke: score(X_val, y_val) → AUC
    Smoke->>Smoke: assert AUC > 0.5 (better than random)
    Smoke->>CI: SmokeTResult(passed=True, auc=0.XX, n_rows=100)
```

### Smoke test invariants

1. **No crash** — training runs to completion
2. **Better than random** — AUC > 0.5 on held-out 20%
3. **Reproducible** — same AUC for same seed across 2 runs (tolerance: ±0.001)
4. **Feature count matches** — model coefficients count = expected features

---

## 5 — AUC Regression Guard

The AUC guard prevents code changes from silently degrading model quality:

```
baseline_auc = load("artifacts/baseline_auc.json")     # saved from last good run
current_auc  = train_and_score(smoke_data, seed=42)

if current_auc < baseline_auc - TOLERANCE:
    FAIL  # regression detected
elif current_auc > baseline_auc:
    PASS + save new baseline
else:
    PASS  # within tolerance
```

**TOLERANCE = 0.01** — allows micro-regressions from refactors, blocks real degradation.

---

## Class Diagram

```mermaid
classDiagram
    class DataContractChecker {
        +schema: dict
        +label_column: str
        +min_positive_rate: float
        +max_positive_rate: float
        +check_schema(df) DataContractResult
        +check_label_dist(df) DataContractResult
        +check_null_rates(df) DataContractResult
        +run_all(df) list~DataContractResult~
    }

    class BehavioralChecker {
        +predict_fn: Callable
        +check_monotonicity(feature, low, high) BehavioralResult
        +check_robustness(X, noise_pct) BehavioralResult
        +check_directional(feature, direction) BehavioralResult
        +check_invariance(X, feature, value_a, value_b) BehavioralResult
    }

    class SmokeTrainer {
        +n_rows: int
        +n_features: int
        +max_iter: int
        +seed: int
        +run() SmokeResult
        +check_reproducibility() bool
    }

    class AUCGuard {
        +baseline_auc: float
        +tolerance: float
        +check(current_auc) AUCGuardResult
        +update_baseline(new_auc) None
    }

    DataContractChecker --> "0..*" DataContractResult
    BehavioralChecker --> "0..*" BehavioralResult
    SmokeTrainer --> SmokeResult
    AUCGuard --> AUCGuardResult
```
