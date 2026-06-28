# Day 4 — ML System & Product Design

> Tags: `[T]` theory · `[NEW]`  
> Deliverable: **Completed system design doc** for the credit-risk platform (fills the project charter table)

---

## 1. Why Product Design Comes Before Code

ML engineers who skip this step build technically correct systems that answer the wrong question.

```mermaid
flowchart LR
    A["❌ Skip design\n→ optimise AUC\n→ ship\n→ business says\n'this is useless'"] 
    B["✅ Design first\n→ model FP/FN cost\n→ tune threshold\n→ ship\n→ business outcomes improve"]
    A -.->|"what actually happens"| B
```

**The question to answer before any model:** _What decision does this system support, and what is the cost of each type of error?_

---

## 2. System Overview: Credit-Risk Platform

### 2.1 Decision Supported

> **Approve / Review / Decline** a credit card application in real-time.

- **Approve:** System is confident the applicant will repay.
- **Review:** System is uncertain → route to human underwriter.
- **Decline:** System predicts high default probability.

### 2.2 Users and Consumers

```mermaid
flowchart TD
    APP[Applicant] -->|submits application| GW[API Gateway]
    GW --> SERVE[Risk Scoring API]
    SERVE -->|approve| BANK[Banking Core System]
    SERVE -->|review| UW[Underwriting Queue]
    SERVE -->|decline| BANK
    BANK --> APP

    SERVE -.->|"audit log"| AUDIT[Compliance Reporting]
    UW -.->|"human decision"| LABEL[Ground Truth Pipeline]
    LABEL -.->|"30-90 day delay"| RETRAIN[CT Trigger]
```

| Consumer | What they need | SLA |
|---|---|---|
| Banking Core System | Binary approve/decline + score | p95 < 200 ms |
| Underwriting team | Score + top feature explanations | p95 < 500 ms |
| Compliance team | Audit trail per decision | T+1 batch |
| Monitoring system | Per-prediction log with features | Real-time |

---

## 3. FP vs FN Cost Analysis

This is the most important design decision for a credit model. **AUC alone does not drive a business decision — cost does.**

| Error Type | What Happens | Financial Impact |
|---|---|---|
| **False Positive (FP)** | Good customer declined | Lost revenue (LTV ~$2,000/year) + customer churn |
| **False Negative (FN)** | Bad customer approved | Default loss (~$8,000 average) + regulatory risk |

**FN cost ≫ FP cost** → the model should be conservative (higher precision, lower recall on approvals).

```mermaid
xychart-beta
    title "Expected Loss vs Decision Threshold"
    x-axis "Threshold" [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y-axis "Expected Loss ($K)" 0 --> 500
    line [480, 320, 200, 130, 100, 110, 150, 220, 380]
```

The optimal threshold minimises **total expected loss = (FP_count × FP_cost) + (FN_count × FN_cost)**.

We will compute this empirically in **Day 16** (Phase 2: Calibration & Thresholds).

---

## 4. Latency Budget

```mermaid
sequenceDiagram
    participant Client
    participant GW as API Gateway
    participant FS as Feature Store
    participant MODEL as Model Server
    participant EXPL as Explainer

    Note over Client,EXPL: Total budget: 200ms (p95)

    Client->>GW: POST /score (10ms network)
    GW->>FS: feature lookup (20ms p95)
    FS-->>GW: feature vector
    GW->>MODEL: infer (15ms p95)
    MODEL-->>GW: score + confidence
    GW->>EXPL: SHAP explain (30ms, async, optional)
    GW-->>Client: response (total ~55ms median, <200ms p95)
```

| Component | Budget (p95) | Notes |
|---|---|---|
| Feature store lookup | 20 ms | Redis online store |
| Model inference | 15 ms | Tabular model, fast |
| SHAP explanation | 30 ms | Only for "review" decisions |
| Network + GW | 10 ms | Local; 30ms on cloud |
| **Total** | **< 200 ms** | Hard SLO for online serving |

Batch inference: nightly, no latency SLO, throughput-optimised.

---

## 5. Rollback Behavior

```mermaid
stateDiagram-v2
    [*] --> Challenger: new model promoted to staging
    Challenger --> ShadowMode: traffic mirror (no live decisions)
    ShadowMode --> Canary: metrics OK (5% live traffic)
    Canary --> Champion: metrics OK (100% live traffic)
    Canary --> AutoRollback: gate failure detected
    Champion --> AutoRollback: drift / quality decay detected
    AutoRollback --> Champion: revert to previous alias
    AutoRollback --> [*]: alert fired, human reviews
```

**MLflow alias strategy:**
- `champion` → current production model
- `challenger` → staging candidate
- `shadow` → shadow-mode candidate

Auto-revert fires when: approval rate drops >5%, default rate rises >2%, or p95 latency breaches 200ms SLO.

---

## 6. Late Labels & Ground Truth Timeline

Credit decisions have **delayed ground truth** — default/repayment signal arrives 30–90 days after the decision.

```mermaid
gantt
    title Label Arrival Timeline
    dateFormat  YYYY-MM-DD
    section Application
    Application submitted    :a1, 2024-01-01, 1d
    Decision made (score)    :a2, after a1, 1d
    section Label
    First payment due        :l1, 2024-02-01, 1d
    Delinquency flagged      :l2, 2024-04-01, 1d
    Default confirmed        :l3, 2024-07-01, 1d
    section Pipeline
    Label joins feature log  :p1, after l3, 7d
    Retraining triggered     :p2, after p1, 1d
```

**Implications:**
- We cannot retrain daily on recent data — labels don't exist yet.
- Need **label contracts** (Day 20) defining arrival expectations.
- Retraining is triggered on a cadence + label-coverage threshold.
- Must handle **label corrections** (delinquency that later recovered).

---

## 7. Minimum Viable Monitoring

From the Monitoring gate requirement: detect **operational, ML quality, and business outcomes separately**.

```mermaid
flowchart TD
    subgraph "Operational Monitoring"
        O1["p95/p99 latency"]
        O2["Error rate (5xx)"]
        O3["Feature store miss rate"]
        O4["Model server CPU/GPU"]
    end

    subgraph "ML Monitoring"
        M1["Feature drift (PSI on top-10 features)"]
        M2["Prediction score distribution shift"]
        M3["Confidence calibration"]
        M4["Slice-level performance (region/income)"]
    end

    subgraph "Business Monitoring"
        B1["Approval rate (daily)"]
        B2["Estimated default rate"]
        B3["Human review queue depth"]
        B4["Label-delayed actual default rate"]
    end

    O1 & O2 & O3 & O4 --> PROM["Prometheus"]
    M1 & M2 & M3 & M4 --> EV["Evidently"]
    B1 & B2 & B3 & B4 --> DASH["Business Dashboard"]

    PROM & EV & DASH --> GRAF["Grafana (unified view)"]
```

**Alerting thresholds (initial):**

| Metric | Alert threshold | Severity |
|---|---|---|
| p95 latency | > 200 ms | Critical |
| Feature drift (PSI) | > 0.2 on any top-10 feature | Warning |
| Approval rate delta | > ±5% vs 7-day average | Warning |
| Score distribution shift | KS statistic > 0.1 | Warning |
| 5xx error rate | > 1% over 5 minutes | Critical |

---

## 8. Risk Matrix (System Design Perspective)

```mermaid
quadrantChart
    title System Design Risks
    x-axis Low Severity --> High Severity
    y-axis Low Probability --> High Probability

    quadrant-1 Watch
    quadrant-2 Mitigate Now
    quadrant-3 Accept
    quadrant-4 Reduce Probability

    Late Label Delay: [0.85, 0.75]
    Feature Store Outage: [0.45, 0.80]
    Threshold Miscalibration: [0.60, 0.90]
    Model Staleness: [0.80, 0.70]
    Batch/Online Skew: [0.70, 0.85]
    Regulatory Non-Compliance: [0.25, 0.95]
    Data Schema Change: [0.75, 0.60]
```

---

## 9. Summary: System Design Decision Table

| Field | Decision |
|---|---|
| **Decision supported** | Approve / Review / Decline a credit application |
| **Primary consumers** | Banking core, underwriting queue, compliance |
| **FP cost** | ~$2,000 (lost LTV) — bad but recoverable |
| **FN cost** | ~$8,000 average default + regulatory risk — dominant |
| **Threshold strategy** | Minimise total expected loss; human review band |
| **Latency budget** | p95 < 200 ms online; batch nightly |
| **Rollback behavior** | Auto-revert to previous registry alias on gate failure |
| **Late labels** | Default signal arrives 30–90 days after decision |
| **Label correction** | Handle delinquency recoveries; versioned ground truth |
| **Minimum monitoring** | Drift on top-10 features + p95 latency + approval rate |

---

## Key Takeaways

- **AUC is not a business metric.** Cost-sensitive evaluation (Day 16) produces the real threshold.
- **FN >> FP** for credit risk → conservative model is the right default.
- **Late labels are not an edge case** — they define your retraining cadence.
- **Rollback must be automatic.** A human reviewing a gate failure at 3 AM will be slow.
- **Monitor business outcomes separately** from ML metrics and infra metrics — they have different owners and different alert paths.
