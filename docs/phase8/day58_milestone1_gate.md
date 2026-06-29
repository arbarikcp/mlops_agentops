# Day 58 — Consolidation + Milestone 1 Gate

## What Milestone 1 Proves

> Given a **prediction**, you can trace the model version, data version, code version,
> feature values, request ID, and decision outcome — and you can roll back, retry a
> failed job safely, and detect drift/quality/infra/business issues separately.

This is the end of **Era A — Classical MLOps**. Everything built in Phases 0–8 now
fits together into one auditable, reproducible, monitorable system.

---

## The Six Production Gates

```mermaid
graph TD
    G1[Gate 1: Reproducibility<br/>Trace model + data + code + env from run ID]
    G2[Gate 2: Serving<br/>Deploy, rollback, load-test, explain p95/p99]
    G3[Gate 3: Pipeline<br/>Failed job retries safely without corrupting artifacts]
    G4[Gate 4: Monitoring<br/>Detect drift, quality decay, infra errors, business outcomes separately]
    G5[Gate 5: Security<br/>Threat model, permissions, secrets, SBOM, provenance, audit trail]
    G6[Gate 6: AgentOps<br/>Replay session: tool called, why, permission, cost]

    G1 -->|Phase 0-3| M1[M1: Classical MLOps Platform]
    G2 -->|Phase 4-5| M1
    G3 -->|Phase 5| M1
    G4 -->|Phase 7| M1
    G5 -->|Phase 8| M1
    G6 -->|Phase 20-23| M3[M3: AgentOps]

    style M1 fill:#2d6a4f,color:#fff
```

---

## Full Traceability Chain

A single `prediction_id` must resolve:

```
prediction_id: "pred-9f3b2e"
  ├── prediction_ts: 2026-06-29T10:15:22Z
  ├── entity_key:    "customer-42"
  ├── score:         0.731
  ├── decision:      "decline"
  │
  ├── model_version: "credit-risk-v1.2"
  │     ├── mlflow_run_id: "abc123def456"
  │     │     ├── code_sha:   "6c6a398"
  │     │     ├── params:     {"max_iter": 100, "C": 0.1}
  │     │     └── metrics:    {"auc": 0.847, "ks": 0.412}
  │     └── data_version: "v1"
  │           └── dvc_commit: "8f2e..."
  │
  ├── features (at request time, PIT-correct):
  │     ├── age:          38
  │     ├── income:       62_000
  │     ├── derog_marks:  2
  │     └── loan_amount:  25_000
  │
  ├── correlation_id: "req-abc-001"   # request-level trace (multiple services)
  │
  └── outcome (when available):
        ├── outcome_ts: 2026-07-29T08:00:00Z
        └── default:    0
```

---

## Milestone 1 Gate Checklist

```mermaid
flowchart TD
    subgraph Repro["Gate 1 — Reproducibility"]
        R1["✅ MLflow run_id links code + data + config"]
        R2["✅ DVC data version matches run"]
        R3["✅ Artifact SHA-256 in provenance.json"]
        R4["✅ Environment pinned (uv.lock)"]
    end

    subgraph Serving["Gate 2 — Serving"]
        S1["✅ FastAPI /predict returns score + decision + latency"]
        S2["✅ Helm rollback restores previous revision"]
        S3["✅ Locust load test: p99 < 500ms at 50 RPS"]
        S4["✅ ONNX parity: score delta < 0.001"]
    end

    subgraph Pipeline["Gate 3 — Pipeline"]
        P1["✅ Dagster retry on transient failure"]
        P2["✅ Idempotent steps: re-run doesn't duplicate artifacts"]
        P3["✅ Pipeline gate blocks promotion if AUC < threshold"]
    end

    subgraph Monitoring["Gate 4 — Monitoring"]
        M1["✅ PSI/KS/MMD drift detected separately per feature"]
        M2["✅ Closed loop: label join → AUC recompute → retrain trigger"]
        M3["✅ SLO error budget tracked (GREEN/YELLOW/RED)"]
        M4["✅ Three alert channels: infra / ml / business"]
    end

    subgraph Security["Gate 5 — Security"]
        T1["✅ Artifact signed (HMAC-SHA256 in tests, cosign in prod)"]
        T2["✅ SBOM generated (CycloneDX format)"]
        T3["✅ Provenance JSON: commit SHA + data version + signer"]
        T4["✅ Prediction log audit trail with correlation IDs"]
    end

    Repro --> OK[M1 Gate: PASS ✅]
    Serving --> OK
    Pipeline --> OK
    Monitoring --> OK
    Security --> OK
```

---

## Phase-by-Phase Build Map

| Phase | Days | What was built | Gate |
|---|---|---|---|
| 0 | 1–6 | Orientation, system design, DVC, stack | — |
| 1 | 7–9 | Reproducibility: MLflow, DVC integration | G1 |
| 2 | 10–14 | Experiment tracking, model registry | G1 |
| 3 | 15–18 | Calibration, threshold, slice eval, Phase 3 gate | G1 |
| 4 | 19–28 | Serving: FastAPI, BentoML, batch, ONNX, load test | G2 |
| 5 | 29–37 | Pipelines: Dagster, ZenML, failure modes, gate | G3 |
| 6 | 38–45 | Feature store, PIT join, materialization, feedback | G1+G3 |
| 7 | 46–53 | Monitoring: drift, Prometheus, Grafana, closed loop, SLO | G4 |
| 8 | 54–58 | CI/CD: ML testing pyramid, GitLab CI, signing, SBOM | G5 |

---

## Rollback Procedure (CD)

```mermaid
sequenceDiagram
    participant Ops as On-Call
    participant Helm
    participant K8s
    participant Monitor as SLO Monitor

    Monitor->>Ops: SLO budget RED — p99 latency spike
    Ops->>Helm: helm history ml-api
    Helm-->>Ops: Revision 7 (current), Revision 6 (last good)
    Ops->>Helm: helm rollback ml-api 6
    Helm->>K8s: restore Revision 6 pods
    K8s-->>Monitor: pod health restored
    Monitor-->>Ops: SLO budget recovering (YELLOW → GREEN)
    Ops->>Ops: write postmortem
```

---

## What You Now Have: Era A Summary

```
platform/
  training/       ← reproducible training, MLflow, HPO, calibration, registry
  serving/        ← FastAPI, BentoML, ONNX, batch, load test, security
  pipelines/      ← Dagster, ZenML, failure modes, validation + pipeline gates
  features/       ← feature store, PIT join, materialization, streaming, skew
  monitoring/     ← drift, Evidently, Prometheus, Grafana, prediction log, SLO
  ci/             ← ML testing pyramid, GitLab CI builder, signing, SBOM, M1 gate
```

End of **Era A — Classical MLOps**. Next: **Era B — LLMOps**.
