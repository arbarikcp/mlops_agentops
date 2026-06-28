# PROGRESS — MLOps + AgentOps Platform

> Updated after every session. Each entry links to the day's theory doc and lists the deliverable.

---

## Phase 0 — Orientation & System Design (Days 1–6)

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 1 | Why ML Rots | [day01_why_ml_rots.md](docs/phase0/day01_why_ml_rots.md) | Threat Model v0 started | ✅ |
| 2 | Tooling Landscape | [day02_tooling_landscape.md](docs/phase0/day02_tooling_landscape.md) | Annotated stack map | ✅ |
| 3 | Local Platform | [day03_local_platform.md](docs/phase0/day03_local_platform.md) | `make up` working | ✅ |
| 4 | ML System Design | [day04_system_design.md](docs/phase0/day04_system_design.md) | System design doc | ✅ |
| 5 | Project Charter | [day05_project_charter.md](docs/phase0/day05_project_charter.md) | Charter + repo scaffold | ✅ |
| 6 | Dataset & EDA | [day06_dataset_eda.md](docs/phase0/day06_dataset_eda.md) | EDA notebook + data contract v0 | ✅ |

---

## Phase 1 — Reproducibility, Tracking, Registry (Days 7–14)

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 7 | Non-determinism | [day07_nondeterminism.md](docs/phase1/day07_nondeterminism.md) | `training/train.py` — seeds fixed, deterministic | ✅ |
| 8 | DVC + MinIO | [day08_dvc_minio.md](docs/phase1/day08_dvc_minio.md) | DVC config + MinIO remote + threat checkpoint | ✅ |
| 9 | DVC Pipelines | [day09_dvc_pipelines.md](docs/phase1/day09_dvc_pipelines.md) | `dvc.yaml` DAG — ingest→featurize→train | ✅ |
| 10 | MLflow Tracking | [day10_mlflow_tracking.md](docs/phase1/day10_mlflow_tracking.md) | `mlflow_train.py` — params+metrics+model+signature | ✅ |
| 11 | MLflow Registry | [day11_mlflow_registry.md](docs/phase1/day11_mlflow_registry.md) | `registry.py` — aliases + compare + provenance | ✅ |
| 12 | Optuna HPO | [day12_optuna_hpo.md](docs/phase1/day12_optuna_hpo.md) | `hpo.py` — 20 trials, nested MLflow runs | ✅ |
| 13 | Lineage | [day13_lineage.md](docs/phase1/day13_lineage.md) | `pipelines/lineage.py` — OpenLineage emitter | ✅ |
| 14 | Reproducibility Gate | [day14_reproducibility_gate.md](docs/phase1/day14_reproducibility_gate.md) | `verify_reproducibility.sh` — 6-step gate dry-run | ✅ |

---

## Phase 2 — Calibration & Thresholds (Days 15–18)

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 15 | Calibration | — | Reliability diagram | ☐ |
| 16 | Threshold Tuning | — | Cost-sensitive threshold | ☐ |
| 17 | Confidence & Abstain | — | Reject/abstain pipeline | ☐ |
| 18 | Slice Performance | — | Slice-level eval + OOD | ☐ |

---

## Phase 3 — Data & Label Contracts (Days 19–21)

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 19 | Data Contracts | — | Pandera + GE contract | ☐ |
| 20 | Label Contracts | — | Ground truth pipeline | ☐ |
| 21 | Train/Serve Skew | — | Skew detection script | ☐ |

---

## Phase 4 — Packaging & Serving (Days 22–30)

| Day | Title | Deliverable | Status |
|---|---|---|---|
| 22 | Serialization (ONNX) | ONNX export | ☐ |
| 23 | Inference Patterns | (theory) | ☐ |
| 24 | FastAPI | Versioned endpoint | ☐ |
| 25 | Containerize | Non-root multi-stage image | ☐ |
| 26 | BentoML | Bento with adaptive batching | ☐ |
| 27 | Batch Inference | Idempotent batch script | ☐ |
| 28 | Model API Contract | Versioned schema | ☐ |
| 29 | Load Testing | k6/Locust report p95/p99 | ☐ |
| 30 | Serving Security | mTLS + rate limits + Serving gate | ☐ |

---

## Phase 5 — Orchestration & Pipelines (Days 31–37)

| Day | Title | Deliverable | Status |
|---|---|---|---|
| 31 | Orchestration Principles | (theory) | ☐ |
| 32 | Dagster Pipeline | Training pipeline as assets | ☐ |
| 33 | ML-Native (KFP/ZenML) | Light build | ☐ |
| 34 | Validation Gate | Pandera + GE in pipeline | ☐ |
| 35 | Model Validation Gate | Champion/challenger | ☐ |
| 36 | Pipeline Failure Modes | Idempotency proof | ☐ |
| 37 | Survey + Pipeline Gate | Dry-run | ☐ |

---

## Phase 6 — Feature Store & Closed Feedback Loop (Days 38–45)

*(to fill as we progress)*

---

## Phase 7 — Monitoring & Closed Loop (Days 46–53)

*(to fill as we progress)*

---

## Phase 8 — CI/CD for ML (Days 54–58) → MILESTONE 1 GATE

*(to fill as we progress)*

---

## Production Gates Status

| Gate | Status | Evidence |
|---|---|---|
| Reproducibility | ☐ Not started | — |
| Serving | ☐ Not started | — |
| Pipeline | ☐ Not started | — |
| Monitoring | ☐ Not started | — |
| Security | ☐ Not started | — |
| AgentOps | ☐ Not started | — |

---

## Threat Model Versions

| Version | Day | Changes |
|---|---|---|
| v0 | Day 1 | Initial STRIDE, all components pre-build |
| v1 | Day 14 | (TBD) |

---

## Key Decisions Log

| Day | Decision | Rationale |
|---|---|---|
| 2 | `uv` for env management | 10-100x faster than pip, lockfile support |
| 2 | Dagster for orchestration | Asset model, better lineage than Airflow |
| 2 | AWS as cloud (deep) | EKS/SageMaker ecosystem |
| 4 | Threshold: minimise total expected loss | FN ($8K) >> FP ($2K) |
| 4 | 3-class output (approve/review/decline) | Human review band reduces wrong decisions |
| 6 | UCI Credit Card Default dataset | Regulated, imbalanced, temporal — forces good habits |
