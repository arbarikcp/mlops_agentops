# PROGRESS — MLOps + AgentOps Platform

> Updated after every phase. Each entry links to theory docs, lists deliverables,
> and includes the exact commands to run from a fresh `git checkout` of the phase tag.

---

## How to Use This File

Each completed phase has:
1. **Day table** — what was covered and the deliverable per day.
2. **What's in this phase** — theory docs, code files, tests created.
3. **Quick start** — commands to run from a `git checkout <tag>`.

```bash
# Jump to any phase:
git checkout phase0   # Orientation only (no runnable code)
git checkout phase1   # Reproducibility + tracking + registry
```

---

## Phase 0 — Orientation & System Design (Days 1–6)
**Tag:** `phase0`

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 1 | Why ML Rots | [day01_why_ml_rots.md](docs/phase0/day01_why_ml_rots.md) | Threat Model v0 started | ✅ |
| 2 | Tooling Landscape | [day02_tooling_landscape.md](docs/phase0/day02_tooling_landscape.md) | Annotated stack map | ✅ |
| 3 | Local Platform | [day03_local_platform.md](docs/phase0/day03_local_platform.md) | `make up` working | ✅ |
| 4 | ML System Design | [day04_system_design.md](docs/phase0/day04_system_design.md) | System design doc | ✅ |
| 5 | Project Charter | [day05_project_charter.md](docs/phase0/day05_project_charter.md) | Charter + repo scaffold | ✅ |
| 6 | Dataset & EDA | [day06_dataset_eda.md](docs/phase0/day06_dataset_eda.md) | EDA notebook + data contract v0 | ✅ |

### What's in This Phase

**Theory docs** (`docs/phase0/`):

| File | Content |
|---|---|
| [day01_why_ml_rots.md](docs/phase0/day01_why_ml_rots.md) | ML technical debt taxonomy, maturity levels (L0→L2), AgentOps divergence |
| [threat_model_v0.md](docs/phase0/threat_model_v0.md) | Full STRIDE analysis — 20 threats, risk quadrant matrix |
| [day02_tooling_landscape.md](docs/phase0/day02_tooling_landscape.md) | 2026 stack map with decision rationale for every tool |
| [day03_local_platform.md](docs/phase0/day03_local_platform.md) | Local infra architecture, port map, security decisions |
| [day04_system_design.md](docs/phase0/day04_system_design.md) | FP/FN cost model, latency budget, rollback state machine, late-label timeline |
| [day05_project_charter.md](docs/phase0/day05_project_charter.md) | Project charter table, milestone Gantt, module ownership rules |
| [day06_dataset_eda.md](docs/phase0/day06_dataset_eda.md) | UCI Credit Default dataset choice, EDA findings, data contract draft v0 |

**Platform scaffold** (`platform/`):

| File | Purpose |
|---|---|
| [Makefile](platform/Makefile) | `make up / down / status / smoke-test` |
| [docker-compose.yml](platform/docker-compose.yml) | MinIO + Postgres + Redis + MLflow, health-checked |
| [.env.example](platform/.env.example) | Credential template (copy → `.env`) |
| `infra/local/init-postgres.sql` | DB init with per-role access control |
| `infra/minio/init-buckets.sh` | Private bucket bootstrap |

### Quick Start (from `git checkout phase0`)

```bash
cd platform
cp .env.example .env      # fill in MINIO and Postgres passwords
make up                   # starts MinIO + Postgres + Redis + MLflow
make smoke-test           # verify all 5 services are reachable
make down                 # stop when done
```

Ports after `make up`:
- MinIO console: http://localhost:9001
- MLflow UI: http://localhost:5000
- Postgres: localhost:5432
- Redis: localhost:6379

---

## Phase 1 — Reproducibility, Tracking, Registry (Days 7–14)
**Tag:** `phase1`

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 7 | Non-determinism | [day07_nondeterminism.md](docs/phase1/day07_nondeterminism.md) | `training/train.py` — all seeds fixed | ✅ |
| 8 | DVC + MinIO | [day08_dvc_minio.md](docs/phase1/day08_dvc_minio.md) | DVC config + MinIO remote + threat checkpoint | ✅ |
| 9 | DVC Pipelines | [day09_dvc_pipelines.md](docs/phase1/day09_dvc_pipelines.md) | `dvc.yaml` DAG — ingest→featurize→train | ✅ |
| 10 | MLflow Tracking | [day10_mlflow_tracking.md](docs/phase1/day10_mlflow_tracking.md) | `mlflow_train.py` — params + metrics + model signature | ✅ |
| 11 | MLflow Registry | [day11_mlflow_registry.md](docs/phase1/day11_mlflow_registry.md) | `registry.py` — aliases + compare + provenance chain | ✅ |
| 12 | Optuna HPO | [day12_optuna_hpo.md](docs/phase1/day12_optuna_hpo.md) | `hpo.py` — 20 trials, nested MLflow runs, leaderboard | ✅ |
| 13 | Lineage | [day13_lineage.md](docs/phase1/day13_lineage.md) | `pipelines/lineage.py` — OpenLineage emitter | ✅ |
| 14 | Reproducibility Gate | [day14_reproducibility_gate.md](docs/phase1/day14_reproducibility_gate.md) | `verify_reproducibility.sh` — 6-step gate dry-run | ✅ |

### What's in This Phase

**Theory docs** (`docs/phase1/`):

| File | Content |
|---|---|
| [day07_nondeterminism.md](docs/phase1/day07_nondeterminism.md) | 5 sources of non-determinism, fixing each, lockfiles, content hashing |
| [day08_dvc_minio.md](docs/phase1/day08_dvc_minio.md) | DVC architecture, lifecycle commands, DVC vs Iceberg, data poisoning threat |
| [day09_dvc_pipelines.md](docs/phase1/day09_dvc_pipelines.md) | `dvc.yaml` structure, `dvc repro` DAG, `dvc.lock` audit trail, metrics diff |
| [day10_mlflow_tracking.md](docs/phase1/day10_mlflow_tracking.md) | MLflow architecture (Postgres + MinIO), experiments/runs/params/metrics, signatures |
| [day11_mlflow_registry.md](docs/phase1/day11_mlflow_registry.md) | Versions vs aliases, provenance chain, rollback, artifact provenance threat |
| [day12_optuna_hpo.md](docs/phase1/day12_optuna_hpo.md) | TPE sampler, MedianPruner, nested MLflow runs, leaderboard, search space design |
| [day13_lineage.md](docs/phase1/day13_lineage.md) | OpenLineage spec, event types, facets, producer/consumer model, console backend |
| [day14_reproducibility_gate.md](docs/phase1/day14_reproducibility_gate.md) | 6-step gate checklist, failure diagnosis flowchart, Phase 1 summary |

**Code** (`platform/`):

| File | What it does |
|---|---|
| `training/config.py` | Pydantic models for `params.yaml` — typed, validated, flat-export for MLflow |
| `training/features.py` | `clean_raw_data`, `engineer_features` (7 derived features), `split_data` (time-based). DVC CLI entry point. |
| `training/evaluate.py` | `compute_metrics` (AUC, AUPR, Brier, ECE), `compute_confusion_details` (FP/FN cost) |
| `training/train.py` | Deterministic train: set all seeds → load → featurize → train LightGBM → save pkl + JSON |
| `training/mlflow_train.py` | Same pipeline wrapped in `mlflow.start_run` — logs params, metrics, model, git tag |
| `training/registry.py` | `register_model`, `set_alias`, `load_model_by_alias`, `compare_champion_challenger` |
| `training/hpo.py` | Optuna study (TPE + MedianPruner), 3-fold CV objective, nested MLflow runs, final retrain |
| `data/ingest.py` | Download UCI Credit Default .xls, normalise columns, validate shape, save CSV |
| `data/contracts/raw_schema.py` | Pandera schema (all 25 columns), `validate_raw`, class balance check, CLI |
| `pipelines/lineage.py` | `LineageEmitter` context manager, `LineageDataset` facets, console/HTTP backends |
| `dvc.yaml` | Pipeline DAG: ingest → featurize → train → mlflow_train |
| `params.yaml` | Single source of truth for all hyperparams (DVC tracks changes) |
| `scripts/verify_reproducibility.sh` | 6-step gate dry-run: run retrieval → git tag → DVC → lockfile → determinism → metric match |

**Tests** (`platform/tests/`):

| File | Tests |
|---|---|
| `tests/unit/test_features.py` | 14 tests — clean_raw_data, engineer_features, split_data (purity, determinism, no-leak) |
| `tests/unit/test_evaluate.py` | 13 tests — AUC bounds, perfect/random/worst classifier, ECE, cost calculation |
| `tests/data/test_raw_schema.py` | 11 tests — valid data passes, each schema rule rejects bad values |

### Quick Start (from `git checkout phase1`)

**Prerequisites:** Docker Desktop (or Colima), `uv` installed.

```bash
# 1. Start infrastructure
cd platform
cp .env.example .env     # fill: MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, POSTGRES_PASSWORD
make up

# 2. Install Python dependencies
make install             # runs: uv sync

# 3. Configure DVC remote (MinIO)
make dvc-init

# 4. Download data + run pipeline
make data-download       # downloads UCI dataset, dvc add, dvc push
make pipeline            # dvc repro: featurize → train
cat metrics/train_metrics.json

# 5. Train with MLflow tracking
make mlflow-train
open http://localhost:5000    # view run in MLflow UI

# 6. Run HPO sweep (takes ~10 min for 20 trials)
make hpo-quick           # 3 trials (smoke test)
make hpo                 # 20 trials (full sweep)

# 7. Run tests
make test                # all 35 unit tests
make test-unit           # unit tests only
make test-data           # schema contract tests only

# 8. Verify determinism
make determinism-check   # trains twice, diffs metrics (must be identical)

# 9. Reproducibility gate dry-run
make reproduce-check     # 6-step gate — needs a run_id from step 5
```

**Debugging:**
```bash
make status              # check all Docker services
make smoke-test          # verify service connectivity
docker logs mlops-mlflow -f    # MLflow server logs
LOG_LEVEL=DEBUG python -m training.train    # verbose training output
```

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

| Gate | Status | Tag | Evidence |
|---|---|---|---|
| Reproducibility | ☐ In progress | (Day 14 dry-run done) | `make reproduce-check` passes |
| Serving | ☐ Not started | — | — |
| Pipeline | ☐ Not started | — | — |
| Monitoring | ☐ Not started | — | — |
| Security | ☐ Not started | — | — |
| AgentOps | ☐ Not started | — | — |

---

## Threat Model Versions

| Version | Tag | Day | Changes |
|---|---|---|---|
| v0 | `phase0` | Day 1 | Initial STRIDE, 20 threats, all components pre-build |
| v1 | (Day 14) | Day 14 | T-01 partially mitigated (DVC hash); T-03/S-03 partial (MLflow hash) |

---

## Key Decisions Log

| Day | Decision | Rationale |
|---|---|---|
| 2 | `uv` for env management | 10-100x faster than pip, lockfile, pyproject.toml |
| 2 | Dagster for orchestration | Asset model, native lineage, better than Airflow for ML |
| 2 | AWS as cloud (deep) | EKS/SageMaker ecosystem depth |
| 4 | Threshold: minimise total expected loss | FN ($8K) >> FP ($2K) |
| 4 | 3-class output (approve/review/decline) | Human review band reduces wrong automated decisions |
| 6 | UCI Credit Card Default dataset | Regulated, imbalanced, temporal — forces good MLOps habits |
| 7 | `PYTHONHASHSEED` set in DVC `cmd:`, not in Python | Must be set before process starts — env var only |
| 8 | MinIO as DVC remote | S3-compatible, local parity with prod S3, free |
| 10 | Manual MLflow logging over autolog | Full control of param names, custom metrics (ECE, cost) |
| 11 | Aliases over MLflow stages | Stages deprecated in MLflow 2.x; aliases are atomic pointer swaps |
| 12 | TPE sampler + MedianPruner | TPE learns from past trials; pruner cuts bad trials early |
