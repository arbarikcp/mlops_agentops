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
**Tag:** `phase2`

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 15 | Calibration | [day15_calibration.md](docs/phase2/day15_calibration.md) | `training/calibration.py` — isotonic/sigmoid calibrator, reliability data | ✅ |
| 16 | Threshold Tuning | [day16_threshold_tuning.md](docs/phase2/day16_threshold_tuning.md) | `training/threshold.py` — cost-optimal threshold + sweep | ✅ |
| 17 | Confidence & Abstain | [day17_confidence_abstain.md](docs/phase2/day17_confidence_abstain.md) | `training/decision.py` — `ThresholdBand`, 3-class routing | ✅ |
| 18 | Slice Performance | [day18_slice_performance.md](docs/phase2/day18_slice_performance.md) | `training/slice_eval.py` — slice AUC + OOD detection | ✅ |

### What's in This Phase

**Theory docs** (`docs/phase2/`):

| File | Content |
|---|---|
| [day15_calibration.md](docs/phase2/day15_calibration.md) | Calibration problem, reliability diagrams, Platt vs isotonic, ECE/Brier, sklearn compatibility note |
| [day16_threshold_tuning.md](docs/phase2/day16_threshold_tuning.md) | Cost-sensitive threshold, $C_{FP}/(C_{FP}+C_{FN})$ heuristic, cost curve, calibration→threshold order |
| [day17_confidence_abstain.md](docs/phase2/day17_confidence_abstain.md) | 3-class routing economics, band width strategy, human-in-the-loop band, serving integration |
| [day18_slice_performance.md](docs/phase2/day18_slice_performance.md) | Simpson's paradox, protected attributes, fairness criteria (incompatible), OOD via Isolation Forest |

**Code** (`platform/training/`):

| File | What it does |
|---|---|
| `training/calibration.py` | `fit_calibrator` (isotonic or sigmoid on held-out set), `CalibrationReport`, `reliability_data` — sklearn 1.4+ compatible |
| `training/threshold.py` | `find_cost_optimal_threshold` (sweeps 200 points, minimises FP×$2K + FN×$8K), `threshold_sweep` DataFrame |
| `training/decision.py` | `ThresholdBand` (frozen dataclass, vectorised routing), `find_review_band`, `calibrate_band_for_cost` |
| `training/slice_eval.py` | `evaluate_slices` (per-slice AUC/AP/ECE), `worst_slices`, `slice_gap_report`, `fit_ood_detector`, `ood_report` |

**Tests** (`platform/tests/unit/`):

| File | Tests |
|---|---|
| `tests/unit/test_calibration.py` | 12 tests — fit calibrator both methods, probs in [0,1], report fields, ECE range, reliable gap |
| `tests/unit/test_threshold.py` | 12 tests — result type, threshold range, cost matches manual, confusion sums to N, FN-heavy shifts threshold down |
| `tests/unit/test_decision.py` | 15 tests — approve/review/decline routing, invalid band raises, batch vectorised, stats sum to 1, approve default rate < decline |
| `tests/unit/test_slice_eval.py` | 20 tests — expected columns, metrics in range, skips tiny/missing slices, OOD fraction in range, OOD scores lower for shifted dist |

### Quick Start (from `git checkout phase2`)

**Prerequisites:** Phase 1 complete — model trained and in `models/credit_risk_model.pkl`, processed data in `data/processed/features.parquet`.

```bash
cd platform

# 1. Start infrastructure and install deps:
cp .env.example .env && make up && make install

# 2. Ensure a trained model exists:
make train            # or: make mlflow-train

# 3. Calibration (Day 15):
make calibrate        # ECE before/after, saves metrics/reliability_diagram.csv

# 4. Cost-optimal threshold (Day 16):
make threshold-analysis  # finds t* that minimises FP×$2K + FN×$8K

# 5. Decision band (Day 17):
make decision-band    # approve/review/decline routing stats

# 6. Slice evaluation + OOD (Day 18):
make slice-eval       # per-slice AUC for EDUCATION, SEX, MARRIAGE + OOD fraction

# 7. Run all Phase 2 tests:
make test             # all 59 unit tests (includes Phase 1 tests)
uv run pytest tests/unit/test_calibration.py tests/unit/test_threshold.py \
    tests/unit/test_decision.py tests/unit/test_slice_eval.py -v  # phase 2 only
```

**Key outputs after running:**

| File | Contents |
|---|---|
| `metrics/reliability_diagram.csv` | Calibration curve bins (mean_predicted, fraction_positive, gap) |
| `metrics/threshold_sweep.csv` | Full cost curve across 200 thresholds |
| `metrics/slice_metrics.csv` | Per-slice AUC, AP, ECE for all demographic groups |

**Debugging:**
```bash
# Calibration not improving?
# → Check calibration set size (need >= 1000 for isotonic; use sigmoid for smaller sets)

# Optimal threshold too low (< 0.10)?
# → Verify FN/FP costs in threshold.py DEFAULT_FN_COST / DEFAULT_FP_COST
# → Check class balance in the dataset

# Slice gap too large for a protected group?
# → Investigate training data representation for that group
# → Do NOT promote model to champion until investigated

# OOD fraction high on test set?
# → Feature pipeline may have changed between training and test
# → Check for data leakage or temporal shift
```

---

## Phase 3 — Data & Label Contracts (Days 19–21)
**Tag:** `phase3`

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 19 | Data Contracts | [day19_data_contracts.md](docs/phase3/day19_data_contracts.md) | `feature_schema.py`, `contract_registry.py`, `statistical_checks.py` | ✅ |
| 20 | Label Contracts | [day20_label_contracts.md](docs/phase3/day20_label_contracts.md) | `label_contract.py`, `ground_truth.py` | ✅ |
| 21 | Train/Serve Skew | [day21_train_serve_skew.md](docs/phase3/day21_train_serve_skew.md) | `monitoring/reference_stats.py`, `monitoring/skew_detector.py` | ✅ |

### What's in This Phase

**Theory docs** (`docs/phase3/`):

| File | Content |
|---|---|
| [day19_data_contracts.md](docs/phase3/day19_data_contracts.md) | 3 enforcement layers (schema/domain/statistical), freshness, ownership, contract versioning, Pandera patterns |
| [day20_label_contracts.md](docs/phase3/day20_label_contracts.md) | Label delay problem in credit risk, provenance fields, correction protocol, backfill, label arrival curve, leakage pitfall |
| [day21_train_serve_skew.md](docs/phase3/day21_train_serve_skew.md) | 3 types of skew, PSI formula + thresholds, KS test, JS divergence, reference stats architecture, per-slice PSI, alert thresholds |

**Code** (`platform/`):

| File | What it does |
|---|---|
| `data/contracts/feature_schema.py` | Pandera schema for post-featurization dataset — 32 base + 7 derived columns, semantic bounds on derived features, `check_no_infinite_values()` |
| `data/contracts/contract_registry.py` | `ContractMetadata` (frozen dataclass: owner, version, enforcement_mode), `ContractRegistry.validate()` (strict / warn / log_only), freshness check, `default_registry` |
| `data/contracts/statistical_checks.py` | `DatasetStats`, `ColumnStats`, `compute_dataset_stats()`, `check_null_drift()`, `check_mean_drift()` (z-score), `check_class_balance()` |
| `data/contracts/label_contract.py` | `LabelMetadata`, Pandera `label_batch_schema`, `validate_label_batch()`, `check_label_arrival()`, `check_single_policy_version()`, `check_correction_rate()` |
| `data/contracts/ground_truth.py` | `join_predictions_with_outcomes()` (filters by outcome delay), `detect_label_corrections()`, `backfill_labels()`, `LabelArrivalCurve` (T+1/7/30/90/180) |
| `monitoring/__init__.py` | Module init with Phase 3–5 roadmap comment |
| `monitoring/reference_stats.py` | `ReferenceStats`, `compute_reference_stats()`, `save_reference_stats()` / `load_reference_stats()` (JSON), `check_feature_alignment()` |
| `monitoring/skew_detector.py` | `compute_psi()`, `compute_ks()`, `compute_js()`, `FeatureSkewResult`, `SkewReport`, `detect_skew()`, `skew_summary()` → DataFrame |

**Tests** (`platform/tests/unit/`):

| File | Tests |
|---|---|
| `tests/unit/test_feature_schema.py` | 17 tests — valid passes, derived feature bounds, cleaned categoricals, infinite value detection |
| `tests/unit/test_contract_registry.py` | 20 tests — immutability, version collision, strict/warn/log_only modes, freshness checks, default registry |
| `tests/unit/test_statistical_checks.py` | 20 tests — compute_dataset_stats serialisation roundtrip, null drift detection, mean drift z-score, class balance range |
| `tests/unit/test_label_contract.py` | 20 tests — LabelMetadata validation, schema enforcement, arrival fraction, policy version consistency, correction rate |
| `tests/unit/test_ground_truth.py` | 20 tests — join filters provisional, backfill deduplication, correction detection, arrival curve horizons |
| `tests/unit/test_reference_stats.py` | 16 tests — roundtrip JSON serialisation, feature alignment, missing feature detection, parent dir creation |
| `tests/unit/test_skew_detector.py` | 22 tests — PSI=0 for identical, PSI high for shifted, KS significance, JS bounds, detect_skew report shape, skew_summary sorted |

**Total Phase 3 tests: 145 (all passing)**

### Quick Start (from `git checkout phase3`)

**Prerequisites:** Phase 1 complete — model trained, `data/processed/features.parquet` exists.

```bash
cd platform
make install      # uv sync

# Day 19 — Data contracts
make data-contract   # validates features.parquet against Pandera + statistical checks

# Day 20 — Label contracts (uses synthetic batch — no real outcome data needed)
make label-contract  # validates label batch schema + arrival timing

# Day 21 — Skew detection
make skew-detect     # computes reference stats + detects train vs test skew
                     # saves: metrics/reference_stats.json, metrics/skew_report.csv

# Run Phase 3 unit tests only
uv run pytest tests/unit/test_feature_schema.py tests/unit/test_contract_registry.py \
    tests/unit/test_statistical_checks.py tests/unit/test_label_contract.py \
    tests/unit/test_ground_truth.py tests/unit/test_reference_stats.py \
    tests/unit/test_skew_detector.py -v

# Run all tests (all phases)
make test
```

**Key outputs after running:**

| File | Contents |
|---|---|
| `metrics/reference_stats.json` | Training feature statistics snapshot (mean, std, percentiles per column) |
| `metrics/skew_report.csv` | Per-feature PSI, KS stat, KS p-value, JS divergence, severity, flag |

**Debugging:**
```bash
# Contract validation fails with SchemaErrors?
uv run python -m data.contracts.feature_schema data/processed/features.parquet --verbose

# PSI flagged but you believe data is fine?
# → Check which feature was flagged (skew_report.csv, sorted by PSI desc)
# → PSI > 0.20 on a single feature is significant; check upstream pipeline for that column
# → Remember: PSI uses Gaussian approximation for reference — non-Gaussian features may show noise

# Label arrival < 90% confirmed?
# → Check outcome_delay_days in LabelMetadata vs your actual label lag
# → Consider increasing the wait window before the next re-train

# Multiple policy versions in label batch?
# → This means labels were generated under different rules
# → Must re-derive all labels under the new policy before training
```

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
