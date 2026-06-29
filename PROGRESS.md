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
**Tag:** `phase4`

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 22 | Serialization | [day22_serialization.md](docs/phase4/day22_serialization.md) | `serving/serialization.py` — ONNX export, pickle risk, SHA-256 checksum | ✅ |
| 23 | Inference Patterns | [day23_inference_patterns.md](docs/phase4/day23_inference_patterns.md) | `serving/inference.py` — ModelRunner, LatencyTracker, online + batch | ✅ |
| 24 | FastAPI | [day24_fastapi.md](docs/phase4/day24_fastapi.md) | `serving/api.py` — /health, /ready, /v1/predict, /v1/predict/batch | ✅ |
| 25 | Containerize | [day25_containerize.md](docs/phase4/day25_containerize.md) | `serving/Dockerfile` (multi-stage, non-root), `.dockerignore` | ✅ |
| 26 | BentoML | [day26_bentoml.md](docs/phase4/day26_bentoml.md) | `serving/bento_service.py` — AdaptiveBatcher, BentoPackager | ✅ |
| 27 | Batch Inference | [day27_batch_inference.md](docs/phase4/day27_batch_inference.md) | `serving/batch_inference.py` — BatchInferenceJob, ManifestStore, plan_backfill | ✅ |
| 28 | Model API Contract | [day28_api_contract.md](docs/phase4/day28_api_contract.md) | `serving/api_contract.py` — FieldSchema, ApiContractChecker, RollbackPlan | ✅ |
| 29 | Load Testing | [day29_load_testing.md](docs/phase4/day29_load_testing.md) | `serving/load_test.py`, `serving/locustfile.py` — LatencyProfiler, LoadTestRunner | ✅ |
| 30 | Serving Security | [day30_serving_security.md](docs/phase4/day30_serving_security.md) | `serving/security.py` — ApiKeyStore, RateLimiter, SecurityConfig + Serving Gate dry-run | ✅ |

### What's in This Phase

**Theory docs** (`docs/phase4/`):

| File | Content |
|---|---|
| [day22_serialization.md](docs/phase4/day22_serialization.md) | ONNX vs pickle vs safetensors, pickle CVE, parity validation, opset, float32 precision |
| [day23_inference_patterns.md](docs/phase4/day23_inference_patterns.md) | Online/batch/streaming comparison, latency budget breakdown, p50/p95/p99 explanation |
| [day24_fastapi.md](docs/phase4/day24_fastapi.md) | Pydantic v2 schemas, /health vs /ready distinction, API versioning, lifespan startup |
| [day25_containerize.md](docs/phase4/day25_containerize.md) | Multi-stage build, non-root user, image scanning (Trivy), threat checkpoint |
| [day26_bentoml.md](docs/phase4/day26_bentoml.md) | Runner abstraction, adaptive batching algorithm, max_batch_size vs max_latency_ms tuning |
| [day27_batch_inference.md](docs/phase4/day27_batch_inference.md) | Idempotency key pattern, manifest protocol, backfill strategy, checksum verification |
| [day28_api_contract.md](docs/phase4/day28_api_contract.md) | Breaking vs compatible schema changes, deprecation protocol, rollback plan SLA |
| [day29_load_testing.md](docs/phase4/day29_load_testing.md) | k6 script, Locust scenarios, load test phases (baseline/ramp/soak/spike) |
| [day30_serving_security.md](docs/phase4/day30_serving_security.md) | AuthN/AuthZ, rate limiting, secrets management, mTLS, Serving Gate checklist |

**Code** (`platform/serving/`):

| File | What it does |
|---|---|
| `serving/__init__.py` | Module init with serving module roadmap |
| `serving/serialization.py` | `ModelSerializer` — ONNX export, checksum verify, parity check, pickle risk |
| `serving/inference.py` | `ModelRunner` — ONNX session management, online predict, batch predict; `LatencyTracker` |
| `serving/api.py` | FastAPI app — /health, /ready, /v1/predict, /v1/predict/batch, /v1/model/info |
| `serving/Dockerfile` | Multi-stage Docker build, non-root appuser (UID 1001), HEALTHCHECK |
| `.dockerignore` | Excludes .git, .env, tests/, data/raw/ from image |
| `serving/bento_service.py` | `AdaptiveBatcher`, `RunnerConfig`, `BentoPackager` — BentoML concepts without bentoml dep |
| `serving/batch_inference.py` | `BatchInferenceJob`, `ManifestStore`, `BatchJobManifest`, `plan_backfill` |
| `serving/api_contract.py` | `FieldSchema`, `ApiContractVersion`, `ApiContractChecker`, `CompatibilityReport`, `RollbackPlan` |
| `serving/load_test.py` | `LatencyProfiler`, `LoadTestConfig`, `LoadTestResult`, `LoadTestRunner` |
| `serving/locustfile.py` | Locust scenario (90% predict / 9% health / 1% info) with traffic mix |
| `serving/security.py` | `ApiKey`, `ApiKeyStore` (SHA-256 hashing), `RateLimiter` (sliding window), `SecurityConfig` |

**Tests** (`platform/tests/unit/`):

| File | Tests |
|---|---|
| `tests/unit/test_serialization.py` | 23 tests — SHA-256, checksum verify, pickle risk levels, ONNX export mock, parity check |
| `tests/unit/test_inference.py` | 26 tests — LatencyTracker p-tiles, ModelRunner load/predict-single/predict-batch, threshold |
| `tests/unit/test_api.py` | 35 tests — /health always 200, /ready 503→200, /v1/predict validation, 422 cases |
| `tests/unit/test_dockerfile.py` | 19 tests — multi-stage FROM, USER appuser, EXPOSE 8080, no secrets, slim base, .dockerignore |
| `tests/unit/test_bento_service.py` | 27 tests — RunnerConfig validation, flush_now, submit threshold, stats, BentoPackager YAML |
| `tests/unit/test_batch_inference.py` | 21 tests — ManifestStore write/read/list, idempotency skip, force re-run, backfill plan |
| `tests/unit/test_api_contract.py` | 28 tests — field roundtrip, compatible/breaking/warning changes, v1→v2 built-in, RollbackPlan |
| `tests/unit/test_load_test.py` | 26 tests — LatencyProfiler record/measure, LoadTestConfig validation, runner results, locustfile |
| `tests/unit/test_security.py` | 36 tests — key hashing, ApiKey expiry, ApiKeyStore validate/revoke, RateLimiter window, SecurityConfig |

**Total Phase 4 tests: 241 (all passing)**

### Quick Start (from `git checkout phase4`)

**Prerequisites:** Phase 3 complete. Python 3.11+, `uv` installed.

```bash
cd platform

# 1. Install deps (includes FastAPI, uvicorn, httpx):
uv sync

# 2. Run all Phase 4 tests:
uv run pytest tests/unit/test_serialization.py tests/unit/test_inference.py \
    tests/unit/test_api.py tests/unit/test_dockerfile.py \
    tests/unit/test_bento_service.py tests/unit/test_batch_inference.py \
    tests/unit/test_api_contract.py tests/unit/test_load_test.py \
    tests/unit/test_security.py --override-ini="addopts=" -v

# 3. Start the FastAPI server locally (requires ONNX model):
make serve-local
# Then: curl http://localhost:8080/health
# Then: curl http://localhost:8080/ready

# 4. Serving Gate dry-run:
make serving-gate-check

# 5. Run Locust load test (requires server running):
# locust -f serving/locustfile.py --host http://localhost:8080 \
#     --users 50 --spawn-rate 5 --run-time 60s --headless
```

**Key outputs after running:**

| Artifact | Contents |
|---|---|
| `serving/Dockerfile` | Production-ready multi-stage image definition |
| `models/credit_risk_model.onnx` | ONNX-exported model (via serialization.export_to_onnx) |
| `models/credit_risk_model.onnx.sha256` | SHA-256 checksum for artifact integrity |
| `manifests/*.json` | Batch job completion manifests (idempotency records) |

**Debugging:**

```bash
# FastAPI 503 on /v1/predict?
# → Check /ready — model may not be loaded (needs MODEL_PATH set)

# Parity check fails after ONNX export?
# → float32 precision loss — widen threshold to 5e-3
# → Check column order matches feature_names used at export

# Rate limiter blocking legitimate traffic?
# → Adjust RATE_LIMIT_PER_MIN env var or window_seconds

# Dockerfile image too large?
# → Ensure multi-stage (builder → runtime) — check two FROM statements
# → Run: docker build -f serving/Dockerfile . --target runtime

# Load test p99 > SLA?
# → Profile with LatencyProfiler — split inference vs I/O vs serialisation
# → Check ONNX session is pre-loaded (not loaded per-request)
```

---

## Phase 5 — Orchestration & Pipelines (Days 31–37)
**Tag:** `phase5`

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 31 | Orchestration Principles | [day31_orchestration_principles.md](docs/phase5/day31_orchestration_principles.md) | `pipelines/dag.py` — DagStep, SimpleDag, RunContext, BackfillPlanner | ✅ |
| 32 | Dagster Pipeline | [day32_dagster_pipeline.md](docs/phase5/day32_dagster_pipeline.md) | `pipelines/dagster_pipeline.py` — PipelineConfig, TrainingAssets, TrainingPipeline | ✅ |
| 33 | ZenML Pipeline | [day33_zenml_pipeline.md](docs/phase5/day33_zenml_pipeline.md) | `pipelines/zenml_pipeline.py` — StepDef, ZenPipeline, ArtifactStore, CachePolicy | ✅ |
| 34 | Data Validation Gate | [day34_validation_gate.md](docs/phase5/day34_validation_gate.md) | `pipelines/validation_gate.py` — SchemaCheck, StatisticalCheck, DataValidationGate | ✅ |
| 35 | Model Validation Gate | [day35_model_gate.md](docs/phase5/day35_model_gate.md) | `pipelines/model_gate.py` — ModelGate, ChampionRegistry, GateThresholds | ✅ |
| 36 | Pipeline Failure Modes | [day36_failure_modes.md](docs/phase5/day36_failure_modes.md) | `pipelines/failure_modes.py` — FailureClassifier, IdempotencyProof, LineageAuditor | ✅ |
| 37 | Survey + Pipeline Gate | [day37_survey_pipeline_gate.md](docs/phase5/day37_survey_pipeline_gate.md) | `pipelines/pipeline_gate.py` — PipelineGateRunner, OrchestrationSurvey | ✅ |

### What's in This Phase

**Theory docs** (`docs/phase5/`):

| File | Content |
|---|---|
| [day31_orchestration_principles.md](docs/phase5/day31_orchestration_principles.md) | DAG vs asset-centric, idempotency patterns, retry strategy, backfill, lineage, conditional promotion |
| [day32_dagster_pipeline.md](docs/phase5/day32_dagster_pipeline.md) | Dagster @op/@asset/@resource, IO Manager, partitioning, sensors, Dagster vs Airflow comparison |
| [day33_zenml_pipeline.md](docs/phase5/day33_zenml_pipeline.md) | ZenML stack, step caching (cache key derivation), materializer, step output versioning |
| [day34_validation_gate.md](docs/phase5/day34_validation_gate.md) | Pandera + GE integration, gate vs log distinction, lazy collection of failures |
| [day35_model_gate.md](docs/phase5/day35_model_gate.md) | Metric hierarchy, champion/challenger delta, auto-promote logic, rollback strategy |
| [day36_failure_modes.md](docs/phase5/day36_failure_modes.md) | Three failure classes (transient/deterministic/corruption), idempotency proof, lineage audit |
| [day37_survey_pipeline_gate.md](docs/phase5/day37_survey_pipeline_gate.md) | Prefect/Metaflow/Argo/SageMaker/Vertex AI survey, Pipeline gate dry-run checklist |

**Code** (`platform/pipelines/`):

| File | What it does |
|---|---|
| `pipelines/dag.py` | `DagStep` (retry + cleanup), `SimpleDag` (dependency-aware), `RunContext`, `RetryPolicy`, `BackfillPlanner` |
| `pipelines/dagster_pipeline.py` | `PipelineConfig`, `ResourceRegistry`, `TrainingAssets` (6 asset steps), `TrainingPipeline` |
| `pipelines/zenml_pipeline.py` | `StepDef`, `ZenPipeline`, `ArtifactStore` (disk-persisted cache index), `CachePolicy`, `StackConfig` |
| `pipelines/validation_gate.py` | `SchemaCheck`, `StatisticalCheck`, `DataValidationGate` (lazy), `ValidationGateFailure`, `credit_risk_gate()` |
| `pipelines/model_gate.py` | `ModelMetrics`, `GateThresholds`, `ChampionRegistry` (rollback), `ModelGate`, `compute_model_metrics()` |
| `pipelines/failure_modes.py` | `FailureClassifier`, `IdempotencyProof`, `RetryChecker`, `LineageAuditor` |
| `pipelines/pipeline_gate.py` | `PipelineGateRunner` (combines all checks), `OrchestrationSurvey` (profiles + `recommend()`) |

**Tests** (`platform/tests/unit/`):

| File | Tests |
|---|---|
| `tests/unit/test_dag.py` | 39 tests — RetryPolicy, RunContext, DagStep retry/cleanup, SimpleDag dependency, BackfillPlanner |
| `tests/unit/test_dagster_pipeline.py` | 33 tests — PipelineConfig, ResourceRegistry, TrainingAssets (all 6 steps), TrainingPipeline end-to-end |
| `tests/unit/test_zenml_pipeline.py` | 34 tests — StackConfig, ArtifactStore (save/load/cache), StepDef caching, ZenPipeline, credit risk pipeline |
| `tests/unit/test_validation_gate.py` | 38 tests — SchemaCheck (all constraints), StatisticalCheck (all types), DataValidationGate lazy mode |
| `tests/unit/test_model_gate.py` | 39 tests — ModelMetrics validation, GateThresholds from_env, ChampionRegistry promote/rollback, ModelGate evaluate |
| `tests/unit/test_failure_modes.py` | 33 tests — FailureClassifier (priority), IdempotencyProof (pass/fail/exception), RetryChecker, LineageAuditor |
| `tests/unit/test_pipeline_gate.py` | 30 tests — PipelineGateRunner all checks, OrchestrationSurvey recommend/compare |

**Total Phase 5 tests: 246 (all passing)**

### Quick Start (from `git checkout phase5`)

```bash
cd platform

# 1. Install deps:
uv sync

# 2. Run all Phase 5 tests:
uv run pytest tests/unit/test_dag.py tests/unit/test_dagster_pipeline.py \
    tests/unit/test_zenml_pipeline.py tests/unit/test_validation_gate.py \
    tests/unit/test_model_gate.py tests/unit/test_failure_modes.py \
    tests/unit/test_pipeline_gate.py --override-ini="addopts=" -v

# 3. Run the training pipeline (uses synthetic data if features.parquet not present):
uv run python -c "
from pipelines.dagster_pipeline import TrainingPipeline, PipelineConfig
pipeline = TrainingPipeline.build(PipelineConfig(n_estimators=10, auc_threshold=0.01))
result = pipeline.run()
print(f'Pipeline: {\"SUCCESS\" if result.succeeded else \"FAILED\"}')
for m in result.materializations:
    print(f'  {m.asset_key}: {m.row_count} rows')
"

# 4. Run the Pipeline gate dry-run:
make pipeline-gate-check

# 5. Get orchestration tool recommendation:
uv run python -c "
from pipelines.pipeline_gate import OrchestrationSurvey
survey = OrchestrationSurvey()
recs = survey.recommend(need_asset_centric=True, need_ml_native=True)
for r in recs: print(f'{r.name}: {r.best_for}')
"
```

**Key outputs after running:**

| Artifact | Contents |
|---|---|
| `models/credit_risk_lgbm.pkl` | Trained LightGBM (or GradientBoosting fallback) model |
| `models/champion_model.pkl` | Promoted champion (if AUC gate passed) |
| `.zenml_artifacts/` | ZenML-style versioned artifact store (step cache) |

**Debugging:**

```bash
# Pipeline promotion blocked?
# → Lower auc_threshold in PipelineConfig (0.01 for synthetic data)
# → Check validation_report step output

# ZenML cache not hitting?
# → Inputs changed — check that n_rows and auc_threshold match exactly
# → Delete .zenml_artifacts/ to reset the cache

# Data validation gate failing?
# → Check DataFrame has LIMIT_BAL, AGE, default.payment.next.month columns
# → Check row count >= 100

# Model gate rejecting below threshold?
# → Adjust GateThresholds.min_auc for test environment
# → Use GateThresholds.from_env() and set GATE_MIN_AUC=0.50
```

---

## Phase 6 — Feature Store & Closed Feedback Loop (Days 38–45) ✅
**Tag:** `phase6`

### Day Table

| Day | Title | Theory | Code | Status |
|---|---|---|---|---|
| 38 | Feature Store Problem | [day38_feature_store_problem.md](docs/phase6/day38_feature_store_problem.md) | — | ✅ |
| 39 | Feast Architecture | [day39_feast_architecture.md](docs/phase6/day39_feast_architecture.md) | `features/feature_store.py` | ✅ |
| 40 | Feature Views & PIT Joins | [day40_feature_views.md](docs/phase6/day40_feature_views.md) | `features/feature_views.py` | ✅ |
| 41 | Materialization & Online Store | [day41_materialization.md](docs/phase6/day41_materialization.md) | `features/materialization.py` | ✅ |
| 42 | Streaming Features | [day42_streaming_features.md](docs/phase6/day42_streaming_features.md) | `features/streaming.py` | ✅ |
| 43 | Feature Monitoring | [day43_feature_monitoring.md](docs/phase6/day43_feature_monitoring.md) | `features/feature_monitor.py` | ✅ |
| 44 | Label Feedback Loop | [day44_label_feedback.md](docs/phase6/day44_label_feedback.md) | `features/feedback_loop.py` | ✅ |
| 45 | Consolidation: Zero Skew | [day45_zero_skew.md](docs/phase6/day45_zero_skew.md) | `features/skew_checker.py` | ✅ |

### Code Modules

| Module | Key Classes | Description |
|---|---|---|
| `features/feature_store.py` | `FeatureStore`, `OfflineStore`, `InMemoryOnlineStore`, `FeatureRegistry` | Core feature store primitives — PIT join, offline/online read/write, registry persistence |
| `features/feature_views.py` | `Entity`, `Feature`, `FeatureView`, `FeatureService`, `PointInTimeJoin` | Feature view definitions + PIT join algorithm (no future leakage) |
| `features/materialization.py` | `IncrementalMaterializer`, `MaterializationJob`, `MaterializationInterval` | Batch, incremental, and backfill materialization with idempotency |
| `features/streaming.py` | `PushSource`, `PushSchema`, `OnDemandTransform`, `StreamProcessor` | Real-time feature ingestion via push sources + on-demand transforms |
| `features/feature_monitor.py` | `FeatureMonitor`, `FreshnessChecker`, `FeatureQualityChecker`, `FeatureDriftMonitor` | Three-pillar monitoring: freshness / quality / drift (PSI + KS) |
| `features/feedback_loop.py` | `LabelFeedbackLoop`, `GroundTruthJoiner`, `MetricRecomputer`, `RetrainDecider` | 8-step closed feedback loop: delayed labels, AUC recompute, two-condition retrain trigger |
| `features/skew_checker.py` | `TrainServeSkewChecker`, `TrainServeSkewReport`, `SkewEvidence` | Zero train-serve skew: schema skew, feature PSI, prediction score delta |

### Test Coverage

| Test File | Tests | Coverage |
|---|---|---|
| `tests/unit/test_feature_store.py` | 33 | 91% |
| `tests/unit/test_feature_views.py` | 30 | 100% |
| `tests/unit/test_materialization.py` | 24 | 96% |
| `tests/unit/test_streaming.py` | 24 | 100% |
| `tests/unit/test_feature_monitor.py` | 30 | 95% |
| `tests/unit/test_feedback_loop.py` | 29 | 98% |
| `tests/unit/test_skew_checker.py` | 18 | 98% |
| **Total** | **206** | |

### Quick Start

```bash
# Run all Phase 6 tests
make feature-store-gate-check

# Run unit tests only
uv run pytest tests/unit/test_feature_store.py tests/unit/test_feature_views.py \
    tests/unit/test_materialization.py tests/unit/test_streaming.py \
    tests/unit/test_feature_monitor.py tests/unit/test_feedback_loop.py \
    tests/unit/test_skew_checker.py -v
```

### Key Concepts

- **PIT Correctness** — For each entity row at time T, only use feature snapshots from ≤ T. Implemented in `PointInTimeJoin.join()` and `OfflineStore.get_historical_features()`.
- **Materialization Idempotency** — Re-running for the same window overwrites with the same values. Safe to re-trigger on failure.
- **Freshness vs Missing** — `STALE` triggers an alert but doesn't block the gate. Only `MISSING` (never materialized) blocks `overall_passed`.
- **Two-Condition Retrain Trigger** — Both `n_new_labels >= min_batch_size` AND `|delta_auc| >= threshold` must be satisfied. Prevents noisy small-batch retrains.
- **Zero Skew Claim** — `TrainServeSkewChecker.run(train_df, serve_df).zero_skew == True` is the measurable contract.

---

## Phase 7 — Monitoring & the Closed Loop (Days 46–53)
**Tag:** `phase7` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 46 | Monitoring Taxonomy | [day46_monitoring_taxonomy.md](docs/phase7/day46_monitoring_taxonomy.md) | `monitoring/taxonomy.py` — operational vs ML vs business monitors | ☐ |
| 47 | Drift & Concept Drift | [day47_drift.md](docs/phase7/day47_drift.md) | `monitoring/drift.py` — PSI, KS, MMD, classifier-based drift | ☐ |
| 48 | Evidently Integration | [day48_evidently.md](docs/phase7/day48_evidently.md) | `monitoring/evidently_reports.py` — reports + test suites, in-pipeline | ☐ |
| 49 | Prometheus Custom Metrics | [day49_prometheus.md](docs/phase7/day49_prometheus.md) | `monitoring/prometheus_metrics.py` — custom ML metrics + PromQL examples | ☐ |
| 50 | Grafana: Golden Signals | [day50_grafana.md](docs/phase7/day50_grafana.md) | `infra/grafana/dashboards/ml_golden_signals.json` — dashboard + alerts | ☐ |
| 51 | Prediction Logging | [day51_prediction_logging.md](docs/phase7/day51_prediction_logging.md) | `monitoring/prediction_logger.py` — structured log + correlation ID | ☐ |
| 52 | Closed-Loop Learning | [day52_closed_loop.md](docs/phase7/day52_closed_loop.md) | `monitoring/closed_loop.py` — 8-step loop: predict→decide→outcome→deploy | ☐ |
| 53 | SLOs + Monitoring Gate | [day53_slo_monitoring_gate.md](docs/phase7/day53_slo_monitoring_gate.md) | `monitoring/slo.py` + Monitoring gate dry-run | ☐ |

### Planned Outputs

| Module | What it will do |
|---|---|
| `monitoring/taxonomy.py` | `MonitorType` enum, `MonitorRegistry`, separate alert channels per type |
| `monitoring/drift.py` | `compute_psi()`, `compute_mmd()`, `ClassifierDrift`, `DriftReport` |
| `monitoring/evidently_reports.py` | `DataDriftReport`, `ModelPerformanceTestSuite`, pipeline integration hook |
| `monitoring/prometheus_metrics.py` | `MLMetrics` (prediction_count, latency_histogram, drift_score_gauge) |
| `monitoring/prediction_logger.py` | `PredictionLogger` — JSON log, correlation ID, audit-ready schema |
| `monitoring/closed_loop.py` | `ClosedLoop.tick()` — joins labels, recomputes metrics, triggers retrain gate |
| `monitoring/slo.py` | `SLO`, `ErrorBudget`, `SLOReport`, incident severity mapper |

---

## Phase 8 — CI/CD for ML (Days 54–58) → MILESTONE 1 GATE
**Tag:** `phase8` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 54 | CI/CD for ML | [day54_cicd_for_ml.md](docs/phase8/day54_cicd_for_ml.md) | ML testing pyramid doc; code + data + model CI explained | ☐ |
| 55 | Testing ML Code | [day55_testing_ml.md](docs/phase8/day55_testing_ml.md) | `ci/tests/` — unit, data, behavioral, training smoke tests | ☐ |
| 56 | GitLab CI Pipelines | [day56_gitlab_ci.md](docs/phase8/day56_gitlab_ci.md) | `.gitlab-ci.yml` — stages, runners, caching, artifact upload | ☐ |
| 57 | Automated Build + CD + SBOM | [day57_cd_signing.md](docs/phase8/day57_cd_signing.md) | `ci/sign.py` — Sigstore signing, SBOM generation, rollback gate | ☐ |
| 58 | Consolidation + M1 Gate | [day58_milestone1_gate.md](docs/phase8/day58_milestone1_gate.md) | **MILESTONE 1 GATE** — all 4 gates green | ☐ |

> **M1 Gate — you pass when:** given a prediction, you can trace the model version, data version, code version, feature values, request ID, and decision outcome — and you can roll back, retry a failed job safely, and detect drift/quality/infra/business issues separately. Threat model at **v1**.

---

## Phase 9 — Kubernetes for ML (Days 59–70)
**Tag:** `phase9` *(pending)*
**Milestone:** 2 — Kubernetes & Cloud ML Platform

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 59 | K8s for ML — Fundamentals | [day59_k8s_fundamentals.md](docs/phase9/day59_k8s_fundamentals.md) | `infra/k8s/` — Pod, Deployment, Service manifests; resource limits | ☐ |
| 60 | kind Cluster — Deploy + Ingress | [day60_kind_cluster.md](docs/phase9/day60_kind_cluster.md) | `infra/k8s/serving-deployment.yaml` + ingress + `make k8s-deploy` | ☐ |
| 61 | Helm Chart | [day61_helm_chart.md](docs/phase9/day61_helm_chart.md) | `infra/helm/credit-risk/` — full Helm chart + values.yaml | ☐ |
| 62 | Storage on K8s | [day62_k8s_storage.md](docs/phase9/day62_k8s_storage.md) | PVC templates, init-container model pull, storage strategy doc | ☐ |
| 63 | GPU on K8s | [day63_gpu_k8s.md](docs/phase9/day63_gpu_k8s.md) | NVIDIA GPU Operator config, device plugin, node selector/taint manifests | ☐ |
| 64 | KServe InferenceService | [day64_kserve.md](docs/phase9/day64_kserve.md) | `infra/k8s/kserve-inference.yaml` — predictor + transformer, scale-to-zero | ☐ |
| 65 | KServe Canary & Traffic Split | [day65_kserve_canary.md](docs/phase9/day65_kserve_canary.md) | `infra/k8s/kserve-canary.yaml` — canary %, shadow/mirror config | ☐ |
| 66 | Ray on K8s (KubeRay) | [day66_ray_k8s.md](docs/phase9/day66_ray_k8s.md) | `infra/k8s/ray-cluster.yaml` + Ray Serve multi-model pipeline | ☐ |
| 67 | Autoscaling: HPA + KEDA | [day67_autoscaling.md](docs/phase9/day67_autoscaling.md) | `infra/k8s/hpa.yaml`, `infra/k8s/keda-scaledobject.yaml` | ☐ |
| 68 | Kueue GPU Scheduling | [day68_kueue.md](docs/phase9/day68_kueue.md) | `infra/k8s/kueue/` — ClusterQueue, LocalQueue, job quota manifests | ☐ |
| 69 | Prometheus + Grafana on K8s | [day69_k8s_observability.md](docs/phase9/day69_k8s_observability.md) | `infra/k8s/monitoring/` — ServiceMonitor, Grafana dashboards + RBAC | ☐ |
| 70 | Kubeflow Survey + Consolidation | [day70_kubeflow_survey.md](docs/phase9/day70_kubeflow_survey.md) | Kubeflow Pipelines/Katib/Training Operator comparison doc | ☐ |

---

## Phase 10 — Reliability Lab: Failure Injection (Days 71–73)
**Tag:** `phase10` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 71 | Chaos Fundamentals + Infra Failures | [day71_chaos_infra.md](docs/phase10/day71_chaos_infra.md) | `ci/chaos/` — MLflow down, MinIO down, KServe stuck, GPU node gone | ☐ |
| 72 | ML-Specific Incident Drills | [day72_ml_incidents.md](docs/phase10/day72_ml_incidents.md) | Runbooks: bad artifact pushed, stale features, broken retriever | ☐ |
| 73 | Game Day + Runbooks + Postmortems | [day73_game_day.md](docs/phase10/day73_game_day.md) | `docs/runbooks/` — per-incident runbook templates + postmortem format | ☐ |

---

## Phase 11 — GitOps & Continuous Training (Days 74–77)
**Tag:** `phase11` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 74 | GitOps: Argo CD / Flux | [day74_gitops.md](docs/phase11/day74_gitops.md) | `infra/argocd/` — Application manifests, sync policy, rollback | ☐ |
| 75 | Progressive Delivery for Models | [day75_progressive_delivery.md](docs/phase11/day75_progressive_delivery.md) | Blue-green + canary on K8s — Argo Rollouts config | ☐ |
| 76 | Continuous Training Automation | [day76_continuous_training.md](docs/phase11/day76_continuous_training.md) | `ci/ct_trigger.py` — retrain → registry → deploy; Argo Workflows / Events | ☐ |
| 77 | Consolidation | [day77_consolidation.md](docs/phase11/day77_consolidation.md) | End-to-end CT test: trigger → train → gate → promote → deploy | ☐ |

---

## Phase 12 — Cloud MLOps: AWS Deep, GCP Mapped (Days 78–90) → MILESTONE 2 GATE
**Tag:** `phase12` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 78 | Cloud Landscape + IAM-First | [day78_cloud_landscape.md](docs/phase12/day78_cloud_landscape.md) | Cost model doc; IAM strategy; managed-for-dev vs K8s-for-prod framework | ☐ |
| 79 | AWS Foundations | [day79_aws_foundations.md](docs/phase12/day79_aws_foundations.md) | `infra/aws/` — S3 bucket policy, IAM roles, ECR repo, VPC subnets (IaC) | ☐ |
| 80 | SageMaker Training | [day80_sagemaker_training.md](docs/phase12/day80_sagemaker_training.md) | `ci/sagemaker_train.py` — ProcessingJob + TrainingJob + Experiments | ☐ |
| 81 | SageMaker Registry + Endpoints | [day81_sagemaker_registry.md](docs/phase12/day81_sagemaker_registry.md) | Model package group, endpoint config (real-time/serverless/async/batch) | ☐ |
| 82 | SageMaker Pipelines + Lineage | [day82_sagemaker_pipelines.md](docs/phase12/day82_sagemaker_pipelines.md) | `ci/sagemaker_pipeline.py` — Pipeline DAG, model approval, lineage tracking | ☐ |
| 83 | SageMaker Monitor + Clarify | [day83_sagemaker_monitor.md](docs/phase12/day83_sagemaker_monitor.md) | Data Quality / Model Quality / Clarify bias monitors wired up | ☐ |
| 84 | AWS Serving on EKS + Bedrock | [day84_aws_serving.md](docs/phase12/day84_aws_serving.md) | EKS deployment of credit-risk API; Bedrock architecture overview | ☐ |
| 85 | AWS Cost & Security | [day85_aws_security.md](docs/phase12/day85_aws_security.md) | Spot training config, KMS key policy, PrivateLink endpoint, budget alert | ☐ |
| 86 | Terraform for ML Infra | [day86_terraform.md](docs/phase12/day86_terraform.md) | `infra/terraform/` — S3, ECR, SageMaker domain, EKS node group | ☐ |
| 87 | GCP Mapping 1:1 | [day87_gcp_mapping.md](docs/phase12/day87_gcp_mapping.md) | Vertex AI ↔ SageMaker equivalence table + Vertex pipeline hello-world | ☐ |
| 88 | Platform Portability | [day88_portability.md](docs/phase12/day88_portability.md) | Portability layer doc: MLflow/Feast/K8s as cloud-agnostic core | ☐ |
| 89 | End-to-End Deploy on AWS | [day89_e2e_aws.md](docs/phase12/day89_e2e_aws.md) | Full backbone running on AWS (EKS + S3 + SageMaker + RDS) | ☐ |
| 90 | Consolidation + M2 Gate | [day90_milestone2_gate.md](docs/phase12/day90_milestone2_gate.md) | **MILESTONE 2 GATE** — K8s + AWS, autoscaling, canary, 5 failure recoveries | ☐ |

> **M2 Gate — you pass when:** the platform runs on K8s + AWS, fully IaC'd, with autoscaling, canary, proven rollback, cost controls, and you've recovered from at least 5 injected failures with documented runbooks. Threat model at **v2**.

---

## Phase 13 — Scaling & Inference Optimization (Days 91–99)
**Tag:** `phase13` *(pending)*
**Milestone:** 3 — Production RAG / LLMOps

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 91 | Distributed Training Theory | [day91_distributed_training.md](docs/phase13/day91_distributed_training.md) | Data/model/pipeline/tensor parallelism; DDP, FSDP, ZeRO deep-dive | ☐ |
| 92 | Ray Train Multi-GPU | [day92_ray_train.md](docs/phase13/day92_ray_train.md) | `llm/ray_train_job.py` — multi-GPU training job with Ray Train | ☐ |
| 93 | Training Optimization | [day93_training_optimization.md](docs/phase13/day93_training_optimization.md) | `llm/train_optimized.py` — mixed precision, gradient checkpointing, data loading | ☐ |
| 94 | Inference Optimization Theory | [day94_inference_optimization.md](docs/phase13/day94_inference_optimization.md) | KV cache, PagedAttention, continuous batching, batching strategies | ☐ |
| 95 | Quantization for Serving | [day95_quantization.md](docs/phase13/day95_quantization.md) | `llm/quantize.py` — PTQ/QAT, GPTQ/AWQ evaluation, distillation pipeline | ☐ |
| 96 | Compilation + Runtimes | [day96_runtimes.md](docs/phase13/day96_runtimes.md) | ONNX Runtime, TensorRT-LLM, `torch.compile` benchmark harness | ☐ |
| 97 | GPU Utilization & Cost | [day97_gpu_cost.md](docs/phase13/day97_gpu_cost.md) | MIG partition config, spot strategy, idle GPU detection script | ☐ |
| 98 | vLLM Single-Node Deep | [day98_vllm_single_node.md](docs/phase13/day98_vllm_single_node.md) | `llm/vllm_serve.py` — vLLM server config, benchmark, throughput profiling | ☐ |
| 99 | vLLM on K8s | [day99_vllm_k8s.md](docs/phase13/day99_vllm_k8s.md) | `infra/k8s/vllm-deployment.yaml` + GPU metrics + capacity planning doc | ☐ |

---

## Phase 14 — LLMOps Core (Days 100–108)
**Tag:** `phase14` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 100 | LLMOps vs MLOps | [day100_llmops_vs_mlops.md](docs/phase14/day100_llmops_vs_mlops.md) | Prompts-as-artifacts, non-determinism handling, cost-as-metric patterns | ☐ |
| 101 | Serving LLMs on K8s | [day101_llm_serving.md](docs/phase14/day101_llm_serving.md) | `infra/k8s/kserve-llm.yaml` — KServe LLMInferenceService / Ray Serve | ☐ |
| 102 | Prompt Management & Versioning | [day102_prompt_management.md](docs/phase14/day102_prompt_management.md) | `llm/prompt_registry.py` — prompts-as-code, versioned registry, A/B config | ☐ |
| 103 | LLM Eval I — Offline | [day103_llm_eval_offline.md](docs/phase14/day103_llm_eval_offline.md) | `llm/eval_offline.py` — reference-based / free / LLM-as-judge eval harness | ☐ |
| 104 | LLM Eval II — RAGAS | [day104_ragas.md](docs/phase14/day104_ragas.md) | `llm/eval_ragas.py` — faithfulness, context relevance, answer correctness | ☐ |
| 105 | Fine-Tuning Ops | [day105_finetuning_ops.md](docs/phase14/day105_finetuning_ops.md) | `llm/finetune.py` — LoRA/QLoRA pipeline, dataset versioning, eval-gated gate | ☐ |
| 106 | LLM Observability | [day106_llm_observability.md](docs/phase14/day106_llm_observability.md) | `llm/otel_tracer.py` — OTel GenAI trace: reasoning→tool→guardrail→response | ☐ |
| 107 | LLM Monitoring in Prod | [day107_llm_monitoring.md](docs/phase14/day107_llm_monitoring.md) | `llm/quality_monitor.py` — hallucination drift, online eval on sampled traffic | ☐ |
| 108 | LLM Gateway Architecture | [day108_llm_gateway.md](docs/phase14/day108_llm_gateway.md) | `llm/gateway.py` — model routing, quota enforcement, semantic caching | ☐ |

---

## Phase 15 — RAG Production Operations (Days 109–114) → MILESTONE 3 GATE
**Tag:** `phase15` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 109 | Index Build Pipeline | [day109_index_pipeline.md](docs/phase15/day109_index_pipeline.md) | `llm/index_pipeline.py` — build, version, rollback index | ☐ |
| 110 | Chunking + Hybrid Retrieval | [day110_hybrid_retrieval.md](docs/phase15/day110_hybrid_retrieval.md) | `llm/retriever.py` — BM25 + vector hybrid, reranker | ☐ |
| 111 | Multi-Tenant Retrieval Security | [day111_rag_security.md](docs/phase15/day111_rag_security.md) | `llm/acl_filter.py` — metadata filtering, document ACL propagation | ☐ |
| 112 | Stale Docs + Embedding Migration | [day112_rag_maintenance.md](docs/phase15/day112_rag_maintenance.md) | `llm/index_maintenance.py` — stale removal, embedding model migration | ☐ |
| 113 | Retrieval Failure Taxonomy | [day113_retrieval_eval.md](docs/phase15/day113_retrieval_eval.md) | `llm/golden_query_set.py` — golden set, synthetic query gen, failure taxonomy | ☐ |
| 114 | RAG Guardrails + M3 Gate | [day114_milestone3_gate.md](docs/phase15/day114_milestone3_gate.md) | `llm/guardrails.py` — prompt injection, source trust, Llama Guard + **M3 GATE** | ☐ |

> **M3 Gate — you pass when:** for any answer you can prove "this came from these retrieved chunks, using this prompt version, this embedding model, this index version, this LLM version, and this eval score" — with guardrails active and cost tracked. Threat model at **v3**.

---

## Phase 16 — AgentOps Core (Days 115–122)
**Tag:** `phase16` *(pending)*
**Milestone:** 4 — Production AgentOps

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 115 | Why AgentOps is Distinct | [day115_agentops_intro.md](docs/phase16/day115_agentops_intro.md) | Agent lifecycle doc; agent threat model started | ☐ |
| 116 | Agent Observability Fundamentals | [day116_agent_observability.md](docs/phase16/day116_agent_observability.md) | `agent/tracer.py` — span taxonomy, trace-per-tick, OTel GenAI canonical trace | ☐ |
| 117 | Instrumenting Agents | [day117_agent_instrumentation.md](docs/phase16/day117_agent_instrumentation.md) | `agent/session_replay.py` — AgentOps SDK integration, session replay | ☐ |
| 118 | Agent Eval I — Trajectory | [day118_agent_eval_trajectory.md](docs/phase16/day118_agent_eval_trajectory.md) | `agent/trajectory_eval.py` — tool-use correctness, task success, step efficiency | ☐ |
| 119 | Agent Eval II — LLM-as-Judge | [day119_agent_eval_judge.md](docs/phase16/day119_agent_eval_judge.md) | `agent/composite_eval.py` — composite metrics, gatekeeping, Agent Evals via MCP | ☐ |
| 120 | Agent Testing | [day120_agent_testing.md](docs/phase16/day120_agent_testing.md) | `agent/simulation.py` — simulation environments, scenario/replay, regression suite | ☐ |
| 121 | Agent Reliability | [day121_agent_reliability.md](docs/phase16/day121_agent_reliability.md) | `agent/circuit_breaker.py` — retries, fallbacks, timeouts, runaway-loop detection | ☐ |
| 122 | Agent Memory & State Ops | [day122_agent_memory.md](docs/phase16/day122_agent_memory.md) | `agent/memory.py` — short/long-term memory, vector memory, persistence & recovery | ☐ |

---

## Phase 17 — Agent Security & Tool Safety (Days 123–127)
**Tag:** `phase17` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 123 | Tool Permission Model | [day123_tool_permissions.md](docs/phase17/day123_tool_permissions.md) | `agent/permissions.py` — per-tool scopes, user identity propagation | ☐ |
| 124 | Tool Approval Policies | [day124_tool_approval.md](docs/phase17/day124_tool_approval.md) | `agent/approval_policy.py` — high-risk action classifier, dry-run mode | ☐ |
| 125 | Tool Budget + Sandbox | [day125_tool_sandbox.md](docs/phase17/day125_tool_sandbox.md) | `agent/sandbox.py` — call/timeout budgets, sandbox exec, result validation | ☐ |
| 126 | MCP Trust + Audit Log + Kill Switch | [day126_mcp_trust.md](docs/phase17/day126_mcp_trust.md) | `agent/audit_log.py` — MCP server trust levels, structured audit log, kill switch | ☐ |
| 127 | Agent Failure Injection | [day127_agent_chaos.md](docs/phase17/day127_agent_chaos.md) | `agent/failure_injection.py` — tool timeout, infinite loop, guardrail-service down | ☐ |

---

## Phase 18 — Agent Deployment & Multi-Agent (Days 128–130) → MILESTONE 4 GATE
**Tag:** `phase18` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 128 | Multi-Agent Ops | [day128_multi_agent.md](docs/phase18/day128_multi_agent.md) | `agent/orchestrator.py` — message tracing, hierarchical debugging | ☐ |
| 129 | Agent Deployment Patterns | [day129_agent_deployment.md](docs/phase18/day129_agent_deployment.md) | K8s long-running agent + async queue + human-in-the-loop approval gate | ☐ |
| 130 | Consolidation + M4 Gate | [day130_milestone4_gate.md](docs/phase18/day130_milestone4_gate.md) | **MILESTONE 4 GATE** — session replay + kill switch + full audit trail | ☐ |

> **M4 Gate — you pass when:** you can replay an agent session and explain every tool call, failure, retry, permission, cost, and output — with a working kill switch and audit trail. Threat model at **v4**.

---

## Phase 19 — Security, Governance & Responsible AI (Days 131–138)
**Tag:** `phase19` *(pending)*
**Milestone:** 5 — Governance, Capstone & SOTA

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 131 | MLSecOps: Threat Model Consolidation | [day131_mlsecops.md](docs/phase19/day131_mlsecops.md) | Lifecycle threat model v4 consolidated + gap analysis | ☐ |
| 132 | Supply Chain Security | [day132_supply_chain.md](docs/phase19/day132_supply_chain.md) | `ci/sbom.py` — SBOM generation, Sigstore signing, provenance chain | ☐ |
| 133 | Adversarial & Privacy Attacks | [day133_adversarial.md](docs/phase19/day133_adversarial.md) | Evasion, membership inference, model inversion/extraction defenses | ☐ |
| 134 | Privacy-Preserving ML | [day134_privacy_ml.md](docs/phase19/day134_privacy_ml.md) | PII handling, differential privacy basics, federated inference overview | ☐ |
| 135 | Access Control at Scale | [day135_access_control.md](docs/phase19/day135_access_control.md) | RBAC manifests, secret rotation runbook, KMS/CMEK key policy | ☐ |
| 136 | Model Governance | [day136_model_governance.md](docs/phase19/day136_model_governance.md) | Model card template, registry-as-governance, approval workflow | ☐ |
| 137 | Regulatory + Fairness Ops | [day137_regulatory.md](docs/phase19/day137_regulatory.md) | EU AI Act / NIST AI RMF mapping; Clarify/SHAP gate checklist | ☐ |
| 138 | Governance Evidence Pack | [day138_governance_pack.md](docs/phase19/day138_governance_pack.md) | `docs/governance/` — model card + data card + eval card + risk register | ☐ |

---

## Phase 20 — Capstone & State-of-the-Art (Days 139–148) → MILESTONE 5 GATE
**Tag:** `phase20` *(pending)*

### Day Table

| Day | Title | Theory | Deliverable | Status |
|---|---|---|---|---|
| 139 | Capstone: Era A Integration | [day139_capstone_era_a.md](docs/phase20/day139_capstone_era_a.md) | Classical MLOps (credit-risk) — all gates green, CI/CD/CT, K8s + AWS | ☐ |
| 140 | Capstone: Era B Integration | [day140_capstone_era_b.md](docs/phase20/day140_capstone_era_b.md) | LLMOps (RAG assistant) — vLLM serving, RAGAS eval, prompt registry | ☐ |
| 141 | Capstone: Era C Integration | [day141_capstone_era_c.md](docs/phase20/day141_capstone_era_c.md) | AgentOps (support agent) — risk model + RAG assistant, MCP tools | ☐ |
| 142 | Capstone: Three-Era Unification | [day142_capstone_unify.md](docs/phase20/day142_capstone_unify.md) | Single platform — one trace through all three eras end-to-end | ☐ |
| 143 | Capstone: All Six Gates Green | [day143_all_gates.md](docs/phase20/day143_all_gates.md) | Reproducibility ✅ Serving ✅ Pipeline ✅ Monitoring ✅ Security ✅ AgentOps ✅ | ☐ |
| 144 | Capstone: Terraform + Full IaC | [day144_capstone_iac.md](docs/phase20/day144_capstone_iac.md) | Entire platform Terraform-managed + DR runbook | ☐ |
| 145 | SOTA Serving: llm-d | [day145_sota_serving.md](docs/phase20/day145_sota_serving.md) | Disaggregated inference, prefix-cache-aware routing, llm-d survey | ☐ |
| 146 | SOTA Eval + Self-Improving Loops | [day146_sota_eval.md](docs/phase20/day146_sota_eval.md) | Full-traffic online eval economics, self-improving eval loop design | ☐ |
| 147 | Frontier Research | [day147_frontier.md](docs/phase20/day147_frontier.md) | Federated/edge inference, agentic infrastructure research, how to stay current | ☐ |
| 148 | Retrospective + Portfolio | [day148_retrospective.md](docs/phase20/day148_retrospective.md) | Portfolio doc + golden-path platform template + **MILESTONE 5 GATE** | ☐ |

> **M5 Gate — you pass when:** all six production gates are green across all three eras, the platform is fully IaC'd, and you can hand a stranger a `git clone` + `make up` that gives them the entire running system.

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
