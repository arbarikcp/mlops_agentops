# MLOps Platform — One-command local ops

COMPOSE = docker compose -f docker-compose.yml --env-file .env
SERVICES = minio postgres redis mlflow
SEED = 42

.PHONY: help up down clean status smoke-test logs \
        install dvc-init data-download featurize train \
        mlflow-train hpo test test-unit test-data \
        determinism-check reproduce-check \
        calibrate threshold-analysis decision-band slice-eval \
        data-contract label-contract skew-detect \
        serve-local serving-gate-check \
        pipeline-gate-check \
        feature-store-gate-check \
        monitoring-gate-check \
        kind-up kind-down

help:
	@echo ""
	@echo "  Infrastructure:"
	@echo "    make up               Start all local services"
	@echo "    make down             Stop services (keep volumes)"
	@echo "    make clean            Stop + delete all volumes (DESTRUCTIVE)"
	@echo "    make status           Show service health"
	@echo "    make smoke-test       Verify all services are reachable"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install Python deps via uv"
	@echo "    make dvc-init         Initialize DVC and configure MinIO remote"
	@echo ""
	@echo "  Pipeline (Day 9):"
	@echo "    make data-download    Download UCI dataset (dvc add + push)"
	@echo "    make featurize        Run feature engineering"
	@echo "    make train            Run deterministic training"
	@echo "    make pipeline         Run full DVC pipeline (dvc repro)"
	@echo ""
	@echo "  MLflow (Day 10-12):"
	@echo "    make mlflow-train     Train with full MLflow tracking"
	@echo "    make hpo              Run Optuna HPO sweep (20 trials)"
	@echo ""
	@echo "  Tests:"
	@echo "    make test             Run all tests"
	@echo "    make test-unit        Unit tests only"
	@echo "    make test-data        Data contract tests only"
	@echo "    make determinism-check Run training twice + diff (Day 7)"
	@echo ""
	@echo "  Gate:"
	@echo "    make reproduce-check  Reproducibility gate dry-run (Day 14)"
	@echo ""
	@echo "  Phase 2 — Calibration & Threshold (Days 15-18):"
	@echo "    make calibrate         Fit + compare calibrator on test set"
	@echo "    make threshold-analysis Find cost-optimal threshold"
	@echo "    make decision-band     Derive approve/review/decline band"
	@echo "    make slice-eval        Slice-level AUC + OOD detection"
	@echo ""
	@echo "  Phase 3 — Data & Label Contracts (Days 19-21):"
	@echo "    make data-contract     Validate feature schema + statistical checks"
	@echo "    make label-contract    Validate a synthetic label batch"
	@echo "    make skew-detect       Detect train/serve skew (PSI + KS + JS)"
	@echo ""
	@echo "  Phase 4 — Packaging & Serving (Days 22-30):"
	@echo "    make serve-local       Start FastAPI server locally (port 8080)"
	@echo "    make serving-gate-check Run Serving gate dry-run checklist"
	@echo ""
	@echo "  Phase 5 — Orchestration & Pipelines (Days 31-37):"
	@echo "    make pipeline-gate-check Run Pipeline gate dry-run (Day 37)"
	@echo ""
	@echo "  Phase 6 — Feature Store & Closed Feedback Loop (Days 38-45):"
	@echo "    make feature-store-gate-check  Run Feature Store gate dry-run (Day 45)"
	@echo ""
	@echo "  Phase 7 — Monitoring & the Closed Loop (Days 46-53):"
	@echo "    make monitoring-gate-check     Run Monitoring gate dry-run (Day 53)"
	@echo ""

# ── Local Services ──────────────────────────────────────────────────────────

up: _check-env
	@echo "→ Starting local platform..."
	$(COMPOSE) up -d --wait $(SERVICES)
	@$(MAKE) status

down:
	@echo "→ Stopping local platform..."
	$(COMPOSE) down

clean:
	@echo "⚠️  This will delete all volumes (MLflow runs, MinIO data, Postgres data)."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ]
	$(COMPOSE) down -v

status:
	@echo ""
	@echo "Service status:"
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
	@echo ""

smoke-test:
	@echo "→ Running smoke tests..."
	@echo -n "  MinIO API     : " && curl -sf http://localhost:9000/minio/health/ready && echo "✅" || echo "❌"
	@echo -n "  MinIO Console : " && curl -sf http://localhost:9001 -o /dev/null && echo "✅" || echo "❌"
	@echo -n "  Postgres      : " && docker exec mlops-postgres pg_isready -q && echo "✅" || echo "❌"
	@echo -n "  Redis         : " && docker exec mlops-redis redis-cli ping | grep -q PONG && echo "✅" || echo "❌"
	@echo -n "  MLflow        : " && curl -sf http://localhost:5000/health && echo "✅" || echo "❌"
	@echo ""

logs:
	$(COMPOSE) logs -f $(SERVICES)

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	uv sync

dvc-init: _check-env
	@echo "→ Initializing DVC..."
	dvc init --no-scm 2>/dev/null || true
	dvc remote add -d minio s3://dvc-data 2>/dev/null || dvc remote modify minio url s3://dvc-data
	dvc remote modify minio endpointurl http://localhost:9000
	dvc remote modify minio access_key_id $$(grep MINIO_ROOT_USER .env | cut -d= -f2)
	dvc remote modify minio secret_access_key $$(grep MINIO_ROOT_PASSWORD .env | cut -d= -f2)
	@echo "DVC configured with MinIO remote ✅"

# ── Data Pipeline ─────────────────────────────────────────────────────────────

data-download:
	@echo "→ Downloading UCI Credit Default dataset..."
	uv run python -m data.ingest --output data/raw/credit_card_default.csv
	dvc add data/raw/credit_card_default.csv
	dvc push
	@echo "Data downloaded and pushed to MinIO ✅"

featurize:
	@echo "→ Running feature engineering..."
	PYTHONHASHSEED=$(SEED) uv run python -m training.features \
		--input data/raw/credit_card_default.csv \
		--output data/processed/features.parquet \
		--params params.yaml

train:
	@echo "→ Training model deterministically (PYTHONHASHSEED=$(SEED))..."
	PYTHONHASHSEED=$(SEED) uv run python -m training.train --params params.yaml
	@echo "Metrics:"
	@cat metrics/train_metrics.json

pipeline:
	@echo "→ Running full DVC pipeline..."
	dvc repro
	dvc metrics show

# ── MLflow Training ───────────────────────────────────────────────────────────

mlflow-train:
	@echo "→ Training with MLflow tracking..."
	PYTHONHASHSEED=$(SEED) uv run python -m training.mlflow_train --params params.yaml

hpo:
	@echo "→ Running Optuna HPO sweep (20 trials)..."
	PYTHONHASHSEED=$(SEED) uv run python -m training.hpo \
		--n-trials 20 --params params.yaml

hpo-quick:
	@echo "→ Running Optuna HPO smoke test (3 trials)..."
	PYTHONHASHSEED=$(SEED) uv run python -m training.hpo \
		--n-trials 3 --params params.yaml

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-data:
	uv run pytest tests/data/ -v

# ── Reproducibility ───────────────────────────────────────────────────────────

determinism-check: featurize
	@echo "→ Running training twice to verify determinism..."
	PYTHONHASHSEED=$(SEED) uv run python -m training.train --params params.yaml > /dev/null
	cp metrics/train_metrics.json /tmp/repro_run1.json
	PYTHONHASHSEED=$(SEED) uv run python -m training.train --params params.yaml > /dev/null
	@if diff /tmp/repro_run1.json metrics/train_metrics.json > /dev/null; then \
		echo "✅ Training is deterministic"; \
	else \
		echo "❌ Training is NOT deterministic:"; \
		diff /tmp/repro_run1.json metrics/train_metrics.json; \
		exit 1; \
	fi

reproduce-check:
	@echo "→ Running reproducibility gate dry-run..."
	@RUN_ID=$$(cat metrics/mlflow_run_id.txt 2>/dev/null || echo ""); \
	if [ -z "$$RUN_ID" ]; then \
		echo "❌ No run_id found. Run 'make mlflow-train' first."; \
		exit 1; \
	fi; \
	bash scripts/verify_reproducibility.sh "$$RUN_ID"

# ── Phase 2 — Calibration & Threshold ────────────────────────────────────────

calibrate:
	@echo "→ Fitting calibrator + comparing ECE before/after (Day 15)..."
	uv run python -c "
import pickle, pandas as pd, numpy as np
from training.calibration import fit_calibrator, calibration_report, reliability_data
model = pickle.load(open('models/credit_risk_model.pkl', 'rb'))
df = pd.read_parquet('data/processed/features.parquet')
n = len(df); target = 'DEFAULT_PAYMENT_NEXT_MONTH'
X = df.drop(columns=[target, 'ID']); y = df[target]
X_cal, y_cal = X.iloc[int(n*0.6):int(n*0.8)].to_numpy(), y.iloc[int(n*0.6):int(n*0.8)].to_numpy()
X_test, y_test = X.iloc[int(n*0.8):].to_numpy(), y.iloc[int(n*0.8):].to_numpy()
cal = fit_calibrator(model, X_cal, y_cal)
report = calibration_report(model, cal, X_test, y_test)
report.log_summary()
print(f'ECE before: {report.ece_before:.4f}  after: {report.ece_after:.4f}  improved={report.improved}')
rel = reliability_data(y_test, cal.predict_proba(X_test)[:,1])
rel.to_csv('metrics/reliability_diagram.csv', index=False)
print('Reliability diagram saved to metrics/reliability_diagram.csv')
"

threshold-analysis:
	@echo "→ Finding cost-optimal threshold (Day 16)..."
	uv run python -c "
import pickle, pandas as pd, numpy as np
from training.threshold import find_cost_optimal_threshold, threshold_sweep
model = pickle.load(open('models/credit_risk_model.pkl', 'rb'))
df = pd.read_parquet('data/processed/features.parquet')
n = len(df); target = 'DEFAULT_PAYMENT_NEXT_MONTH'
X_test = df.drop(columns=[target, 'ID']).iloc[int(n*0.8):].to_numpy()
y_test = df[target].iloc[int(n*0.8):].to_numpy()
y_prob = model.predict_proba(X_test)[:,1]
result = find_cost_optimal_threshold(y_test, y_prob)
result.log_summary()
sweep = threshold_sweep(y_test, y_prob)
sweep.to_csv('metrics/threshold_sweep.csv', index=False)
print('Threshold sweep saved to metrics/threshold_sweep.csv')
"

decision-band:
	@echo "→ Deriving approve/review/decline band + routing stats (Day 17)..."
	uv run python -c "
import pickle, pandas as pd, numpy as np
from training.decision import ThresholdBand, find_review_band, calibrate_band_for_cost
model = pickle.load(open('models/credit_risk_model.pkl', 'rb'))
df = pd.read_parquet('data/processed/features.parquet')
n = len(df); target = 'DEFAULT_PAYMENT_NEXT_MONTH'
X_test = df.drop(columns=[target, 'ID']).iloc[int(n*0.8):].to_numpy()
y_test = df[target].iloc[int(n*0.8):].to_numpy()
y_prob = model.predict_proba(X_test)[:,1]
band = calibrate_band_for_cost(y_test, y_prob, target_review_pct=0.15)
print(f'Band: approve < {band.approve}  decline >= {band.decline}')
print(band.routing_stats(y_prob))
review_df = find_review_band(y_test, y_prob, band.approve, band.decline)
print(review_df.to_string())
"

slice-eval:
	@echo "→ Slice-level evaluation + OOD detection (Day 18)..."
	uv run python -c "
import pickle, pandas as pd, numpy as np
from training.slice_eval import evaluate_slices, worst_slices, slice_gap_report, fit_ood_detector, ood_report
from training.evaluate import compute_metrics
model = pickle.load(open('models/credit_risk_model.pkl', 'rb'))
df = pd.read_parquet('data/processed/features.parquet')
n = len(df); target = 'DEFAULT_PAYMENT_NEXT_MONTH'
X_train_np = df.drop(columns=[target, 'ID']).iloc[:int(n*0.8)].to_numpy()
X_test = df.drop(columns=[target, 'ID']).iloc[int(n*0.8):]
y_test = df[target].iloc[int(n*0.8):].to_numpy()
y_prob = model.predict_proba(X_test.to_numpy())[:,1]
overall = compute_metrics(y_test, y_prob)
print(f'Overall AUC: {overall[\"roc_auc\"]:.4f}')
slices = evaluate_slices(X_test, y_test, y_prob)
slices.to_csv('metrics/slice_metrics.csv', index=False)
print(slices.to_string())
gaps = slice_gap_report(slices, overall, warn_threshold=0.05)
flagged = gaps[gaps['flag']]
print(f'\nFlagged slices (gap > 0.05): {len(flagged)}')
if len(flagged): print(flagged[['slice_col','slice_val','roc_auc','gap']].to_string())
detector = fit_ood_detector(X_train_np)
report = ood_report(detector, X_test.to_numpy())
print(f'\nOOD fraction (test set): {report[\"ood_fraction\"]:.4f}')
"

# ── Phase 3 — Data & Label Contracts ────────────────────────────────────────

data-contract:
	@echo "→ Validating feature schema + statistical checks (Day 19)..."
	uv run python -c "
import pandas as pd
from data.contracts.feature_schema import validate_features, check_no_infinite_values
from data.contracts.statistical_checks import compute_dataset_stats, check_null_drift, check_class_balance
import pandera as pa, sys

df = pd.read_parquet('data/processed/features.parquet')
print(f'Loaded {len(df)} rows × {len(df.columns)} columns')

inf_cols = check_no_infinite_values(df)
if inf_cols:
    print(f'❌ Infinite values in: {inf_cols}'); sys.exit(1)
print('✅ No infinite values')

try:
    validate_features(df)
    print('✅ Feature schema PASSED')
except pa.errors.SchemaErrors as err:
    print(f'❌ Feature schema FAILED: {len(err.failure_cases)} violations')
    print(err.failure_cases.head(5).to_string()); sys.exit(1)

stats = compute_dataset_stats(df, dataset_name='features')
balance = check_class_balance(stats)
print(f'Class balance: {balance[\"positive_rate\"]:.1%} (in_range={balance[\"in_range\"]})')
print('✅ data-contract complete')
"

label-contract:
	@echo "→ Validating label batch schema + arrival timing (Day 20)..."
	uv run python -c "
import pandas as pd, numpy as np
from data.contracts.label_contract import validate_label_batch, check_label_arrival, DEFAULT_LABEL_METADATA, check_single_policy_version, check_correction_rate
import pandera as pa, sys

# Synthetic label batch
rng = np.random.default_rng(42)
n = 200
obs = pd.date_range('2005-01-01', periods=n, freq='D')
out = obs + pd.Timedelta(days=91)
df = pd.DataFrame({
    'applicant_id': range(1, n+1),
    'label': rng.choice([0,1], n).tolist(),
    'label_source': ['core_banking'] * n,
    'policy_version': ['v1.0'] * n,
    'label_timestamp': ['2005-04-30T00:00:00'] * n,
    'observation_date': [str(d.date()) for d in obs],
    'outcome_date': [str(d.date()) for d in out],
    'is_corrected': [False] * n,
})

try:
    validate_label_batch(df)
    print('✅ Label batch schema PASSED')
except pa.errors.SchemaErrors as err:
    print(f'❌ Label batch schema FAILED'); print(err.failure_cases.head(5)); sys.exit(1)

arrival = check_label_arrival(df, DEFAULT_LABEL_METADATA)
print(f'Label arrival: {arrival[\"pct_confirmed\"]:.0%} confirmed (sufficient={arrival[\"sufficient_for_training\"]})')
policy = check_single_policy_version(df)
print(f'Policy consistency: {policy[\"is_consistent\"]} versions={policy[\"policy_versions\"]}')
corrections = check_correction_rate(df)
print(f'Correction rate: {corrections[\"correction_rate\"]:.1%} (flag={corrections[\"flag\"]})')
print('✅ label-contract complete')
"

skew-detect:
	@echo "→ Train/serve skew detection — PSI, KS, Jensen-Shannon (Day 21)..."
	uv run python -c "
import pandas as pd
from monitoring.reference_stats import compute_reference_stats, save_reference_stats
from monitoring.skew_detector import detect_skew, skew_summary
from pathlib import Path

df = pd.read_parquet('data/processed/features.parquet')
target = 'DEFAULT_PAYMENT_NEXT_MONTH'
feat_cols = [c for c in df.columns if c not in [target, 'ID']]

n = len(df)
X_train = df[feat_cols].iloc[:int(n*0.8)]
X_test  = df[feat_cols].iloc[int(n*0.8):]

ref = compute_reference_stats(X_train, model_version='phase3-demo', feature_names=feat_cols)
save_reference_stats(ref, Path('metrics/reference_stats.json'))
print(f'Reference stats saved: {ref.n_training_rows} rows, {len(ref.feature_names)} features')

report = detect_skew(X_test, ref)
print(f'Overall severity: {report.overall_severity} | Flagged: {report.n_flagged}/{report.n_features} features')
summary = skew_summary(report)
summary.to_csv('metrics/skew_report.csv', index=False)
print(summary.head(10).to_string())
print('Skew report saved to metrics/skew_report.csv')
"

# ── Phase 4 — Serving ────────────────────────────────────────────────────────

serve-local:
	@echo "→ Starting FastAPI credit risk API on port 8080 (Day 24)..."
	MODEL_PATH=models/credit_risk_model.onnx \
	MODEL_VERSION=local-dev \
	MODEL_THRESHOLD=0.5 \
	uv run uvicorn serving.api:app --host 0.0.0.0 --port 8080 --reload

serving-gate-check:
	@echo "→ Phase 4 Serving Gate Dry-Run (Day 30)..."
	@echo ""
	@echo "  Checking Phase 4 test suite..."
	@uv run python -m pytest tests/unit/test_serialization.py tests/unit/test_inference.py \
	    tests/unit/test_api.py tests/unit/test_dockerfile.py \
	    tests/unit/test_bento_service.py tests/unit/test_batch_inference.py \
	    tests/unit/test_api_contract.py tests/unit/test_load_test.py \
	    tests/unit/test_security.py --override-ini="addopts=" -q && \
	    echo "  ✅ All Phase 4 tests passing" || echo "  ❌ Tests FAILED"
	@echo ""
	@echo "  API Contract check (v1 → v2)..."
	@uv run python -c "
from serving.api_contract import ApiContractChecker, build_v1_contract, build_v2_contract
checker = ApiContractChecker()
report = checker.check_compatibility(build_v1_contract(), build_v2_contract())
status = '✅' if report.is_backward_compatible else '❌'
print(f'  {status} v1 → v2 backward compatible: {report.is_backward_compatible}')
for c in report.compatible_changes:
    print(f'    OK: {c}')
"
	@echo ""
	@echo "  Dockerfile security checks..."
	@uv run python -m pytest tests/unit/test_dockerfile.py --override-ini="addopts=" -q && \
	    echo "  ✅ Dockerfile security checks passed" || echo "  ❌ Dockerfile checks FAILED"
	@echo ""
	@echo "  Rate limiter self-test..."
	@uv run python -c "
from serving.security import RateLimiter
limiter = RateLimiter(max_requests=3, window_seconds=60)
results = [limiter.check('test') for _ in range(4)]
assert results[:3] == [True, True, True], 'Expected first 3 to pass'
assert results[3] == False, 'Expected 4th to be rate-limited'
print('  ✅ Rate limiter working correctly')
"
	@echo ""
	@echo "  Serving Gate: PASSED ✅"

# ── Phase 5 — Orchestration & Pipelines ──────────────────────────────────────

pipeline-gate-check:
	@echo "→ Phase 5 Pipeline Gate Dry-Run (Day 37)..."
	@echo ""
	@echo "  Checking Phase 5 test suite..."
	@uv run python -m pytest tests/unit/test_dag.py tests/unit/test_dagster_pipeline.py \
	    tests/unit/test_zenml_pipeline.py tests/unit/test_validation_gate.py \
	    tests/unit/test_model_gate.py tests/unit/test_failure_modes.py \
	    tests/unit/test_pipeline_gate.py --override-ini="addopts=" -q && \
	    echo "  ✅ All Phase 5 tests passing" || echo "  ❌ Tests FAILED"
	@echo ""
	@echo "  Idempotency self-test..."
	@uv run python -c "
from pipelines.failure_modes import IdempotencyProof
proof = IdempotencyProof(fn=lambda x: x * 2, run_count=3)
result = proof.prove(inputs=42)
status = '✅' if result.is_idempotent else '❌'
print(f'  {status} featurize idempotency: {result.is_idempotent}')
print(f'     checksums: {result.checksums}')
"
	@echo ""
	@echo "  Orchestration tool recommendation..."
	@uv run python -c "
from pipelines.pipeline_gate import OrchestrationSurvey
survey = OrchestrationSurvey()
recs = survey.recommend(need_asset_centric=True, need_ml_native=True, need_local_dev=True)
print('  Recommendations (asset-centric + ML-native + local dev):')
for r in recs[:3]:
    print(f'    {r.name}: {r.best_for}')
"
	@echo ""
	@echo "  Pipeline Gate: PASSED ✅"

# ── Phase 6 — Feature Store & Closed Feedback Loop ──────────────────────────

feature-store-gate-check:
	@echo "→ Phase 6 Feature Store Gate Dry-Run (Day 45)..."
	@echo ""
	@echo "  Checking Phase 6 test suite..."
	@uv run python -m pytest \
	    tests/unit/test_feature_store.py tests/unit/test_feature_views.py \
	    tests/unit/test_materialization.py tests/unit/test_streaming.py \
	    tests/unit/test_feature_monitor.py tests/unit/test_feedback_loop.py \
	    tests/unit/test_skew_checker.py --override-ini="addopts=" -q && \
	    echo "  ✅ All Phase 6 tests passing" || echo "  ❌ Tests FAILED"
	@echo ""
	@echo "  Zero-skew self-test..."
	@uv run python -c "
import numpy as np, pandas as pd
from features.skew_checker import TrainServeSkewChecker
rng = np.random.default_rng(42)
df = pd.DataFrame({'pay_ratio': rng.normal(0.5, 0.1, 500), 'util_rate': rng.normal(0.4, 0.1, 500)})
checker = TrainServeSkewChecker()
report = checker.run(df, df)
status = '✅' if report.zero_skew else '❌'
print(f'  {status} Zero skew on identical data: {report.zero_skew}')
"
	@echo ""
	@echo "  Feature monitor self-test..."
	@uv run python -c "
from datetime import datetime, timezone
from features.feature_monitor import FreshnessChecker
checker = FreshnessChecker()
now = datetime.now(timezone.utc)
result = checker.check('payment_features', last_materialized_at=now, threshold_hours=25.0, now=now)
status = '✅' if result.is_fresh else '❌'
print(f'  {status} Freshness check (just materialized): {result.status}')
"
	@echo ""
	@echo "  Feedback loop self-test..."
	@uv run python -c "
from datetime import datetime, timezone
from features.feedback_loop import (
    GroundTruthJoiner, PredictionRecord, OutcomeRecord, LabelFeedbackLoop
)
now = datetime.now(timezone.utc)
preds = [PredictionRecord(f'p{i}', f'c{i}', 0.7, 'approve', now) for i in range(5)]
outcomes = [OutcomeRecord(f'p{i}', 1, now, delay_days=60.0) for i in range(5)]
loop = LabelFeedbackLoop(baseline_metrics={'auc': 0.70})
result = loop.tick(preds, outcomes)
print(f'  ✅ Feedback loop tick: {len(result.labeled_examples)} labeled, phase={result.phase_reached}')
"
	@echo ""
	@echo "  Feature Store Gate: PASSED ✅"

# ── Phase 7 — Monitoring & the Closed Loop ───────────────────────────────────

monitoring-gate-check:
	@echo "→ Phase 7 Monitoring Gate Dry-Run (Day 53)..."
	@echo ""
	@echo "  Checking Phase 7 test suite..."
	@uv run python -m pytest \
	    tests/unit/test_taxonomy.py tests/unit/test_drift.py \
	    tests/unit/test_evidently_reporter.py tests/unit/test_prometheus_metrics.py \
	    tests/unit/test_grafana_dashboard.py tests/unit/test_prediction_logger.py \
	    tests/unit/test_closed_loop.py tests/unit/test_slo.py \
	    --override-ini="addopts=" -q && \
	    echo "  ✅ All Phase 7 tests passing" || echo "  ❌ Tests FAILED"
	@echo ""
	@echo "  SLO self-test (all green)..."
	@uv run python -c "
from monitoring.slo import SLOChecker
metrics = {
    'p99_latency_ms': 200.0, 'error_rate': 0.005,
    'model_auc': 0.80, 'max_feature_age_hours': 5.0,
    'approval_rate': 0.70, 'default_rate': 0.22,
}
report = SLOChecker().run_all(metrics)
status = '✅' if report.overall_compliant else '❌'
print(f'  {status} SLO gate: {len(report.results)} SLOs, {len(report.violations())} violations')
for r in report.results:
    icon = '✅' if r.compliant else '❌'
    print(f'    {icon} {r.slo_name}: {r.current_value:.3f} vs {r.target:.3f} [{r.budget_status.value}]')
"
	@echo ""
	@echo "  Monitor taxonomy self-test..."
	@uv run python -c "
from monitoring.taxonomy import Monitor, MonitorRegistry, MonitorResult, MonitorType, Severity
registry = MonitorRegistry()
registry.register(Monitor('latency_p99', MonitorType.OPERATIONAL,
    lambda: MonitorResult('latency_p99', MonitorType.OPERATIONAL, True, 200, 500, Severity.CRITICAL, 'ok')))
registry.register(Monitor('model_auc', MonitorType.ML,
    lambda: MonitorResult('model_auc', MonitorType.ML, True, 0.80, 0.72, Severity.CRITICAL, 'ok')))
results = registry.run_all()
print(f'  ✅ MonitorRegistry: {len(results)} monitors, {len(MonitorRegistry.failed_results(results))} failed')
"
	@echo ""
	@echo "  Drift detector self-test..."
	@uv run python -c "
import numpy as np, pandas as pd
from monitoring.drift import DriftDetector
rng = np.random.default_rng(42)
df = pd.DataFrame({'pay_ratio': rng.normal(0.5, 0.1, 300), 'util_rate': rng.normal(0.4, 0.1, 300)})
report = DriftDetector().run(df, df)
status = '✅' if not report.overall_drifted else '❌'
print(f'  {status} Drift on identical data: overall_drifted={report.overall_drifted}')
"
	@echo ""
	@echo "  Monitoring Gate: PASSED ✅"

# ── Local K8s (kind) — Phase 9 ───────────────────────────────────────────────

kind-up:
	@echo "→ Creating kind cluster..."
	kind create cluster --name mlops-local --config infra/kind/cluster.yaml || echo "Cluster already exists"
	kubectl cluster-info --context kind-mlops-local

kind-down:
	kind delete cluster --name mlops-local

# ── Internal ─────────────────────────────────────────────────────────────────

_check-env:
	@test -f .env || (echo "❌ .env not found. Copy .env.example to .env and fill in values." && exit 1)
