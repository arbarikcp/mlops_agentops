# MLOps Platform — One-command local ops

COMPOSE = docker compose -f docker-compose.yml --env-file .env
SERVICES = minio postgres redis mlflow
SEED = 42

.PHONY: help up down clean status smoke-test logs \
        install dvc-init data-download featurize train \
        mlflow-train hpo test test-unit test-data \
        determinism-check reproduce-check \
        calibrate threshold-analysis decision-band slice-eval \
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
