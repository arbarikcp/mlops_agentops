# MLOps Platform — One-command local ops

COMPOSE = docker compose -f docker-compose.yml --env-file .env
SERVICES = minio postgres redis mlflow
SEED = 42

.PHONY: help up down clean status smoke-test logs \
        install dvc-init data-download featurize train \
        mlflow-train hpo test test-unit test-data \
        determinism-check reproduce-check \
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
