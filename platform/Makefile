# MLOps Platform — One-command local ops
# Day 3 deliverable: make up / make down

COMPOSE = docker compose -f docker-compose.yml --env-file .env
SERVICES = minio postgres redis mlflow

.PHONY: help up down clean status smoke-test logs kind-up kind-down

help:
	@echo ""
	@echo "  make up          Start all local services"
	@echo "  make down        Stop all services (keep volumes)"
	@echo "  make clean       Stop + delete all volumes (DESTRUCTIVE)"
	@echo "  make status      Show service health"
	@echo "  make smoke-test  Verify all services are reachable"
	@echo "  make logs        Tail logs from all services"
	@echo "  make kind-up     Create local kind K8s cluster"
	@echo "  make kind-down   Delete kind cluster"
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
