# Day 2 — 2026 Tooling Landscape & the Golden Path

> Tags: `[T]` theory  
> Deliverable: **Annotated stack map** (below) — your commitment to the golden path

---

## 1. Why Commit to a Stack?

The MLOps tooling space has exploded. Picking tools randomly leads to:
- Integration tax (glue code between incompatible systems)
- Context switching (different abstractions per tool)
- Shallow mastery (knowing 10 tools at 20% depth vs 3 tools at 90% depth)

**This curriculum picks one deep path.** Survey alternatives enough to compare — don't build them.

---

## 2. The Full Stack Map

```mermaid
flowchart TD
    subgraph "Developer Tooling"
        UV["uv (env mgmt)"]
        PC["pre-commit (hooks)"]
        GL["GitLab CI (CI/CD)"]
    end

    subgraph "Data Layer"
        DVC["DVC (versioning)"]
        GE["Great Expectations (validation)"]
        PAN["Pandera (schema)"]
        MINIO["MinIO / S3 (artifact store)"]
        PG["Postgres (metadata)"]
    end

    subgraph "Feature Layer"
        FEAST["Feast (feature store)"]
        REDIS["Redis (online store)"]
    end

    subgraph "Training Layer"
        MFLOW["MLflow (tracking + registry)"]
        OPTUNA["Optuna (HPO)"]
        RAY["Ray Train (distributed)"]
    end

    subgraph "Orchestration"
        DAG["Dagster / Airflow (pipeline)"]
        KFP["KFP or ZenML (ML-native)"]
    end

    subgraph "Serving"
        FAPI["FastAPI (API layer)"]
        BENTO["BentoML (model server)"]
        KSERVE["KServe (K8s serving)"]
    end

    subgraph "Monitoring"
        EV["Evidently (drift)"]
        PROM["Prometheus (metrics)"]
        GRAF["Grafana (dashboards)"]
    end

    subgraph "Infrastructure"
        DOCKER["Docker"]
        KIND["kind / minikube (local K8s)"]
        HELM["Helm (packaging)"]
        ARGOCD["Argo CD (GitOps)"]
        TF["Terraform (IaC)"]
        EKS["AWS EKS (cloud K8s)"]
    end

    subgraph "LLM Layer"
        VLLM["vLLM (serving)"]
        RAGAS["RAGAS (RAG eval)"]
        OTEL["OpenTelemetry (tracing)"]
    end

    subgraph "Agent Layer"
        LG["LangGraph-style FSM"]
        MCP["MCP tools"]
        REPLAY["Trajectory eval + replay"]
    end

    DVC --> MFLOW
    FEAST --> FAPI
    MFLOW --> KSERVE
    DAG --> MFLOW
    FAPI --> BENTO
    BENTO --> KSERVE
    KSERVE --> PROM
    PROM --> GRAF
    EV --> PROM
    VLLM --> KSERVE
    LG --> MCP
    LG --> OTEL
```

---

## 3. Annotated Tool Decisions

### Environment Management: `uv` over `pip/conda/poetry`

| Tool | Pros | Cons | Verdict |
|---|---|---|---|
| `pip` | Universal | No lockfile, slow | Baseline only |
| `conda` | Env + packages | Slow, heavy | Skip |
| `poetry` | Lockfile, groups | Config-heavy | Good, but slow |
| **`uv`** | **10–100x faster than pip, lockfile, pyproject.toml** | Newer, less ecosystem | **Golden path** |

`uv` resolves and installs in seconds. At scale (Docker layer caching, CI) this compounds fast.

---

### Data Versioning: DVC + MinIO

```mermaid
sequenceDiagram
    participant Dev
    participant DVC
    participant Git
    participant MinIO

    Dev->>DVC: dvc add data/raw/loans.parquet
    DVC->>Git: track .dvc pointer file
    DVC->>MinIO: push actual data blob (content-addressed)
    Git->>Git: commit pointer + dvc.lock
    Note over Dev,MinIO: Reproduce any state: git checkout + dvc pull
```

**Why not lakeFS / Iceberg versioning?**
- lakeFS adds operational complexity (another service).
- Iceberg snapshots version table state but don't version raw files.
- DVC is simpler for small-to-medium artifact versioning with full MLflow integration.

Survey: lakeFS (branch-based data versioning), Delta Lake (ACID tables).

---

### Tracking + Registry: MLflow (self-hosted, Postgres + MinIO)

```mermaid
flowchart LR
    TR[Training Script] -->|log params, metrics, artifacts| MLF[MLflow Tracking Server]
    MLF -->|metadata| PG[Postgres]
    MLF -->|artifacts| MINIO[MinIO]
    MLF -->|promote| REG[Model Registry]
    REG -->|alias: champion| SERVE[Serving]
```

**Why not W&B/Neptune?**
- Both are excellent but cloud-hosted (cost, data egress, vendor lock).
- MLflow is the dominant open-source choice; matches SageMaker/Vertex API shape.
- Self-hosted = full control over artifact provenance (required for Security gate).

Survey: W&B (richer UI), Neptune (collaborative), Comet.

---

### Orchestration: Dagster (or Airflow) — one deep build

**Dagster** (primary choice):
- Asset-based model (what you produce, not what you run)
- Native data lineage, software-defined assets
- Better type safety, modern Python-first

**Airflow** (alternative): DAG-based, massive ecosystem, widely deployed.

Pick one and go deep. Survey the rest (Prefect, Metaflow, KFP, ZenML) conceptually.

---

### Serving Path: FastAPI → BentoML → KServe

```mermaid
flowchart LR
    M[MLflow Model] -->|load| B[BentoML Runner]
    B -->|wrap| A[FastAPI endpoint]
    A -->|containerize| D[Docker image]
    D -->|deploy| K[KServe InferenceService]
    K -->|scale| KEDA[KEDA autoscaler]
```

- **FastAPI**: thin wrapper, full control, Pydantic schemas for API contracts.
- **BentoML**: adaptive batching, runner abstraction, production-grade serving.
- **KServe**: K8s-native, scale-to-zero, canary, shadow mode, transformer pipelines.

---

### Cloud: AWS deep, GCP 1:1 mapped

| Concern | AWS | GCP equivalent |
|---|---|---|
| Object store | S3 | GCS |
| Container registry | ECR | Artifact Registry |
| Managed K8s | EKS | GKE |
| ML training | SageMaker Training | Vertex AI Training |
| ML registry | SageMaker Model Registry | Vertex Model Registry |
| ML pipelines | SageMaker Pipelines | Vertex Pipelines (KFP) |
| Model monitoring | SageMaker Model Monitor | Vertex Model Monitoring |
| Serverless inference | SageMaker Serverless | Vertex Endpoints |
| LLM gateway | Bedrock | Vertex AI (Gemini) |
| IaC | Terraform on AWS | Same Terraform, GCP provider |

---

## 4. What We Are Not Building (and Why)

| Tool | Why we skip deep build |
|---|---|
| Seldon Core | Overlaps KServe; similar patterns |
| Triton Inference Server | Deep GPU serving — survey after vLLM |
| Metaflow | Netflix-specific ergonomics; Dagster covers ground |
| Tecton / Featureform | Cloud-managed Feast alternatives; understand Feast first |
| CrewAI / Autogen | Agent frameworks; survey after LangGraph deep build |
| AgentOps SDK / Galileo | Commercial; understand patterns first, then compare |
| llm-d | SOTA disaggregated serving — last chapter, after vLLM mastered |

---

## 5. The Canonical OTel Trace (starts Day 1, built incrementally)

Every phase adds a span to this trace:

```mermaid
sequenceDiagram
    participant Client
    participant Auth
    participant FeatureStore
    participant ModelServer
    participant Retriever
    participant Reranker
    participant ToolCall
    participant Guardrail
    participant Response

    Client->>Auth: request + token [span: auth]
    Auth->>FeatureStore: feature lookup [span: feature_lookup]
    FeatureStore->>ModelServer: feature vector [span: inference]
    ModelServer->>Retriever: context retrieval [span: retriever]
    Retriever->>Reranker: rerank results [span: reranker]
    Reranker->>ToolCall: execute tool [span: tool_call]
    ToolCall->>Guardrail: output check [span: guardrail]
    Guardrail->>Response: final response [span: response]
    Response-->>Client: response + feedback signal [span: feedback]
```

We add one span per phase milestone. By Phase C, the full trace is live.

---

## 6. Deliverable: Your Stack Commitment

Complete this table for yourself:

| Area | My choice | Why |
|---|---|---|
| Env manager | `uv` | Speed, lockfile, pyproject.toml |
| Data versioning | DVC + MinIO | Simple, MLflow integrated |
| Experiment tracking | MLflow (self-hosted) | Open source, full control |
| Pipeline orchestration | Dagster | Asset model, lineage |
| Serving (local) | FastAPI + BentoML | Control → abstraction path |
| Serving (K8s) | KServe | Scale-to-zero, canary built-in |
| Monitoring | Evidently + Prometheus + Grafana | Drift + infra + dashboards |
| Cloud (deep) | AWS | EKS/SageMaker ecosystem |
| LLM serving | vLLM | PagedAttention, active community |
| Agent framework | LangGraph-style | State machine, deterministic |
| Tracing | OpenTelemetry | Vendor-neutral standard |

---

## Key Takeaways

- Depth beats breadth. Master 3 tools at 90% rather than 10 at 20%.
- The stack is chosen for **open-source + self-hosted** first — adds cloud later without lock-in.
- **MLflow + DVC + Feast** are the reproducibility backbone — every phase adds to them.
- AWS is deep; GCP is mapped 1:1 — you'll be able to port with minimal rework.
