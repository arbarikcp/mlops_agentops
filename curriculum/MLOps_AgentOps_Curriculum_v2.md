# MLOps + AgentOps: A Day-by-Day Engineering Curriculum — **v2 (realigned)**
### From fundamentals to state-of-the-art — theory, local, Kubernetes, and cloud

---

## What changed in v2 (from the expert review)

1. **Milestone-driven, not 150 isolated days.** The plan is now **5 milestones**, each ending in a **hard production gate** (a concrete "you're done when you can answer X" test).
2. **One golden path, depth over breadth.** Each phase marks **Build deeply** vs **Survey conceptually**. Orchestrator tourism is cut (one deep build + one ML-native + the rest surveyed).
3. **Three through-lines run across every phase** instead of being late chapters:
   - **Living Threat Model** — started Day 1, updated each phase via a security checkpoint.
   - **Canonical OTel Trace** — one trace built up incrementally: `auth → feature lookup → inference → retriever → reranker → tool call → guardrail → response → feedback`.
   - **Closed Feedback Loop** — predict → decide → outcome → join label → recompute → trigger → approve → deploy.
4. **New first-class modules added:** ML system design & product framing; calibration / thresholds / cost-sensitive eval; data & label contracts; reliability/chaos lab; expanded RAG production ops; agent tool-safety & permissioning; governance evidence pack.
5. **Cloud = AWS deep, GCP mapped 1:1** (Vertex ↔ SageMaker), to avoid diluting depth. *(Confirm or switch to dual-build.)*
6. **llm-d / disaggregated serving moved to the very end** — only after vLLM single-node → vLLM-on-K8s → KServe/Ray are mastered.
7. **Project charter + progress tracker** added so there's always a clear goal and a record of what's done.

---

## How to use this plan

- **One day at a time, full depth.** Each "Day" ≈ 2–4 focused hours: theory (the *why/how*) then an executable deliverable. "Day" is a unit of work, not a calendar deadline.
- **Tags:** `[T]` theory · `[L]` local · `[K]` Kubernetes · `[C]` cloud · `[S]` security · `[P]` performance · `[M]` monitoring · `[NEW]` added in v2.
- **→** flags where a topic maps onto what you already know (Iceberg, S3, Flink streaming, Postgres, Trino, GitLab CI, K8s).
- **Build deeply** = you write the code/manifests. **Survey** = you read, run a hello-world, and can compare — no deep build.

---

## The backbone project (unchanged — the strongest part)

One evolving platform, three eras:
- **Era A — Classical MLOps:** a **credit-risk** tabular model (regulated + drift-prone → monitoring, security, governance become natural).
- **Era B — LLMOps:** a **policy / RAG assistant** over documents.
- **Era C — AgentOps:** a **support agent** that uses both the risk model and the RAG assistant.

```
platform/
  data/ features/ training/ serving/ pipelines/ infra/
  monitoring/ llm/ agent/ ci/ notebooks/ Makefile
```

---

## Project charter + progress tracker `[NEW]`

Created on Day 5, kept current the whole way. Minimum fields:

| Field | Example |
|---|---|
| Decision the system supports | Approve / review / decline a credit application |
| Primary users / consumers | Underwriting service; human reviewers |
| Cost of FP vs FN | FN (bad loan approved) ≫ FP — drives threshold |
| Latency budget | p95 < 200 ms online; batch nightly |
| Rollback behavior | Auto-revert to previous registry alias on gate failure |
| Late labels | Default/repayment signal arrives 30–90 days later |
| Minimum viable monitoring | Drift on top-10 features + p95 latency + approval rate |
| Current milestone / gate status | M1 in progress; gates ☐☐☐ |

A simple `PROGRESS.md` checklist tracks every day's "definition of done."

---

## The six production gates (the heart of v2) `[NEW]`

| Gate | You must prove |
|---|---|
| **Reproducibility** | From a run ID, reproduce model + data + code + environment. |
| **Serving** | Deploy, roll back, load-test, and explain p95/p99. |
| **Pipeline** | A failed training job retries safely without corrupting artifacts. |
| **Monitoring** | Detect data drift, model-quality decay, infra errors, and bad business outcomes **separately**. |
| **Security** | Show threat model, permissions, secrets, SBOM, model provenance, audit trail. |
| **AgentOps** | Replay a session and prove which tool was called, why, with whose permission, and what it cost. |

Each milestone closes by clearing the relevant gate(s).

---

## Golden-path stack (Build deeply) vs alternatives (Survey)

| Area | Build deeply | Survey conceptually |
|---|---|---|
| Env | `uv` / `pyenv` / Docker | — |
| Data versioning | DVC + MinIO/S3 | lakeFS, Iceberg-as-versioning |
| Tracking + registry | MLflow | W&B, Neptune |
| Data validation | Pandera + Great Expectations | Soda |
| Feature store | Feast | Tecton, Featureform |
| Classical serving | FastAPI → BentoML → KServe | Triton, Seldon |
| Orchestration | **Dagster** (or Airflow) + **one** ML-native (KFP/ZenML) | Prefect, Metaflow, Argo Workflows, SageMaker/Vertex Pipelines |
| K8s serving stack | Helm, KServe, KEDA, Prometheus, Grafana | Karpenter, Kueue (build later) |
| CI/CD | GitLab CI + Argo CD | GitHub Actions, Jenkins |
| Cloud | **AWS (EKS/S3/SageMaker)** | GCP (Vertex) mapped 1:1 |
| LLM serving | vLLM → vLLM-on-K8s → KServe/Ray | llm-d (SOTA, last) |
| RAG eval | RAGAS + failure taxonomy + golden set | DeepEval, TruLens |
| LLM/agent tracing | OpenTelemetry (canonical) | Langfuse, Phoenix, LangSmith (compare) |
| Agent framework | LangGraph-style state machine + MCP tools | CrewAI, Autogen |
| AgentOps | tracing + replay + trajectory eval + policy checks | AgentOps SDK, Galileo (compare) |

---

# ════════ MILESTONE 1 — Classical MLOps Platform ════════
*Runs locally, with CI, basic monitoring, and rollback. Closes the **Reproducibility, Serving, Pipeline, Monitoring** gates.*

## Phase 0 — Orientation & System Design (Days 1–6)
- **Day 1 `[T][S]`** — Why ML rots: hidden technical debt, lifecycle, maturity levels; how AgentOps diverges. **Start the living threat model (v0).**
- **Day 2 `[T]`** — 2026 tooling landscape; commit to the golden path above. *Deliverable:* annotated stack map.
- **Day 3 `[L]`** — Local platform: `uv`, Docker, kind/minikube, MinIO (→ S3), Postgres. *Deliverable:* one-command Makefile up/down.
- **Day 4 `[T][NEW]`** — **ML system / product design:** decision supported, FP-vs-FN cost, real latency need, consumers, rollback behavior, late labels, minimum viable monitoring, **risk matrix**.
- **Day 5 `[L][NEW]`** — **Project charter + `PROGRESS.md` tracker** + backbone repo scaffold, pre-commit, conventions.
- **Day 6 `[L]`** — Dataset selection + EDA + **first data-contract draft** for the raw inputs.

## Phase 1 — Reproducibility, Tracking, Registry (Days 7–14)
- **Day 7 `[T][L]`** — Non-determinism, seeds, lockfiles, hashing. *Deliverable:* deterministic training script.
- **Day 8 `[L][S]`** — DVC + MinIO. → Iceberg snapshots are data versions. **Threat checkpoint: data poisoning + access control.**
- **Day 9 `[L]`** — DVC pipelines (`dvc.yaml`) as a reproducible DAG.
- **Day 10 `[L]`** — MLflow tracking (Postgres + MinIO backends), autolog.
- **Day 11 `[L][S]`** — MLflow Registry: versions/stages/aliases/signatures. **Threat checkpoint: artifact provenance.**
- **Day 12 `[L][P]`** — Optuna sweeps, nested runs, leaderboard.
- **Day 13 `[T]`** — Lineage & metadata (OpenLineage). → your data-lineage experience.
- **Day 14 `[L]`** — *Consolidation* + **Reproducibility gate dry-run.**

## Phase 2 — Decisioning: Calibration, Uncertainty, Thresholds (Days 15–18) `[NEW MODULE]`
*For credit-risk, AUC is not a decision.*
- **Day 15 `[T][L]`** — Probability **calibration** (Platt, isotonic), reliability diagrams.
- **Day 16 `[L]`** — **Threshold tuning** + **cost-sensitive evaluation** (using FP/FN cost from Day 4).
- **Day 17 `[L]`** — Confidence intervals, **reject/abstain option**, human-review routing.
- **Day 18 `[L][M]`** — **Slice-level performance** (region/tenant/income band) + **OOD detection**.

## Phase 3 — Data & Label Contracts (Days 19–21) `[NEW MODULE]`
*Bad labels and unclear contracts break systems more than model code.*
- **Day 19 `[L][S]`** — **Data contracts:** schema, nullability, freshness, ownership, semantic meaning, enforcement. → your data-platform strength.
- **Day 20 `[L]`** — **Label contracts + ground-truth pipelines:** provenance, arrival timing, correction, backfill.
- **Day 21 `[L][M]`** — **Train/serve skew detection** + dataset slicing strategy.

## Phase 4 — Packaging & Serving (Days 22–30)
- **Day 22 `[T][L][S]`** — Serialization: ONNX, safetensors, **pickle risk**. *Deliverable:* ONNX export.
- **Day 23 `[T]`** — Inference patterns: online vs batch vs streaming (→ streaming), latency budgets.
- **Day 24 `[L]`** — FastAPI: Pydantic schemas, health/readiness, versioned endpoints.
- **Day 25 `[L][S]`** — Containerize: multi-stage, non-root, image scanning. **Threat checkpoint: serving surface.**
- **Day 26 `[L]`** — BentoML: bentos, runners, adaptive batching.
- **Day 27 `[L]`** — Batch inference: idempotency, backfills (→ batch pipelines).
- **Day 28 `[L][NEW]`** — **Model API contract:** request/response schema, backward/forward compatibility, **versioned rollback plan**.
- **Day 29 `[L][P]`** — Load testing (k6/Locust), p50/p95/p99, profiling.
- **Day 30 `[L][S]`** — Serving security (authn/authz, mTLS, rate limits, secrets) + *consolidation* + **Serving gate dry-run.**

## Phase 5 — Orchestration & Pipelines (Days 31–37) `[SLIMMED]`
- **Day 31 `[T]`** — Orchestration principles: DAGs, assets, idempotency, retries, backfills, lineage, conditional promotion. → go fast.
- **Day 32 `[L]`** — **Build deeply:** Dagster (or Airflow) training pipeline.
- **Day 33 `[L]`** — **One ML-native path** (KFP or ZenML) — light build, principles-focused.
- **Day 34 `[L]`** — Data-validation **gate** (Pandera + GE) wired into the pipeline.
- **Day 35 `[L]`** — Model-validation **gate**: thresholds, champion/challenger, auto-promote.
- **Day 36 `[L][NEW]`** — **Pipeline failure modes:** idempotency proof, retry-safety, lineage audit.
- **Day 37 `[T][L]`** — **Survey:** Prefect, Metaflow, Argo, SageMaker/Vertex Pipelines (conceptual) + *consolidation* + **Pipeline gate dry-run.**

## Phase 6 — Feature Store & Closed Feedback Loop (Days 38–45)
- **Day 38 `[T]`** — Feature-store problem: skew, point-in-time correctness, reuse.
- **Day 39 `[T][L]`** — Feast architecture (offline = Parquet/Iceberg/MinIO).
- **Day 40 `[L]`** — Feature views, entities, point-in-time joins.
- **Day 41 `[L]`** — Materialization + online store (Redis).
- **Day 42 `[L]`** — Streaming features (push sources, on-demand transforms). → Flink.
- **Day 43 `[M]`** — Feature monitoring: drift, freshness SLAs, data quality.
- **Day 44 `[T][L][NEW]`** — **Label feedback loop + delayed ground truth + active-learning basics. Start the closed feedback loop.**
- **Day 45 `[L]`** — *Consolidation:* zero train/serve skew.

## Phase 7 — Monitoring & the Closed Loop (Days 46–53)
- **Day 46 `[T][M]`** — Monitoring taxonomy: operational vs ML vs business (kept **separate** — gate requirement).
- **Day 47 `[T]`** — Drift & concept drift: PSI, KS, MMD, classifier-based.
- **Day 48 `[L][M]`** — Evidently: reports + test suites, in-pipeline and as a service.
- **Day 49 `[L][M]`** — Prometheus custom ML metrics + PromQL.
- **Day 50 `[L][M]`** — Grafana: golden signals + ML panels + alerts.
- **Day 51 `[L][S]`** — Prediction logging for audit/replay; correlation IDs.
- **Day 52 `[L][M][NEW]`** — **Closed-loop learning system (8 steps):** predict → decide → outcome → join label → recompute → trigger → approve → deploy.
- **Day 53 `[T]`** — SLOs/SLIs/error budgets + incident response intro + *consolidation* + **Monitoring gate dry-run.**

## Phase 8 — CI/CD for ML (Days 54–58)
- **Day 54 `[T]`** — CI/CD-for-ML: code + data + model; ML testing pyramid.
- **Day 55 `[L]`** — Testing ML: unit (transforms), data, **behavioral**, training smoke tests.
- **Day 56 `[L]`** — GitLab CI pipelines (→ your Zscaler-hosted GitLab): stages, runners, caching, artifacts.
- **Day 57 `[L][S][NEW]`** — Automated image+model build; basic CD with approvals + rollback + **release engineering: artifact signing (Sigstore) + SBOM intro.**
- **Day 58 `[L]`** — *Consolidation* + **▶ MILESTONE 1 GATE.**

> **M1 Gate — you pass when:** *given a prediction, you can trace the model version, data version, code version, feature values, request ID, and decision outcome* — and you can roll back, retry a failed job safely, and detect drift/quality/infra/business issues separately. Threat model at **v1**.

---

# ════════ MILESTONE 2 — Kubernetes & Cloud ML Platform ════════
*Move the same system to K8s and AWS (GCP mapped). Adds chaos testing. Re-clears Serving/Pipeline/Monitoring at production scale.*

## Phase 9 — Kubernetes for ML (Days 59–70)
- **Day 59 `[T][K]`** — K8s for ML: pods, deployments, services, config/secrets, resources/limits.
- **Day 60 `[K]`** — kind cluster: deploy service + ingress.
- **Day 61 `[K]`** — Helm chart the service.
- **Day 62 `[K][P]`** — Storage: PVCs, model-storage strategies, init-container pulls, the "storage drag" problem.
- **Day 63 `[T][K]`** — GPU on K8s: NVIDIA GPU Operator, device plugin, selectors/taints.
- **Day 64 `[K]`** — KServe InferenceService: predictor/transformer, scale-to-zero.
- **Day 65 `[K]`** — KServe canary, traffic splitting, shadow/mirror.
- **Day 66 `[K]`** — Ray on K8s (KubeRay) + Ray Serve multi-model pipelines.
- **Day 67 `[K][P]`** — Autoscaling: HPA, **KEDA (queue-depth)**, node autoscaling (Karpenter concept).
- **Day 68 `[K]`** — Kueue GPU scheduling: job queueing, fair sharing.
- **Day 69 `[K][M][S]`** — Prometheus/Grafana on K8s + secrets/RBAC. **Threat checkpoint: secrets at scale.**
- **Day 70 `[T][K]`** — **Survey:** Kubeflow (Pipelines/Katib/Training Operator) + *consolidation*.

## Phase 10 — Reliability Lab: Failure Injection (Days 71–73) `[NEW MODULE]`
*Separates "I deployed a model" from "I can operate a platform."*
- **Day 71 `[K]`** — Chaos fundamentals + **infra failure injection:** MLflow down, MinIO/S3 down, registry down, KServe stuck, GPU node gone, queue backlog.
- **Day 72 `[K][M]`** — **ML-specific incident drills:** bad artifact pushed, stale features, broken retriever. For each: expected behavior, actual, alert, recovery path, prevention.
- **Day 73 `[T]`** — Game-day + runbooks + postmortems.

## Phase 11 — GitOps & Continuous Training (Days 74–77)
- **Day 74 `[K]`** — GitOps: Argo CD / Flux — declarative deploys, sync, rollback.
- **Day 75 `[K]`** — Progressive delivery for models (canary / blue-green) on K8s.
- **Day 76 `[K]`** — CT automation: retrain → registry → deploy; Argo Workflows / Events.
- **Day 77 `[K]`** — *Consolidation*.

## Phase 12 — Cloud MLOps: AWS deep, GCP mapped (Days 78–90)
- **Day 78 `[T][C][S]`** — Cloud landscape + cost model + "managed-for-dev, K8s-for-prod"; **IAM-first**.
- **Day 79 `[C][S]`** — AWS foundations: S3 (→ known), IAM roles, ECR, VPC for ML.
- **Day 80 `[C]`** — SageMaker: training/processing jobs, Experiments.
- **Day 81 `[C]`** — SageMaker registry + endpoints (real-time/serverless/async/batch).
- **Day 82 `[C]`** — SageMaker Pipelines + model approval + lineage.
- **Day 83 `[C][M]`** — SageMaker Model Monitor + Clarify (bias/explainability).
- **Day 84 `[C]`** — AWS serving on EKS + Bedrock overview.
- **Day 85 `[C][S][P]`** — AWS cost & security: spot, KMS, PrivateLink, budget guardrails.
- **Day 86 `[C][L]`** — Terraform for ML infra (provision on AWS).
- **Day 87 `[C]`** — **GCP mapping (1:1, minimal build):** Vertex training/registry/endpoints/Pipelines(KFP)/Model Monitoring/Explainable AI ↔ their SageMaker equivalents.
- **Day 88 `[T]`** — Portability: core on K8s/MLflow/Feast; cloud as substrate.
- **Day 89 `[C]`** — Deploy full backbone end-to-end on AWS.
- **Day 90 `[C]`** — *Consolidation* + **▶ MILESTONE 2 GATE.**

> **M2 Gate — you pass when:** the platform runs on K8s + AWS, fully IaC'd, with autoscaling, canary, proven rollback, cost controls, and you've **recovered from at least 5 injected failures** with documented runbooks. Threat model at **v2**.

---

# ════════ MILESTONE 3 — Production RAG / LLMOps ════════
*Vehicle: your RAG capstone + DeskMate. Closes a RAG-provenance gate.*

## Phase 13 — Scaling & Inference Optimization (Days 91–99)
- **Day 91 `[T]`** — Distributed training: data/model/pipeline/tensor parallelism; DDP, FSDP, ZeRO.
- **Day 92 `[K][P]`** — Ray Train multi-GPU job.
- **Day 93 `[P]`** — Training optimization: mixed precision, gradient checkpointing/accumulation, data loading.
- **Day 94 `[T][P]`** — Inference optimization: batching, KV cache, **PagedAttention**, continuous batching.
- **Day 95 `[P]`** — Quantization for serving: PTQ/QAT, GPTQ/AWQ, distillation. → your quant curriculum.
- **Day 96 `[P]`** — Compilation/runtimes: ONNX Runtime, TensorRT-LLM, `torch.compile`.
- **Day 97 `[P][C]`** — GPU utilization & cost: MIG, fractional GPUs, spot, killing idle GPUs.
- **Day 98 `[L][P]`** — **vLLM single-node deep** (the baseline before anything distributed).
- **Day 99 `[K][P]`** — **vLLM on K8s** + GPU metrics/batching + capacity planning.

## Phase 14 — LLMOps Core (Days 100–108)
- **Day 100 `[T]`** — LLMOps vs MLOps: prompts-as-artifacts, non-determinism, cost-as-metric.
- **Day 101 `[K]`** — Serving LLMs: KServe LLMInferenceService / Ray Serve patterns *(not llm-d yet)*.
- **Day 102 `[L]`** — Prompt management & versioning: prompts-as-code, registries, A/B.
- **Day 103 `[T][L]`** — LLM eval I: offline, reference-based/free, **LLM-as-judge**, eval datasets.
- **Day 104 `[L][M]`** — LLM eval II: **RAGAS** — faithfulness, context relevance, answer correctness. → RAGBench.
- **Day 105 `[L][P]`** — Fine-tuning ops: LoRA/QLoRA, dataset versioning, **eval-gated** promotion. → DeskMate.
- **Day 106 `[M]`** — LLM observability: **OTel GenAI — extend the canonical trace**; compare Langfuse/Phoenix/LangSmith.
- **Day 107 `[M]`** — LLM monitoring in prod: quality/hallucination drift, online eval on sampled traffic; full-traffic eval economics.
- **Day 108 `[P][C][NEW]`** — **LLM gateway architecture:** model routing (cheap→expensive), quota enforcement, semantic caching, cost governance.

## Phase 15 — RAG Production Operations (Days 109–114) `[EXPANDED MODULE]`
- **Day 109 `[L]`** — **Index build pipeline** + **versioning + rollback**.
- **Day 110 `[L]`** — Chunking experiments + **hybrid retrieval (BM25 + vector)** + reranking. → your FAISS/BM25 indices.
- **Day 111 `[L][S]`** — Metadata filtering + **multi-tenant retrieval security** + **document ACL propagation**. **Threat: index poisoning, data exfiltration.**
- **Day 112 `[L]`** — Stale-document removal + **embedding-model migration** + RAG cache invalidation.
- **Day 113 `[L][M]`** — **Retrieval failure taxonomy** + **golden query set** + synthetic query generation.
- **Day 114 `[L][S]`** — Eval by document slice/source/type + RAG guardrails (prompt injection, source trust, OWASP LLM Top 10, Llama Guard) + **▶ MILESTONE 3 GATE.**

> **M3 Gate — you pass when:** for any answer you can prove *"this came from these retrieved chunks, using this prompt version, this embedding model, this index version, this LLM version, and this eval score"* — with guardrails active and cost tracked. Threat model at **v3**.

---

# ════════ MILESTONE 4 — Production AgentOps ════════
*Extend the RAG assistant into a tool-using agent. Closes the AgentOps gate. Security is first-class here.*

## Phase 16 — AgentOps Core (Days 115–122)
- **Day 115 `[T][S]`** — Why AgentOps is distinct (state, tools, emergent goals, non-HTTP failures); agent lifecycle. **Start agent threat model.**
- **Day 116 `[T][M]`** — Observability fundamentals: span taxonomy (reasoning/planning/workflow/task/tool/LLM), trace-per-tick, **OTel GenAI — complete the canonical trace** (…→tool→guardrail→response→feedback).
- **Day 117 `[L][M]`** — Instrumenting agents: AgentOps SDK / LangSmith / Phoenix; **session replay & time-travel debugging**.
- **Day 118 `[T][L]`** — Agent eval I: **trajectory eval**, tool-use correctness, task success, step efficiency.
- **Day 119 `[L]`** — Agent eval II: LLM-as-judge for agents, **composite metrics & gatekeeping**, Agent Evals via MCP.
- **Day 120 `[L]`** — Agent testing: simulation environments, scenario/replay, regression suites.
- **Day 121 `[L]`** — Agent reliability: retries, fallbacks, circuit breakers, timeouts, **runaway-loop detection**.
- **Day 122 `[L]`** — Agent memory & state ops: short/long-term, vector memory, persistence & recovery.

## Phase 17 — Agent Security & Tool Safety (Days 123–127) `[NEW FIRST-CLASS MODULE]`
*Where risk explodes — the agent can take actions, not just answer.*
- **Day 123 `[S]`** — **Tool permission model** + user identity propagation + per-tool scopes.
- **Day 124 `[S]`** — Tool approval policies + **high-risk action classification** + **dry-run mode**.
- **Day 125 `[S]`** — Tool-call budget + tool-timeout budget + **sandbox execution** + tool-result validation.
- **Day 126 `[S]`** — **MCP server trust levels** + **audit-log schema** + **kill switch**.
- **Day 127 `[S][M]`** — **Agent failure injection:** tool timeout, infinite loop, guardrail-service down, observability collector down.

## Phase 18 — Agent Deployment & Multi-Agent (Days 128–130)
- **Day 128 `[L][M]`** — Multi-agent ops: orchestration, message tracing, hierarchical debugging.
- **Day 129 `[K]`** — Deployment patterns: long-running, async/queue-based, **human-in-the-loop, approval gates**.
- **Day 130 `[L]`** — *Consolidation* + **▶ MILESTONE 4 GATE.**

> **M4 Gate — you pass when:** you can replay an agent session and explain every **tool call, failure, retry, permission, cost, and output** — with a working kill switch and audit trail. Threat model at **v4**.

---

# ════════ MILESTONE 5 — Governance, Capstone & SOTA ════════

## Phase 19 — Security, Governance & Responsible AI (Days 131–138)
*Mostly consolidation now — the threat model has been living since Day 1.*
- **Day 131 `[T][S]`** — MLSecOps: consolidate the lifecycle threat model.
- **Day 132 `[S]`** — Supply chain: provenance, SBOM, signing, poisoning, deserialization — deepen.
- **Day 133 `[S]`** — Adversarial & privacy attacks (evasion, membership inference, inversion/extraction) + defenses.
- **Day 134 `[S]`** — Privacy-preserving ML: PII handling, differential privacy basics, federated inference (2026 trend).
- **Day 135 `[S][K][C]`** — Access control & secrets at scale: RBAC, least privilege, rotation, KMS/CMEK.
- **Day 136 `[T]`** — Model governance: model cards, registry-as-governance, approval workflows, audit logs.
- **Day 137 `[T][M]`** — Regulatory (EU AI Act, NIST AI RMF, finance/health) + bias/fairness/explainability ops (Clarify/SHAP gates).
- **Day 138 `[S][NEW]`** — **Governance evidence pack:** model card + data card + eval card + **risk register**.

## Phase 20 — Capstone & State-of-the-Art (Days 139–148)
- **Days 139–144 `[L][K][C]`** — **Capstone:** unify all three eras — fully CI/CD/CT, observable, secured, on K8s + AWS, Terraform-managed, **all six gates green**. Milestones defined when we arrive.
- **Day 145 `[P]`** — SOTA serving: **llm-d / disaggregated inference / prefix-cache-aware routing** (now that vLLM baseline is mastered).
- **Day 146 `[M]`** — SOTA eval/monitoring: full-traffic online eval economics, self-improving eval loops.
- **Day 147 `[T]`** — Frontier: federated/edge inference, agentic infrastructure research; how to read arXiv/conferences to stay current.
- **Day 148 `[T]`** — Retrospective + portfolio + your reusable **golden-path platform template**.

---

## Daily rhythm
1. Read the theory (the *why*). 2. Build the deliverable (the *how*). 3. Wire it into the backbone. 4. Update `PROGRESS.md` + the threat model + the canonical trace if touched. 5. One-line journal entry.

## What I produce per day, when we get there
Theory write-up + diagrams · executable notebook(s)/scripts · Helm/K8s/Terraform where `[K]`/`[C]` · a clear definition of done. Tooling versions verified at the start of each module (the 2026 LLMOps/AgentOps stack moves fast).

*Cloud is currently scoped **AWS-deep, GCP-mapped**. Say the word to switch to dual-build.*
