# Day 90 — Consolidation + Milestone 2 Gate

## WHY

Phase 12 has covered the full cloud deployment spectrum: AWS cost and security
controls (Day 85), Terraform for reproducible infra (Day 86), GCP mapping
(Day 87), platform portability (Day 88), and an end-to-end AWS deployment
(Day 89). Milestone 2 is the formal **production-readiness gate** that
certifies the platform before moving into LLMOps (Phase 13+).

Without a structured gate, teams ship "mostly done" platforms that pass happy-
path demos but fail in production on the exact scenarios the gate is designed
to catch: untraceable model lineage, insecure endpoints, undetected drift.

> **Milestone 2 answers one question:** _Can this platform run in production
> without constant human intervention?_

The six gate dimensions are identical to the Six Production Gates defined in
the curriculum — each is now checked with concrete, automated assertions.

---

## HOW

### The Six Gate Dimensions

| Gate | What it checks |
|---|---|
| **Reproducibility** | Trace endpoint → model → training job → data version → code commit |
| **Serving** | Endpoint health, latency SLO (p95 < 200 ms), rollback path |
| **Pipeline** | SM Pipeline or Argo Workflow completed successfully, retries logged |
| **Monitoring** | Drift monitor active, bias monitor active, alert routing wired |
| **Security** | KMS encryption on artifacts, IAM least-privilege, PrivateLink active |
| **Portability** | Core layer (MLflow, Feast, DVC, KServe) portable, score ≥ 80 % |

### M2GateCheck (one assertion)

Each individual check is a `M2GateCheck`:

```python
@dataclass
class M2GateCheck:
    gate: str           # "reproducibility" | "serving" | "pipeline" | ...
    check_id: str       # e.g. "REPR-01"
    description: str    # human-readable
    passed: bool
    evidence: str       # e.g. ARN, URL, commit SHA
    severity: str       # "critical" | "warning"
```

### Milestone2Gate

Runs all 12+ checks and aggregates results:

```python
gate = Milestone2Gate(
    endpoint_name="credit-risk-prod",
    pipeline_execution_arn="arn:aws:sagemaker:...",
    model_package_arn="arn:aws:sagemaker:...",
    kms_key_arn="arn:aws:kms:...",
    vpc_endpoint_ids=["vpce-...", "vpce-..."],
    portability_score=82.0,
    drift_monitor_schedule="credit-risk-monitor",
    bias_monitor_schedule="credit-risk-bias",
    argocd_app="credit-risk",
)
report = gate.run()
```

`run()` executes each check in order, populates `M2GateCheck.passed`, and
returns a `M2GateReport`.

### M2GateReport

```python
@dataclass
class M2GateReport:
    run_id: str
    timestamp: datetime
    checks: List[M2GateCheck]
    gates_passed: List[str]
    gates_failed: List[str]
    overall_passed: bool         # True only if ALL critical checks pass
    critical_failures: List[str]
    warnings: List[str]
    to_dict() -> dict
    print_summary() -> None
```

`overall_passed` is `True` only when every `severity="critical"` check
passes. Warnings are reported but do not block the gate.

---

### The 12 Gate Checks

| Check ID | Gate | Description | Severity |
|---|---|---|---|
| REPR-01 | Reproducibility | Endpoint has a model ARN tag | critical |
| REPR-02 | Reproducibility | Model ARN links to a training job with `dvc_data_version` tag | critical |
| REPR-03 | Reproducibility | Training job has `git_commit_sha` tag | critical |
| SERV-01 | Serving | Endpoint status is `InService` | critical |
| SERV-02 | Serving | p95 latency < 200 ms (last 1 h CloudWatch) | critical |
| SERV-03 | Serving | Previous endpoint config exists (rollback target) | warning |
| PIPE-01 | Pipeline | Latest pipeline execution status is `Succeeded` | critical |
| PIPE-02 | Pipeline | At least one retry step present in execution graph | warning |
| MON-01 | Monitoring | Data quality monitor schedule is `Scheduled` | critical |
| MON-02 | Monitoring | Bias monitor schedule is `Scheduled` | critical |
| MON-03 | Monitoring | CloudWatch alert on drift metric routes to SNS | warning |
| SEC-01 | Security | Model S3 output is SSE-KMS encrypted | critical |
| SEC-02 | Security | SageMaker execution role has no `*` actions | critical |
| SEC-03 | Security | S3 and ECR VPC endpoints are active | critical |
| PORT-01 | Portability | PortabilityScore >= 80 % | warning |
| PORT-02 | Portability | MLflow, Feast, DVC accessible without cloud SDK | warning |

> The table above lists 16 checks across 6 gates — implementations may
> expand or contract based on environment. The minimum is 12 passing critical
> checks for gate approval.

---

## Class Diagram

```mermaid
classDiagram
    class M2GateCheck {
        +str gate
        +str check_id
        +str description
        +bool passed
        +str evidence
        +str severity
        +is_critical() bool
        +to_dict() dict
    }

    class Milestone2Gate {
        +str endpoint_name
        +str pipeline_execution_arn
        +str model_package_arn
        +str kms_key_arn
        +List~str~ vpc_endpoint_ids
        +float portability_score
        +str drift_monitor_schedule
        +str bias_monitor_schedule
        +str argocd_app
        +List~M2GateCheck~ checks
        +run() M2GateReport
        +_check_reproducibility() List~M2GateCheck~
        +_check_serving() List~M2GateCheck~
        +_check_pipeline() List~M2GateCheck~
        +_check_monitoring() List~M2GateCheck~
        +_check_security() List~M2GateCheck~
        +_check_portability() List~M2GateCheck~
    }

    class M2GateReport {
        +str run_id
        +datetime timestamp
        +List~M2GateCheck~ checks
        +List~str~ gates_passed
        +List~str~ gates_failed
        +bool overall_passed
        +List~str~ critical_failures
        +List~str~ warnings
        +pass_rate() float
        +to_dict() dict
        +print_summary() None
    }

    Milestone2Gate "1" *-- "many" M2GateCheck
    Milestone2Gate ..> M2GateReport : produces
    M2GateReport "1" *-- "many" M2GateCheck
```

---

## Sequence: Gate Execution

```mermaid
sequenceDiagram
    participant Eng as Engineer / CI
    participant MG as Milestone2Gate
    participant SM as SageMaker APIs
    participant CW as CloudWatch
    participant ARGO as Argo CD
    participant MGR as M2GateReport

    Eng->>MG: run()
    MG->>SM: DescribeEndpoint (SERV-01)
    SM-->>MG: status=InService
    MG->>CW: GetMetricStatistics p95 latency (SERV-02)
    CW-->>MG: p95=145ms → passed
    MG->>SM: DescribeModelPackage tags (REPR-01, REPR-02)
    SM-->>MG: tags: {dvc_data_version, git_commit_sha}
    MG->>SM: DescribeTrainingJob (REPR-03)
    SM-->>MG: git_commit_sha=abc123 → passed
    MG->>SM: DescribePipelineExecution (PIPE-01)
    SM-->>MG: status=Succeeded
    MG->>SM: DescribeMonitoringSchedule drift (MON-01)
    SM-->>MG: status=Scheduled
    MG->>SM: DescribeMonitoringSchedule bias (MON-02)
    SM-->>MG: status=Scheduled
    MG->>SM: GetBucketEncryption (SEC-01)
    SM-->>MG: SSE-KMS → passed
    MG->>SM: SimulatePolicy IAM (SEC-02)
    SM-->>MG: no wildcard actions → passed
    MG->>ARGO: argocd app get credit-risk (PIPE-02)
    ARGO-->>MG: sync=Synced, health=Healthy
    MG->>MGR: build report
    MGR-->>Eng: overall_passed=True, 14/16 checks passed
```

---

## Flowchart: Gate Decision Tree

```mermaid
flowchart TD
    START[Milestone2Gate.run] --> R[Reproducibility checks\nREPR-01 REPR-02 REPR-03]
    R --> S[Serving checks\nSERV-01 SERV-02 SERV-03]
    S --> P[Pipeline checks\nPIPE-01 PIPE-02]
    P --> M[Monitoring checks\nMON-01 MON-02 MON-03]
    M --> SEC[Security checks\nSEC-01 SEC-02 SEC-03]
    SEC --> PORT[Portability checks\nPORT-01 PORT-02]
    PORT --> AGG{Any critical\nfailures?}
    AGG -->|Yes| FAIL[overall_passed = False\nblock Phase 13 start\npage on-call]
    AGG -->|No| WARN{Any warnings?}
    WARN -->|Yes| PASS_W[overall_passed = True\nlog warnings\ncreate backlog tickets]
    WARN -->|No| PASS[overall_passed = True\nPhase 13 unlocked]
```

---

## Phase 12 Consolidation Summary

| Day | Topic | Key Output |
|---|---|---|
| 85 | AWS Cost & Security | `AWSSecurityConfig` — Spot + KMS + PrivateLink + Budget |
| 86 | Terraform for ML | `TFConfig` — programmatic `.tf` generation |
| 87 | GCP Mapping | `VertexMLPlatform` — 1:1 AWS ↔ GCP lifecycle |
| 88 | Portability | `PortabilityMatrix` + `CloudAdapter` pattern |
| 89 | E2E AWS Deployment | `AWSDeploymentPlan` — 8-stage orchestrator |
| 90 | Milestone 2 Gate | `Milestone2Gate` — 16 checks across 6 gates |

---

## Key Takeaways

1. **The Milestone 2 Gate is a runnable program**, not a checklist in a
   doc. Every check either passes or fails with machine-readable evidence
   — no human judgement required for the critical path.
2. **Critical vs. warning** distinction prevents gate theatre: warnings
   are tracked but do not block the programme; only critical failures stop
   Phase 13 from starting.
3. **Reproducibility is the hardest gate** to pass: it requires tags to
   be written at training time (git SHA, DVC version) and propagated through
   the registry to the endpoint. Retrofitting this is painful.
4. **Security checks use AWS APIs directly** (`SimulatePolicy`, `GetBucketEncryption`,
   `DescribeVpcEndpoints`) — no human inspection of policies needed.
5. **Portability score >= 80 %** is a warning, not a critical check, because
   some cloud-native coupling is acceptable; the goal is awareness, not
   zero lock-in.
6. Passing Milestone 2 means the classical MLOps platform is production-ready
   and the team can move into **LLMOps (Phase 13)** with confidence that the
   foundation is solid.
