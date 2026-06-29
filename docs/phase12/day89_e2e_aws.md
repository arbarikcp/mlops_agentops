# Day 89 — End-to-End AWS Deployment

## WHY

Individual AWS services are well-understood in isolation — but the production
failure modes almost always appear **at the seams**: the DVC pointer that
references an S3 key that no longer exists, the ECR image that was built
without the KMS key the SageMaker execution role expects, the Argo CD sync
that succeeds but the KServe InferenceService never becomes Ready.

Day 89 wires the entire backbone together on AWS from raw data to a live,
monitored endpoint — and names the failure points at every seam.

> **Goal:** one `AWSDeploymentPlan.execute()` call orchestrates S3 → ECR →
> SageMaker Train → SageMaker Registry → SageMaker Endpoint (or EKS/KServe)
> → Argo CD sync → CloudWatch monitoring.

---

## HOW

### Full AWS Backbone Stack

```
Data Layer          DVC + S3 (versioned, KMS-encrypted)
│
├── Feature Store   Feast offline store → S3 / online → DynamoDB or Redis
│
├── Training        SageMaker Training Job (Spot, VPC, KMS)
│       └──────────► ECR (private, PrivateLink)
│
├── Registry        SageMaker Model Registry (model package group)
│
├── Serving         Option A: SageMaker Endpoint (managed)
│                   Option B: EKS + KServe InferenceService (portable)
│
├── GitOps / CD     Argo CD watches config repo → syncs K8s manifests
│
└── Monitoring      CloudWatch Metrics + SageMaker Model Monitor
                    + Budget alarms (Day 85)
```

### DeploymentStage

Each logical step in the deployment is a `DeploymentStage`:

```python
@dataclass
class DeploymentStage:
    name: str              # e.g. "train", "register", "deploy"
    status: str            # "pending" | "running" | "succeeded" | "failed"
    started_at: datetime | None
    finished_at: datetime | None
    outputs: dict          # stage-specific outputs (job_name, model_arn, …)
    error: str | None
```

### AWSDeploymentPlan

Orchestrates all stages in order. Each stage calls the appropriate AWS SDK
method and writes outputs consumed by the next stage.

```python
plan = AWSDeploymentPlan(
    project="credit-risk",
    region="us-east-1",
    s3_bucket="ml-artifacts",
    ecr_repo="credit-risk-trainer",
    sm_role_arn="arn:aws:iam::123456789:role/SageMakerRole",
    kms_key_arn="arn:aws:kms:us-east-1:123456789:key/abc",
    eks_cluster="ml-cluster",
    argocd_app="credit-risk",
)
report = plan.execute()
```

Stages (in order):

| # | Stage | Key action | Outputs |
|---|---|---|---|
| 1 | `dvc_pull` | `dvc pull` from S3 | `data_version` |
| 2 | `build_push` | `docker build` + `ecr push` | `image_uri` |
| 3 | `train` | SageMaker CreateTrainingJob (Spot) | `job_name`, `model_s3_uri` |
| 4 | `register` | SageMaker CreateModelPackage | `model_package_arn` |
| 5 | `deploy_sm` | SageMaker CreateEndpoint or UpdateEndpoint | `endpoint_name` |
| 6 | `deploy_eks` | `kubectl apply` KServe YAML (optional) | `inference_service_url` |
| 7 | `argocd_sync` | `argocd app sync` | `sync_status` |
| 8 | `monitor` | Enable SageMaker Model Monitor schedule | `monitor_schedule_name` |

### DeploymentReport

`AWSDeploymentPlan.execute()` returns a `DeploymentReport`:

```python
@dataclass
class DeploymentReport:
    plan_name: str
    started_at: datetime
    finished_at: datetime
    stages: List[DeploymentStage]
    succeeded: bool
    summary: dict          # high-level KPIs
    failure_stage: str | None
```

`summary` includes:
- `total_duration_minutes`
- `training_cost_usd` (from CloudWatch billing metric)
- `endpoint_url`
- `model_package_arn`
- `spot_savings_pct`

---

## Class Diagram

```mermaid
classDiagram
    class DeploymentStage {
        +str name
        +str status
        +datetime started_at
        +datetime finished_at
        +dict outputs
        +str error
        +duration_seconds() float
        +is_complete() bool
        +mark_running() None
        +mark_succeeded(outputs) None
        +mark_failed(error) None
    }

    class AWSDeploymentPlan {
        +str project
        +str region
        +str s3_bucket
        +str ecr_repo
        +str sm_role_arn
        +str kms_key_arn
        +str eks_cluster
        +str argocd_app
        +List~DeploymentStage~ stages
        +execute() DeploymentReport
        +_run_dvc_pull() dict
        +_build_and_push() dict
        +_run_training() dict
        +_register_model() dict
        +_deploy_sagemaker() dict
        +_deploy_eks() dict
        +_sync_argocd() dict
        +_enable_monitoring() dict
        +rollback(stage_name: str) None
    }

    class DeploymentReport {
        +str plan_name
        +datetime started_at
        +datetime finished_at
        +List~DeploymentStage~ stages
        +bool succeeded
        +dict summary
        +str failure_stage
        +to_dict() dict
        +print_summary() None
        +failed_stages() List~DeploymentStage~
    }

    AWSDeploymentPlan "1" *-- "many" DeploymentStage
    AWSDeploymentPlan ..> DeploymentReport : produces
    DeploymentReport "1" *-- "many" DeploymentStage
```

---

## Sequence: Full E2E AWS Deployment

```mermaid
sequenceDiagram
    participant Plan as AWSDeploymentPlan
    participant DVC as DVC + S3
    participant ECR as Amazon ECR
    participant SM as SageMaker
    participant REG as SM Model Registry
    participant EKS as EKS + KServe
    participant ARGO as Argo CD
    participant CW as CloudWatch

    Plan->>DVC: dvc pull (versioned data)
    DVC-->>Plan: data ready at /opt/ml/input/

    Plan->>ECR: docker build + push trainer image
    ECR-->>Plan: image_uri

    Plan->>SM: CreateTrainingJob (Spot, KMS, VPC)
    SM-->>SM: pull image from ECR via PrivateLink
    SM-->>SM: read data from S3 via VPC Gateway endpoint
    SM-->>SM: write model.tar.gz to S3 (SSE-KMS)
    SM-->>Plan: job_name, model_s3_uri

    Plan->>REG: CreateModelPackage
    REG-->>Plan: model_package_arn

    Plan->>SM: CreateEndpoint (or UpdateEndpoint)
    SM-->>Plan: endpoint_name, endpoint_url

    Plan->>EKS: kubectl apply KServe YAML
    EKS-->>Plan: InferenceService Ready

    Plan->>ARGO: argocd app sync credit-risk
    ARGO-->>Plan: sync_status=Synced

    Plan->>SM: CreateMonitoringSchedule (data quality + bias)
    SM-->>CW: emit monitoring metrics
    CW-->>Plan: monitor_schedule_name

    Plan-->>Plan: build DeploymentReport
```

---

## Flowchart: Failure & Rollback Path

```mermaid
flowchart TD
    A[execute() starts] --> B{dvc_pull}
    B -->|fail| Z1[DeploymentReport: failed_stage=dvc_pull]
    B -->|ok| C{build_push}
    C -->|fail| Z2[DeploymentReport: failed_stage=build_push]
    C -->|ok| D{train}
    D -->|fail| Z3[rollback: delete incomplete job]
    D -->|ok| E{register}
    E -->|ok| F{deploy_sagemaker}
    F -->|fail| Z4[rollback: revert to previous endpoint config]
    F -->|ok| G{argocd_sync}
    G -->|fail| Z5[rollback: argocd app rollback]
    G -->|ok| H{monitor}
    H -->|ok| I[DeploymentReport: succeeded=True]
```

---

## Key Takeaways

1. **Seam failures are the real risk** — each stage hand-off (S3 URI to
   training job, model ARN to endpoint, image URI to KServe) must be
   validated before the next stage starts.
2. **`DeploymentStage.outputs`** is the contract between stages: each stage
   writes a typed dict; the next stage reads from it — no implicit shared
   state.
3. **Rollback must be designed upfront**: a failed endpoint update should
   revert to the previous production variant, not leave the endpoint in
   an error state.
4. **Argo CD as the final sync gate** ensures the Git config repo is
   authoritative — the deployment is not complete until Argo reports `Synced`.
5. `DeploymentReport.summary` gives on-call engineers a single dict to
   paste into an incident ticket: endpoint URL, training cost, spot savings,
   and which stage failed.
