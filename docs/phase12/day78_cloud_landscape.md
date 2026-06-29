# Day 78 — Cloud Landscape for ML

## WHY — The Managed-vs-DIY Decision Matters

Every ML team eventually hits the same fork: **do we run everything on managed cloud
services, or do we build a self-managed Kubernetes cluster?** The answer is not
philosophical — it is a cost, control, and velocity trade-off that must be made
deliberately, per workload.

Getting this wrong is expensive:
- Going full DIY too early burns engineer time on plumbing instead of models.
- Going full managed too long locks you into vendor pricing with no egress escape.
- Mixing both without a decision matrix leads to zombie infrastructure.

> **IAM-first principle:** Before writing a single training job, design your
> identity and permissions model. Retrofitting least-privilege IAM is 10x harder
> than designing it upfront.

---

## HOW — The Decision Matrix

### Managed-for-Dev vs K8s-for-Prod

| Dimension | Managed Service (e.g., SageMaker) | Self-managed K8s (EKS/GKE/AKS) |
|---|---|---|
| **Setup time** | Hours | Days–weeks |
| **Operational burden** | Low (vendor manages infra) | High (you manage nodes, upgrades) |
| **Customisation** | Limited to service APIs | Full control (custom runtimes, GPU sharing) |
| **Cost at scale** | High (service premium ~30–60%) | Lower but unpredictable ops cost |
| **Spot/preemptible** | Managed spot with auto-retry | Manual spot node groups |
| **Multi-framework** | Framework-specific containers | Any container |
| **Compliance / VPC** | VPC endpoints, PrivateLink | Full network control |
| **Best for** | Experimentation, fast onboarding | Production, cost-sensitive, custom GPU |

### When to use each

```mermaid
flowchart TD
    A[New ML workload] --> B{Team size?}
    B -->|1-5 engineers| C[Managed service first]
    B -->|5+ engineers| D{Monthly spend > $10k?}
    D -->|No| C
    D -->|Yes| E{Custom runtime needed?}
    E -->|No| C
    E -->|Yes| F[K8s / EKS path]
    C --> G{Scaling pain or cost cliff?}
    G -->|Yes| F
    G -->|No| C
```

---

## HOW — IAM-First Principle

IAM (Identity and Access Management) is the foundation. Every cloud resource
access must be authorised by an **identity** — not a shared key.

### Core IAM concepts across providers

| Concept | AWS | GCP | Azure |
|---|---|---|---|
| Human identity | IAM User / SSO | Google Account | Azure AD User |
| Machine identity | IAM Role (assumed) | Service Account | Managed Identity |
| Policy language | JSON policy doc | IAM Conditions | Azure RBAC JSON |
| Delegation | AssumeRole (STS) | Workload Identity | Federated Identity |
| Secrets | Secrets Manager | Secret Manager | Key Vault |

### Least-privilege pattern for ML

```mermaid
sequenceDiagram
    participant Job as Training Job
    participant STS as STS / Token Service
    participant IAM as IAM Policy
    participant S3 as S3 / GCS / Blob
    participant ECR as Container Registry

    Job->>STS: AssumeRole (role=ml-training-role)
    STS-->>Job: Temporary credentials (15 min TTL)
    Job->>IAM: Request validated against policy
    IAM-->>Job: Allow s3:GetObject on bucket/prefix/*
    Job->>S3: Read training data
    Job->>ECR: Pull training image (ecr:GetAuthorizationToken)
    Note over IAM: No s3:DeleteObject, no cross-account access
```

---

## HOW — Cost Model

Cloud ML cost has three axes:

### 1. Compute

```
Cost = (instance_type_$/hr) x (runtime_hrs) x (num_instances)
     + (spot_discount ~70%) if fault-tolerant
```

| Phase | Instance class | Typical choice |
|---|---|---|
| Experimentation | CPU general | m5.xlarge / n1-standard-4 |
| Training (small) | GPU single | p3.2xlarge / a2-highgpu-1g |
| Training (large) | GPU multi | p4d.24xlarge / a3-highgpu-8g |
| Serving (latency) | CPU optimised | c5.2xlarge / c2-standard-8 |
| Serving (GPU) | GPU inference | g4dn.xlarge / g2-standard-4 |

### 2. Storage

```
Cost = (GB_stored x $/GB/month)
     + (GET_requests x $/1000)
     + (PUT_requests x $/1000)
     + (data_transfer_out x $/GB)   <- egress
```

> Egress is the hidden cost. Pulling 1 TB of training data from S3 to an on-prem
> node costs ~$90. Always co-locate compute and storage in the same region.

### 3. Egress Triangle

```mermaid
flowchart LR
    subgraph Region["AWS us-east-1"]
        S3["S3 Bucket"]
        EC2["EC2 / EKS"]
        SM["SageMaker"]
    end
    Internet["Internet / On-Prem"]

    S3 <-->|Free| EC2
    S3 <-->|Free| SM
    EC2 -->|"$0.09/GB"| Internet
    S3 -->|"$0.09/GB"| Internet
```

---

## HOW — Provider Comparison for ML

### AWS for ML (most mature ecosystem)

| Service | Purpose |
|---|---|
| S3 | Training data, model artifacts |
| ECR | Container image registry |
| SageMaker | End-to-end ML platform |
| EKS | Custom K8s inference |
| Bedrock | Managed foundation models |
| Step Functions | ML pipeline orchestration alternative |

### GCP for ML (best TPU, tightest Vertex integration)

| Service | Purpose |
|---|---|
| GCS | Storage |
| Artifact Registry | Container images |
| Vertex AI | End-to-end ML (SageMaker equivalent) |
| GKE | K8s inference |
| Model Garden | Foundation models |
| TPU v4/v5 | Large-scale training |

### Azure for ML (enterprise / Microsoft shop)

| Service | Purpose |
|---|---|
| Azure Blob | Storage |
| ACR | Container registry |
| Azure ML | End-to-end ML |
| AKS | K8s inference |
| Azure OpenAI | Foundation models |

### Decision heuristic

```mermaid
flowchart TD
    A[Choose cloud provider] --> B{Existing infrastructure?}
    B -->|AWS heavy| C[AWS + SageMaker]
    B -->|GCP heavy| D[GCP + Vertex AI]
    B -->|Azure / MS ecosystem| E[Azure ML]
    B -->|Greenfield| F{Primary workload?}
    F -->|LLM fine-tune / TPU| D
    F -->|Enterprise compliance| E
    F -->|General ML + widest tooling| C
```

---

## Data Structures — Class Diagram

```mermaid
classDiagram
    class CloudProvider {
        +String name
        +String region
        +List~String~ available_services
        +Float egress_cost_per_gb
        +validate_region() bool
    }

    class IAMPolicy {
        +String policy_id
        +String effect
        +List~String~ actions
        +List~String~ resources
        +Dict conditions
        +to_json() str
        +validate_least_privilege() bool
    }

    class CostModel {
        +String instance_type
        +Float compute_cost_per_hr
        +Float storage_cost_per_gb_month
        +Float egress_cost_per_gb
        +Bool spot_eligible
        +estimate_training_cost(hours, instances) float
        +estimate_storage_cost(gb, months) float
        +estimate_egress_cost(gb) float
    }

    class MLWorkload {
        +String workload_id
        +String phase
        +String framework
        +Bool gpu_required
        +Bool fault_tolerant
        +CloudProvider provider
        +CostModel cost_model
        +IAMPolicy iam_policy
        +choose_deployment_target() str
    }

    CloudProvider "1" --> "many" IAMPolicy
    MLWorkload "1" --> "1" CloudProvider
    MLWorkload "1" --> "1" CostModel
    MLWorkload "1" --> "1" IAMPolicy
```

---

## Key Takeaways

1. **Managed services first, K8s when justified** — start with SageMaker/Vertex, migrate to K8s only when cost, custom runtime, or GPU-sharing forces it.
2. **IAM-first is non-negotiable** — design identity and least-privilege policies before the first training job runs.
3. **Egress is the hidden cost** — always co-locate compute and storage; never pull training data cross-region.
4. **Three cost axes** — compute (instance type x runtime), storage (GB-month + requests), egress (outbound data). Model all three before committing.
5. **Provider choice is mostly ecosystem** — AWS for breadth, GCP for TPUs/LLMs, Azure for enterprise compliance.
6. **Spot/preemptible is free money** — if your training job checkpoints, using spot cuts compute cost by 60–70%.
