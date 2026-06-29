# Day 79 — AWS Foundations for ML

## WHY — Infrastructure Primitives Are the Load-Bearing Walls

SageMaker, EKS, and every other AWS ML service sit on top of three primitives:
**S3** (storage), **IAM** (identity), and **VPC** (network). Getting these wrong
means you will rebuild them mid-project under production pressure — the worst
possible time.

- **S3 bucket policy misconfiguration** is the #1 cause of ML data leaks.
- **Overly-permissive IAM roles** turn a compromised training container into a
  full account takeover.
- **Missing VPC endpoints** cause training jobs to route data over the public
  internet, triggering egress charges and violating compliance requirements.

> **Principle:** Design S3 + IAM + VPC as a trio. They are not independent.

---

## HOW — S3 Bucket Policies for ML

### Bucket layout for ML

```
s3://company-ml-data/
  raw/                   <- ingestion landing zone (write-once)
  processed/             <- feature engineering output
  train/                 <- partitioned training sets
  eval/                  <- hold-out evaluation sets

s3://company-ml-artifacts/
  models/<run_id>/       <- model binaries + metadata
  checkpoints/<job_id>/  <- spot training recovery
  pipelines/<pipeline>/  <- pipeline step outputs
```

### Key S3 policy patterns

| Pattern | Policy element | Why |
|---|---|---|
| Deny public access | `"Effect":"Deny","Principal":"*","Condition":{"Bool":{"aws:SecureTransport":"false"}}` | Enforce HTTPS |
| Restrict to VPC | `"Condition":{"StringEquals":{"aws:SourceVpc":"vpc-xxxxx"}}` | No internet access |
| Cross-account read | `"Principal":{"AWS":"arn:aws:iam::ACCOUNT_B:role/ModelConsumer"}` | Controlled sharing |
| Lifecycle to Glacier | `Transition after 90 days` | Cost: raw data cold storage |

```mermaid
flowchart LR
    subgraph Account["AWS Account"]
        VPC["VPC (private subnet)"]
        SM["SageMaker Training Job"]
        EP["VPC Endpoint (S3 Gateway)"]
    end
    S3["S3 Bucket\n(company-ml-data)"]
    Internet["Public Internet"]

    SM -->|via VPC endpoint| EP
    EP --> S3
    SM -.->|blocked by bucket policy| Internet
    Note["Bucket policy: deny if not aws:SourceVpc"]
```

---

## HOW — IAM Roles for ML (Least-Privilege)

### Role hierarchy for ML workloads

```mermaid
classDiagram
    class IAMPolicyDoc {
        +String version
        +List~Statement~ statements
        +String policy_arn
        +to_json() str
        +validate() bool
        +attach_to_role(role_name) None
    }

    class Statement {
        +String effect
        +List~String~ actions
        +List~String~ resources
        +Dict principal
        +Dict condition
        +is_least_privilege() bool
    }

    class IAMRole {
        +String role_name
        +String role_arn
        +Dict trust_policy
        +List~IAMPolicyDoc~ attached_policies
        +assume_role_policy_doc() dict
        +add_policy(policy) None
    }

    class ECRLifecycleRule {
        +Int rule_priority
        +String description
        +String tag_status
        +Int count_number
        +String count_type
        +to_dict() dict
    }

    IAMRole "1" --> "many" IAMPolicyDoc
    IAMPolicyDoc "1" --> "many" Statement
```

### Three roles every ML team needs

**1. ml-training-role** (assumed by SageMaker training jobs)
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:PutObject",
    "s3:ListBucket"
  ],
  "Resource": [
    "arn:aws:s3:::company-ml-data/train/*",
    "arn:aws:s3:::company-ml-artifacts/*"
  ]
}
```

**2. ml-serving-role** (assumed by SageMaker endpoints)
```json
{
  "Effect": "Allow",
  "Action": ["s3:GetObject"],
  "Resource": ["arn:aws:s3:::company-ml-artifacts/models/*"]
}
```

**3. ml-pipeline-role** (assumed by SageMaker Pipelines)
```json
{
  "Effect": "Allow",
  "Action": [
    "sagemaker:CreateTrainingJob",
    "sagemaker:DescribeTrainingJob",
    "sagemaker:CreateProcessingJob",
    "iam:PassRole"
  ],
  "Resource": "*",
  "Condition": {
    "StringEquals": {"sagemaker:RootAccess": "Disabled"}
  }
}
```

### IAM PassRole — the critical bridge

```mermaid
sequenceDiagram
    participant Dev as Developer / CI
    participant SM as SageMaker Service
    participant IAM as IAM
    participant Job as Training Job Container

    Dev->>SM: CreateTrainingJob(RoleArn=ml-training-role)
    SM->>IAM: iam:PassRole check on caller identity
    IAM-->>SM: Caller has iam:PassRole -> Allow
    SM->>IAM: AssumeRole(ml-training-role)
    IAM-->>SM: Temporary credentials
    SM->>Job: Inject credentials into container environment
    Job->>S3: s3:GetObject (authorised by ml-training-role)
```

---

## HOW — ECR Image Lifecycle

Unmanaged ECR repositories fill up fast: every CI push produces a new image layer
set, and ECR charges per GB stored (~$0.10/GB/month).

### ECRLifecycleRule patterns

```mermaid
flowchart TD
    A[Image pushed to ECR] --> B{Has 'production' tag?}
    B -->|Yes| C[Keep forever - protected by lifecycle rule]
    B -->|No| D{Has 'staging' tag?}
    D -->|Yes| E[Keep last 10 images]
    D -->|No| F{Untagged?}
    F -->|Yes| G[Delete after 1 day]
    F -->|No| H[Delete after 30 days - dev/feature images]
```

### Lifecycle policy JSON (CDK/Terraform pattern)

```python
class ECRLifecycleRule:
    """Single ECR lifecycle rule."""
    rule_priority: int          # lower = evaluated first
    description: str
    tag_status: str             # "tagged" | "untagged" | "any"
    tag_prefix_list: list[str]  # e.g. ["prod", "staging"]
    count_type: str             # "imageCountMoreThan" | "sinceImagePushed"
    count_number: int           # e.g. 10 images or 30 days
```

Example rules applied in order:
1. Priority 1: Keep all images tagged `prod-*` (count > 999 = never expire)
2. Priority 2: Keep last 5 images tagged `staging-*`
3. Priority 3: Expire untagged images after 1 day
4. Priority 4: Expire all other images after 30 days

---

## HOW — VPC for ML

### VPC topology for ML workloads

```mermaid
flowchart TB
    subgraph VPC["VPC 10.0.0.0/16"]
        subgraph PublicSubnet["Public Subnet 10.0.1.0/24"]
            NAT["NAT Gateway"]
            BastionHost["Bastion Host"]
        end
        subgraph PrivateSubnetA["Private Subnet A 10.0.2.0/24"]
            SMTraining["SageMaker Training"]
            SMEP["SageMaker Endpoint"]
        end
        subgraph PrivateSubnetB["Private Subnet B 10.0.3.0/24"]
            EKSNode["EKS Worker Node"]
        end
        subgraph VPCEndpoints["VPC Endpoints"]
            S3EP["S3 Gateway Endpoint"]
            ECREP["ECR Interface Endpoint"]
            SMEP2["SageMaker API Interface Endpoint"]
        end
    end

    PrivateSubnetA --> S3EP
    PrivateSubnetA --> ECREP
    PrivateSubnetB --> S3EP
    PrivateSubnetB --> ECREP
    PrivateSubnetA -->|outbound only| NAT
    NAT --> Internet["Internet Gateway"]
```

### VPCConfig class

```python
class VPCConfig:
    """VPC configuration for SageMaker training/serving jobs."""
    vpc_id: str
    subnet_ids: list[str]          # private subnets only
    security_group_ids: list[str]
    enable_network_isolation: bool  # blocks all outbound (for compliance)
    s3_vpc_endpoint_id: str        # Gateway endpoint - free
    ecr_vpc_endpoint_id: str       # Interface endpoint - $0.01/hr

    def validate(self) -> bool:
        """Ensure subnets are private (no route to IGW)."""
        ...

    def to_sagemaker_dict(self) -> dict:
        """Return VpcConfig dict for SageMaker API calls."""
        return {
            "Subnets": self.subnet_ids,
            "SecurityGroupIds": self.security_group_ids
        }
```

### VPC Endpoint types for ML

| Endpoint | Type | Cost | Why needed |
|---|---|---|---|
| S3 Gateway | Gateway (free) | $0 | Route S3 traffic inside VPC |
| ECR API | Interface | $0.01/hr | Pull images from private subnet |
| ECR DKR | Interface | $0.01/hr | Docker layer pull (separate from API) |
| SageMaker API | Interface | $0.01/hr | Jobs in private subnet call SM API |
| STS | Interface | $0.01/hr | AssumeRole inside VPC |

> S3 Gateway endpoint is always free — there is no reason not to create one.

---

## End-to-End Flow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant CI as CI Pipeline
    participant ECR as ECR Repository
    participant SM as SageMaker
    participant S3 as S3 (via VPC endpoint)
    participant IAM as IAM

    Dev->>CI: git push
    CI->>ECR: docker build + push (tag=commit-sha)
    CI->>SM: CreateTrainingJob(image=ECR URI, role=ml-training-role, vpc=VPCConfig)
    SM->>IAM: AssumeRole(ml-training-role)
    IAM-->>SM: Temporary creds
    SM->>S3: s3:GetObject train data (via VPC endpoint, no internet)
    SM->>S3: s3:PutObject model artifact
    SM-->>CI: TrainingJob COMPLETED
    CI->>ECR: ECR lifecycle rule cleans old dev images
```

---

## Key Takeaways

1. **S3 + IAM + VPC are a trio** — design them together; changing one affects the others.
2. **Deny public S3 access by default** — add bucket policies that require `aws:SecureTransport` and `aws:SourceVpc`.
3. **Three ML roles minimum** — training, serving, pipeline — each with scoped resource ARNs, not `"Resource": "*"`.
4. **iam:PassRole is the critical bridge** — the calling identity must have PassRole permission to delegate a role to SageMaker.
5. **S3 Gateway endpoint is always free** — create it in every VPC that runs ML jobs.
6. **ECR lifecycle rules prevent runaway storage costs** — expire untagged images in 1 day, dev images in 30 days; protect production tags.
7. **Private subnets + VPC endpoints = compliance-safe ML** — training data never traverses the public internet.
