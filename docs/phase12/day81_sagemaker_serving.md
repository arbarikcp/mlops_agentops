# Day 81 — SageMaker Registry + Endpoints

## WHY — Four Endpoint Types, Not One

Most teams discover SageMaker real-time endpoints and stop there. This is a costly
mistake: the four endpoint types exist because **latency vs throughput tradeoffs are
fundamentally different** for different use cases.

| Endpoint type | Latency | Throughput | Idle cost | Best for |
|---|---|---|---|---|
| **Real-time** | p99 < 100ms | High (auto-scale) | Always-on billing | Online serving, <6 MB payload |
| **Serverless** | p99 100–500ms (cold start) | Low | Pay-per-invocation | Intermittent traffic, dev/test |
| **Async** | Minutes (queue-based) | Very high (batch queue) | Minimal (queue only) | Long-running inference, >6 MB |
| **Batch Transform** | Hours (offline) | Massive | Zero (job-based billing) | Offline scoring, entire dataset |

> **Choosing wrong costs money or users:**
> - Real-time for batch scoring = paying 24/7 for a weekend job.
> - Serverless for real-time search = 400ms cold starts on every user query.
> - Real-time for 100 MB audio files = payload limit errors.

---

## HOW — SageMaker Model Registry

The **Model Registry** is a versioned catalog of approved model packages.
It separates **what** (the model artifact + inference image) from **where**
(which endpoint it is deployed to).

```
ModelPackageGroup: credit-risk-classifier
  ModelPackage v1  (PendingApproval -> Approved)
    ModelArtifacts: s3://artifacts/models/run-001/model.tar.gz
    InferenceSpec:  image=763104351884.../sklearn:1.2, instance=ml.c5.large
    MetadataProps:  auc=0.87, approval_date=2024-03-01
  ModelPackage v2  (PendingApproval)
    ModelArtifacts: s3://artifacts/models/run-002/model.tar.gz
    MetadataProps:  auc=0.89
```

### Registry approval flow

```mermaid
sequenceDiagram
    participant Pipeline as SM Pipeline
    participant Registry as Model Registry
    participant Reviewer as Human / Auto Approver
    participant Deploy as Deployment Step

    Pipeline->>Registry: RegisterModel(group=credit-risk, status=PendingApproval)
    Registry-->>Pipeline: ModelPackageArn v3
    Pipeline->>Reviewer: Notify (SNS / EventBridge)
    Reviewer->>Registry: UpdateApprovalStatus(Approved)
    Registry->>Deploy: EventBridge event: ModelApproved
    Deploy->>Deploy: CreateEndpoint(model_package_arn=v3)
```

---

## HOW — Real-Time Endpoints

Real-time endpoints serve synchronous, low-latency predictions. The endpoint stays
warm at all times and scales instances based on invocation rate.

### Auto-scaling policy

```
Scale-out: InvocationsPerInstance > 500 req/s -> add instance
Scale-in:  InvocationsPerInstance < 100 req/s for 5 min -> remove instance
Min instances: 1 (always warm)
Max instances: 10
```

### Endpoint configuration and variants

Real-time endpoints support **production variants** for A/B testing:

```mermaid
flowchart LR
    Client["Client Request"] --> EP["Endpoint\ncredit-risk-ep"]
    EP -->|"90% traffic"| V1["Variant A\nmodel-v1\n2x ml.c5.large"]
    EP -->|"10% traffic"| V2["Variant B\nmodel-v2\n1x ml.c5.large"]
    V1 --> Response["Response\n{score: 0.73}"]
    V2 --> Response
```

---

## HOW — Serverless Endpoints

Serverless endpoints have **no always-on instances**. SageMaker provisions
compute on each invocation (cold start: 1–5 seconds for first request after idle).

Best pattern: use for development/staging or workloads with < 1 req/min average.

```
ServerlessConfig:
  MemorySizeInMB: 2048      (128 to 6144)
  MaxConcurrency: 5         (max parallel invocations)
```

Cost: `$0.0000600 per GB-second` — zero cost when idle for minutes.

---

## HOW — Async Endpoints

Async endpoints accept requests into an **SQS queue** and process them
asynchronously. The client polls S3 for the result.

```mermaid
sequenceDiagram
    participant Client
    participant EP as Async Endpoint
    participant Queue as Internal Queue
    participant Worker as Inference Worker
    participant S3 as S3 Output Bucket
    participant SNS as SNS (optional)

    Client->>EP: POST /invocations (payload in S3 input path)
    EP-->>Client: 202 Accepted + output_location (S3 URI)
    EP->>Queue: Enqueue request
    Queue->>Worker: Dequeue when worker free
    Worker->>S3: Read input from S3
    Worker->>S3: Write prediction to output_location
    Worker->>SNS: Notify completion (optional)
    Client->>S3: Poll output_location until object exists
```

Use cases: document parsing, audio transcription, large image scoring (> 6 MB).

---

## HOW — Batch Transform

Batch Transform has no endpoint — it is a **job** that scores an entire S3 prefix
and writes results back to S3. No idle cost.

```
Input:  s3://ml-data/score/2024-03-01/*.csv
Output: s3://ml-artifacts/predictions/2024-03-01/
Strategy: MultiRecord (batches rows for throughput)
MaxConcurrentTransforms: 4
```

Best for: nightly scoring of full customer base, offline model evaluation.

---

## Data Structures — Class Diagram

```mermaid
classDiagram
    class SMModelPackage {
        +String model_package_group_name
        +String model_package_arn
        +Int model_package_version
        +String model_artifact_uri
        +String inference_image_uri
        +String approval_status
        +Dict model_metrics
        +Dict metadata_properties
        +approve() None
        +reject(reason) None
        +deploy(endpoint_name) SMEndpoint
    }

    class SMEndpointConfig {
        +String config_name
        +List~ProductionVariant~ production_variants
        +String kms_key_id
        +Dict data_capture_config
        +create() str
        +delete() None
    }

    class ProductionVariant {
        +String variant_name
        +String model_name
        +String instance_type
        +Int initial_instance_count
        +Float initial_variant_weight
        +ServerlessConfig serverless_config
        +to_dict() dict
    }

    class ServerlessConfig {
        +Int memory_size_in_mb
        +Int max_concurrency
        +to_dict() dict
    }

    class SMEndpoint {
        +String endpoint_name
        +String endpoint_config_name
        +String endpoint_status
        +String endpoint_type
        +create() None
        +update(new_config) None
        +invoke(payload) Dict
        +invoke_async(input_s3_uri) str
        +delete() None
        +get_auto_scaling_policy() Dict
    }

    class SMBatchTransform {
        +String job_name
        +String model_name
        +String input_s3_uri
        +String output_s3_uri
        +String instance_type
        +Int instance_count
        +String strategy
        +Int max_concurrent_transforms
        +create() None
        +wait() Dict
    }

    SMEndpointConfig "1" --> "many" ProductionVariant
    ProductionVariant "1" --> "0..1" ServerlessConfig
    SMEndpoint "1" --> "1" SMEndpointConfig
    SMModelPackage ..> SMEndpoint : deploys to
```

---

## HOW — Endpoint Lifecycle

```mermaid
flowchart TD
    A[RegisterModel - PendingApproval] --> B[Human approves in Registry]
    B --> C[CreateModel - pulls artifact + image]
    C --> D[CreateEndpointConfig - variants + scaling]
    D --> E[CreateEndpoint - InService]
    E --> F{Traffic pattern}
    F -->|Spike| G[Auto-scale out]
    F -->|Idle| H[Auto-scale in to min]
    E --> I[UpdateEndpoint - new config, zero downtime rolling]
    I --> E
    E --> J[DeleteEndpoint - stops billing]
```

---

## Choosing the Right Endpoint Type

```mermaid
flowchart TD
    A[New inference workload] --> B{Online or offline?}
    B -->|Offline - batch dataset| C[Batch Transform]
    B -->|Online - real requests| D{Payload size?}
    D -->|"> 6 MB"| E[Async Endpoint]
    D -->|"< 6 MB"| F{Traffic pattern?}
    F -->|Continuous or SLA < 200ms| G[Real-Time Endpoint]
    F -->|Intermittent or dev/test| H[Serverless Endpoint]
```

---

## Key Takeaways

1. **Model Registry decouples training from deployment** — approve once, deploy many times to different endpoints without repackaging.
2. **Real-time = always-on cost** — only use it when you have continuous traffic or strict latency SLAs.
3. **Serverless = zero idle cost** — perfect for development, staging, or sporadic production traffic; accept the cold-start penalty.
4. **Async = the right answer for large payloads** — queue-based; the client polls S3; no 6 MB limit; workers auto-scale from zero.
5. **Batch Transform = cheapest for offline scoring** — job-based billing, no endpoint to manage, reads/writes entire S3 prefixes.
6. **Production variants enable A/B testing** — split traffic by weight; measure metrics per variant before full cutover.
7. **Data capture is one flag** — enable it on the EndpointConfig to log request/response pairs to S3 for model monitoring.
