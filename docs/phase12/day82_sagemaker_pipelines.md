# Day 82 — SageMaker Pipelines, Model Approval & Lineage

## WHY — Pipelines Over Bare Argo CD

Argo Workflows is a capable orchestrator, but it knows nothing about SageMaker.
Every step that calls a SageMaker API requires a custom Argo task template, and
lineage tracking is entirely your responsibility.

SageMaker Pipelines removes this friction:

| Concern | Bare Argo | SageMaker Pipelines |
|---|---|---|
| Step integration | Custom HTTP task templates | Native step types (Training, Processing, etc.) |
| Lineage | DIY — write to external system | Automatic — every artifact, step, run tracked |
| Parameter propagation | Argo Workflow parameters | First-class Pipeline parameters with type checking |
| Model approval gate | Custom webhook/manual step | Built-in ConditionStep + RegisterModel + approval |
| Caching | Not built-in | Step-level cache with configurable TTL |
| UI | Argo UI (generic DAG) | Studio visual pipeline editor |
| IAM | K8s ServiceAccount | SM execution role, no extra RBAC |

> **Use SageMaker Pipelines when your orchestration is primarily SageMaker steps.**
> Use Argo/Airflow when your pipeline spans multiple systems (Spark, Snowflake, etc.)
> and SageMaker is just one participant.

---

## HOW — Pipeline Structure

A SageMaker Pipeline is a **DAG of steps** with typed parameters. Steps share
data via `ProcessingOutput` / `TrainingOutput` references — no manual S3 URI
string concatenation.

```
Pipeline: credit-risk-pipeline
  Parameters:
    training_instance_type: ml.m5.xlarge
    approval_threshold: 0.85
  Steps:
    1. ProcessingStep  (feature engineering)
    2. TrainingStep    (model training, depends on 1)
    3. ProcessingStep  (model evaluation, depends on 1 + 2)
    4. ConditionStep   (auc >= threshold? depends on 3)
       ├── True  -> RegisterModelStep (register as Approved)
       └── False -> FailStep (pipeline fails with message)
```

### Step dependency graph

```mermaid
flowchart TD
    P1["ProcessingStep\nfeature_engineering"] --> T1["TrainingStep\ntrain_model"]
    P1 --> P2["ProcessingStep\nevaluate_model"]
    T1 --> P2
    P2 --> C1{"ConditionStep\nauc >= threshold?"}
    C1 -->|True| R1["RegisterModelStep\nPendingApproval"]
    C1 -->|False| F1["FailStep\nAUC too low"]
    R1 --> A1["LambdaStep\nnotify_reviewer"]
```

---

## HOW — Step Caching

Step caching re-uses outputs from a previous execution if inputs and config are
identical. This avoids re-running expensive feature engineering on unchanged data.

```python
cache_config = CacheConfig(
    enable_caching=True,
    expire_after="30d"   # invalidate after 30 days
)

feature_step = ProcessingStep(
    name="feature-engineering",
    processor=processor,
    inputs=[...],
    outputs=[...],
    cache_config=cache_config
)
```

Cache hit conditions: same step name + same input S3 URIs + same container image
+ same arguments. Any change invalidates the cache for that step and all
downstream steps.

---

## HOW — Model Approval

The `RegisterModelStep` creates a `ModelPackage` in the Model Registry with status
`PendingApproval`. Approval can be:
- **Manual**: a human reviews metrics in SageMaker Studio and clicks Approve.
- **Automated**: a Lambda triggered by EventBridge checks metrics and calls
  `update_model_package(ApprovalStatus="Approved")` if thresholds are met.

### Automated approval flow

```mermaid
sequenceDiagram
    participant Pipeline as SM Pipeline
    participant Registry as Model Registry
    participant EB as EventBridge
    participant Lambda as Approval Lambda
    participant Deploy as CD System

    Pipeline->>Registry: RegisterModel(status=PendingApproval, metrics={auc:0.89})
    Registry->>EB: Event: ModelPackageStateChange (PendingApproval)
    EB->>Lambda: Trigger approval check
    Lambda->>Registry: GetModelPackage -> metrics
    Note over Lambda: auc=0.89 >= threshold=0.85 -> Approve
    Lambda->>Registry: UpdateApprovalStatus(Approved)
    Registry->>EB: Event: ModelPackageStateChange (Approved)
    EB->>Deploy: Trigger deployment pipeline (Argo CD / CodePipeline)
    Deploy->>Deploy: CreateEndpoint(model_package_arn)
```

---

## HOW — Lineage Tracking

SageMaker ML Lineage automatically records:
- **Artifacts**: datasets, model packages, images
- **Contexts**: experiments, trials, pipelines
- **Actions**: training job runs, approval actions
- **Associations**: links between the above

You can query lineage to answer:
- "Which training data was used to produce this endpoint?"
- "Which pipeline run created this model package?"
- "What changed between v1 and v2 of this model?"

```mermaid
flowchart LR
    DS["Artifact\nDataset\ns3://ml-data/train/v3"]
    Image["Artifact\nDocker Image\n763.../pytorch:2.1"]
    TJ["Action\nTrainingJob\ncredit-risk-2024-03"]
    MP["Artifact\nModelPackage\nv3 Approved"]
    EP["Context\nEndpoint\ncredit-risk-prod"]

    DS -->|"ContributedTo"| TJ
    Image -->|"ContributedTo"| TJ
    TJ -->|"Produced"| MP
    MP -->|"DeployedTo"| EP
```

---

## Data Structures — Class Diagram

```mermaid
classDiagram
    class SMPipeline {
        +String pipeline_name
        +String pipeline_description
        +List~PipelineParameter~ parameters
        +List~SMPipelineStep~ steps
        +String role_arn
        +create_or_update() str
        +start(parameter_overrides) str
        +describe() Dict
        +list_executions() List
    }

    class PipelineParameter {
        +String name
        +String parameter_type
        +Any default_value
        +to_dict() dict
    }

    class SMPipelineStep {
        +String step_name
        +String step_type
        +List~SMPipelineStep~ depends_on
        +CacheConfig cache_config
        +to_dict() dict
    }

    class ProcessingStep {
        +ScriptProcessor processor
        +List~ProcessingInput~ inputs
        +List~ProcessingOutput~ outputs
        +get_output(output_name) ProcessingOutput
    }

    class TrainingStep {
        +Estimator estimator
        +Dict inputs
        +get_output() TrainingOutput
    }

    class ConditionStep {
        +List~Condition~ conditions
        +List~SMPipelineStep~ if_steps
        +List~SMPipelineStep~ else_steps
    }

    class RegisterModelStep {
        +String model_package_group_name
        +String approval_status
        +Dict model_metrics
        +Dict metadata_properties
    }

    class SMModelApproval {
        +String model_package_arn
        +String approval_status
        +String approval_description
        +Float auc_threshold
        +check_metrics(metrics) bool
        +approve() None
        +reject(reason) None
        +trigger_deployment() None
    }

    class CacheConfig {
        +Bool enable_caching
        +String expire_after
        +to_dict() dict
    }

    SMPipeline "1" --> "many" PipelineParameter
    SMPipeline "1" --> "many" SMPipelineStep
    SMPipelineStep <|-- ProcessingStep
    SMPipelineStep <|-- TrainingStep
    SMPipelineStep <|-- ConditionStep
    SMPipelineStep <|-- RegisterModelStep
    SMPipelineStep "1" --> "0..1" CacheConfig
    RegisterModelStep ..> SMModelApproval : creates
```

---

## HOW — End-to-End Pipeline Execution

```mermaid
sequenceDiagram
    participant CI as CI/CD
    participant SM as SageMaker Pipelines
    participant S3 as S3
    participant Registry as Model Registry
    participant EB as EventBridge

    CI->>SM: StartPipelineExecution(parameter_overrides)
    SM-->>CI: ExecutionArn

    SM->>SM: ProcessingStep - feature engineering
    S3-->>SM: Cached? Yes - skip (CacheHit)

    SM->>SM: TrainingStep - train model (spot instance)
    SM->>S3: Write model.tar.gz

    SM->>SM: ProcessingStep - evaluate model
    SM->>S3: Write evaluation.json (auc=0.89)

    SM->>SM: ConditionStep - auc >= 0.85?
    Note over SM: True branch taken

    SM->>Registry: RegisterModel(PendingApproval)
    SM->>EB: ModelPackageStateChange event

    EB->>CI: Trigger notification / auto-approval Lambda
    SM-->>CI: Execution SUCCEEDED
```

---

## Key Takeaways

1. **SageMaker Pipelines = native SM orchestration** — step outputs wire directly to step inputs; no manual S3 URI management.
2. **Step caching cuts iteration time** — feature engineering cached for 30 days; changing only the model config skips the expensive preprocessing.
3. **ConditionStep is the quality gate** — pipeline fails fast if AUC is below threshold; no model gets registered unless it earns it.
4. **RegisterModel -> PendingApproval -> EventBridge** — the approval event triggers downstream CD automatically; humans (or Lambda) are the decision point.
5. **Lineage is automatic** — every training job, dataset, model artifact, and endpoint is connected in the lineage graph without extra code.
6. **Use Argo/Airflow when SM is not the only participant** — SageMaker Pipelines shines for SM-native workflows; multi-system pipelines need a generic orchestrator.
