# Day 75 — Progressive Delivery for Models

## The Problem with Big-Bang Model Deploys

A "big-bang" model deploy routes 100% of traffic to the new model immediately.
If the new model has a silent regression (lower AUC on a specific slice), every
prediction is wrong before the alert fires.

Progressive delivery routes a small fraction of traffic to the new model first,
observes SLIs, and promotes or rolls back.

---

## Three Traffic Patterns

| Pattern | Description | Use case |
|---|---|---|
| **Blue-Green** | Two full deployments, instant switch | Zero-downtime code deploys; not great for ML (no gradual observation) |
| **Canary** | New version gets N% of live traffic | Model rollouts — real traffic, real distribution, controllable blast radius |
| **Shadow / Mirror** | Copy requests to new model; responses discarded | Validate before any real exposure; no latency impact on users |

---

## Canary Sequence

```mermaid
sequenceDiagram
    participant CI as CI Pipeline
    participant Git as Config Repo
    participant Argo as Argo CD
    participant K8s as Kubernetes
    participant Mon as Prometheus

    CI->>Git: push values: canary_weight=10, canary_model=v1.3
    Git->>Argo: webhook
    Argo->>K8s: sync: update KServe canaryTrafficPercent=10
    K8s-->>Mon: 10% traffic → v1.3 predictions logged
    Mon->>CI: PSI, AUC on shadow traffic OK (30 min)
    CI->>Git: push: canary_weight=50
    Argo->>K8s: sync: canaryTrafficPercent=50
    Mon->>CI: SLOs still passing (15 min)
    CI->>Git: push: canary_weight=100 (promote)
    Argo->>K8s: sync: stable_model=v1.3, remove canary
    K8s-->>Mon: 100% traffic on v1.3 ✅
```

---

## Argo Rollouts (for non-KServe deployments)

For Deployments (not KServe), Argo Rollouts provides a `Rollout` CRD:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: credit-risk-api
  namespace: ml-serving
spec:
  replicas: 5
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: {duration: 30m}       # observe for 30 min
        - analysis:
            templates:
              - templateName: auc-check
        - setWeight: 50
        - pause: {duration: 15m}
        - setWeight: 100
      canaryMetadata:
        labels:
          model-variant: canary
      stableMetadata:
        labels:
          model-variant: stable
      antiAffinity:
        preferredDuringSchedulingIgnoredDuringExecution:
          weight: 1
  selector:
    matchLabels:
      app: credit-risk-api
  template:
    metadata:
      labels:
        app: credit-risk-api
    spec:
      containers:
        - name: api
          image: ghcr.io/arbarikcp/credit-risk-api:v2.1.0
```

### AnalysisTemplate — AUC Guard as a Rollout Gate

```yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: auc-check
  namespace: ml-serving
spec:
  metrics:
    - name: model-auc
      interval: 5m
      successCondition: result >= 0.78
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            avg_over_time(
              ml_model_auc{model_variant="canary"}[10m]
            )
    - name: psi-check
      interval: 5m
      successCondition: result < 0.2
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            ml_prediction_psi_score{model_variant="canary"}
```

---

## Blue-Green for Code (not model) Changes

When the API code changes (new endpoint, breaking API contract), use
blue-green to validate the new version before switching:

```mermaid
graph LR
    LB[Load Balancer] -->|100%| Blue[Blue: v2.0 - current]
    Blue -.->|0%| Green[Green: v2.1 - candidate]

    LB2[After smoke test] -->|0%| Blue2[Blue: v2.0 - idle]
    LB2 -->|100%| Green2[Green: v2.1 - active]
```

---

## Rollout Decision Matrix

```mermaid
graph TD
    A[New model ready] --> B{SLO gate passed?}
    B -->|No| C[Block — fix model]
    B -->|Yes| D[Deploy at 10% canary]
    D --> E{PSI < 0.2 after 30 min?}
    E -->|No| F[Rollback to stable]
    E -->|Yes| G[Promote to 50%]
    G --> H{AUC on canary traffic >= 0.78?}
    H -->|No| F
    H -->|Yes| I[Promote to 100%]
    I --> J[Archive old model — keep 2 versions]
```

---

## Class Diagram

```mermaid
classDiagram
    class CanaryStep {
        +weight: int
        +pause_minutes: int
        +analysis_template: str
        +to_dict() dict
    }

    class RolloutStrategy {
        +steps: list~CanaryStep~
        +max_surge: int
        +max_unavailable: int
        +validate() None
        +to_dict() dict
    }

    class AnalysisMetric {
        +name: str
        +prometheus_query: str
        +success_condition: str
        +failure_limit: int
        +interval_m: int
        +to_dict() dict
    }

    class AnalysisTemplate {
        +name: str
        +namespace: str
        +metrics: list~AnalysisMetric~
        +to_manifest() dict
    }

    class ArgoRollout {
        +name: str
        +namespace: str
        +image: str
        +replicas: int
        +strategy: RolloutStrategy
        +to_manifest() dict
    }

    ArgoRollout --> RolloutStrategy
    RolloutStrategy --> CanaryStep
    AnalysisTemplate --> AnalysisMetric
```
