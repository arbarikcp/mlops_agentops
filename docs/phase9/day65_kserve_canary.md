# Day 65 — KServe Canary, Traffic Splitting, Shadow/Mirror

## Why Canary for Models?

A canary deployment sends a small fraction of traffic to the new model version
before promoting it fully. This lets you detect accuracy regressions on real
traffic without impacting all users.

```
Before canary:   100% → model-v1
During canary:    95% → model-v1  +  5% → model-v2
After promote:   100% → model-v2
```

---

## Three Traffic Patterns

| Pattern | Traffic | New model receives | Use for |
|---|---|---|---|
| **Canary** | Live (split) | 5–20% of real requests | Gradual rollout with real feedback |
| **Shadow / Mirror** | Mirrored | 100% copy, responses discarded | Safe testing; zero user impact |
| **A/B** | Live (split by rule) | Specific user segment | Feature testing per user group |

---

## KServe Canary YAML

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: credit-risk
  namespace: ml-serving
spec:
  predictor:
    canaryTrafficPercent: 20      # 20% to canary, 80% to stable
    model:
      modelFormat: {name: sklearn}
      storageUri: s3://ml-models/credit-risk/v2.0/   # canary model
      resources:
        requests: {cpu: 500m, memory: 512Mi}
        limits: {cpu: "2", memory: 2Gi}
```

For the **stable** version, the previous revision is kept in place automatically
by KServe. Canary traffic goes to the new spec; everything else to stable.

---

## Shadow / Mirror Mode

```yaml
spec:
  predictor:
    model:
      modelFormat: {name: sklearn}
      storageUri: s3://ml-models/credit-risk/v2.0/
  # Mirror: stable handles all real traffic; v2 gets mirrored copies
  canaryTrafficPercent: 0
```

Shadow mode is implemented at the Istio VirtualService level — KServe generates
the VirtualService automatically when `canaryTrafficPercent: 0` + the new version exists.

---

## Canary Promotion Sequence

```mermaid
sequenceDiagram
    participant Dev
    participant IS as InferenceService
    participant SM as SLO Monitor
    participant CD as CD Pipeline

    Dev->>IS: patch canaryTrafficPercent=10
    IS-->>Dev: 10% traffic → v2 ✅
    SM->>SM: compare v1 AUC vs v2 AUC (5 min window)
    SM-->>Dev: v2 AUC = 0.862 vs v1 AUC = 0.847 ✅
    Dev->>IS: patch canaryTrafficPercent=50
    IS-->>Dev: 50/50 split ✅
    SM-->>Dev: no regression, SLO GREEN
    Dev->>CD: approve → patch canaryTrafficPercent=100
    CD->>IS: promote v2 to stable
    IS-->>Dev: 100% v2 ✅
```

---

## CanaryConfig Builder (Python)

```mermaid
classDiagram
    class CanaryConfig {
        +stable_storage_uri: str
        +canary_storage_uri: str
        +canary_traffic_pct: int
        +model_format: str
        +min_replicas: int
        +max_replicas: int
        +validate() None
        +to_patch_dict() dict
        +promote() CanaryConfig
        +rollback() CanaryConfig
    }
```
