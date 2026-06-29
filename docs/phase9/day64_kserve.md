# Day 64 — KServe InferenceService: Predictor + Transformer, Scale-to-Zero

## What is KServe?

KServe (formerly KFServing) is a K8s-native model serving framework built on top of
Knative and Istio. It provides:

- **InferenceService CRD** — one YAML to deploy a model
- **Scale-to-zero** — pod count drops to 0 when no traffic; scales up on request
- **Protocol standardisation** — V2 Inference Protocol (REST + gRPC)
- **Transformer pipeline** — pre/post-processing sidecar before/after the predictor
- **Canary traffic splitting** — A/B test two model versions at the router level

---

## InferenceService Architecture

```mermaid
graph LR
    Client -->|POST /v1/models/credit-risk:predict| Ingress
    Ingress --> IS[InferenceService]
    IS --> Transformer[Transformer\npre/post-process]
    Transformer --> Predictor[Predictor\nSKLearn / PyTorch / ONNX]
    Predictor --> Model[(Model artifact\nS3 / PVC)]
```

---

## InferenceService YAML

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: credit-risk
  namespace: ml-serving
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn           # or pytorch, onnx, tensorflow, lightgbm
      storageUri: s3://ml-models/credit-risk/v1.2/
      resources:
        requests:
          cpu: 500m
          memory: 512Mi
        limits:
          cpu: "2"
          memory: 2Gi
    minReplicas: 1
    maxReplicas: 10
    scaleTarget: 10             # scale when requests/pod > 10
    scaleMetric: rps
```

---

## Transformer Pattern

A transformer is a separate container that runs pre/post-processing:

```
Client request (raw JSON)
  → Transformer (feature extraction, normalisation)
  → Predictor (model inference)
  → Transformer (score formatting, decision band)
  → Client response
```

```yaml
spec:
  transformer:
    containers:
      - name: credit-risk-transformer
        image: credit-risk-transformer:v1
        resources:
          requests: {cpu: 200m, memory: 256Mi}
          limits: {cpu: "1", memory: 512Mi}
        env:
          - name: PREDICTOR_HOST
            value: localhost   # transformer calls predictor on localhost
  predictor:
    model:
      modelFormat: {name: sklearn}
      storageUri: s3://ml-models/credit-risk/v1.2/
```

---

## Scale-to-Zero

KServe uses Knative Serving's concurrency-based autoscaler:

```mermaid
graph TD
    T0[0 requests → 0 pods]
    T1[Request arrives → activator buffers]
    T2[Scale from 0 → 1 pod cold start]
    T3[Pod ready → request served]
    T4[Traffic sustained → scale up to maxReplicas]
    T5[Traffic stops → scale down → 0 after 60s]

    T0 --> T1 --> T2 --> T3 --> T4 --> T5 --> T0
```

- **Cold start**: 5–30s (model download + warmup)
- **Scale-up**: Knative autoscaler checks every 2s
- **Scale-to-zero delay**: `scaleToZeroGracePeriodSeconds: 60`

---

## KServe Spec Builder (Python)

```mermaid
classDiagram
    class InferenceServiceSpec {
        +name: str
        +namespace: str
        +model_format: str
        +storage_uri: str
        +min_replicas: int
        +max_replicas: int
        +scale_target: int
        +scale_metric: str
        +has_transformer: bool
        +transformer_image: str
        +to_manifest() dict
    }
```
