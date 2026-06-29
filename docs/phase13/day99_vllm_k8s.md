# Day 99 — vLLM on Kubernetes + GPU Metrics + Capacity Planning

## WHY

Single-node vLLM can handle ~50–200 req/s depending on model size and hardware. Production LLM APIs often need 1,000+ req/s and must handle:

- **Traffic spikes:** Autoscale replicas within seconds, not minutes.
- **Rolling updates:** Zero-downtime model upgrades.
- **Resource quotas:** Prevent one team's workload from starving another's GPU pool.
- **Observability:** Know exactly how many tokens/second each replica produces.

Kubernetes provides all of this — but LLM workloads need GPU-aware HPA metrics (queue depth, decode throughput) not CPU-based defaults.

---

## HOW

### Deployment with GPU Limits

```yaml
resources:
  requests:
    nvidia.com/gpu: "4"     # tensor_parallel_size × pipeline_parallel_size
    memory: "24Gi"
  limits:
    nvidia.com/gpu: "4"
```

GPU scheduling: `nvidia.com/gpu` is an extended resource. K8s will only schedule the pod on a node with available GPU slots.

**Node tolerations:** GPU nodes typically have taints; pods must tolerate:
```yaml
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### HPA with Custom Metrics

CPU-based HPA is useless for LLM workloads (GPU is the bottleneck, CPU is idle). vLLM exposes Prometheus metrics including:

- `vllm:request_success_total` — completed requests
- `vllm:num_requests_running` — decode queue depth
- `vllm:gpu_cache_usage_perc` — KV cache utilization

KEDA or custom metrics adapter exposes these to HPA as `vllm_request_rate`.

### Capacity Planning Formula

```
replicas_needed = ceil(target_rps × safety_factor / single_replica_throughput)
```

**Example:** 100 req/s target, 20 req/s per replica, 1.2 safety factor:
```
ceil(100 × 1.2 / 20) = ceil(6.0) = 6 replicas
hourly_cost = 6 × $3/hr = $18/hr
```

### PodMonitor (Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
spec:
  podMetricsEndpoints:
    - port: metrics
      interval: 15s
      path: /metrics
```

vLLM exposes `/metrics` (Prometheus format) out of the box. PodMonitor tells the Prometheus Operator to scrape all pods with matching labels every 15s.

---

## Class Diagram

```mermaid
classDiagram
    class VLLMEngineConfig {
        +str model
        +int tensor_parallel_size
        +int pipeline_parallel_size
        +total_parallel_size() int
    }

    class VLLMDeploymentSpec {
        +str name
        +str image
        +VLLMEngineConfig engine_config
        +int replicas
        +str gpu_resource
        +str memory_limit
        +__post_init__()
        +to_manifest() dict
        +to_dict() dict
    }

    class VLLMServiceSpec {
        +str name
        +str deployment_name
        +int port
        +str service_type
        +__post_init__()
        +to_manifest() dict
        +to_dict() dict
    }

    class VLLMHPASpec {
        +str name
        +str deployment_name
        +int min_replicas
        +int max_replicas
        +float target_rps
        +__post_init__()
        +to_manifest() dict
        +to_dict() dict
    }

    class CapacityPlan {
        +float target_rps
        +float single_replica_throughput
        +float safety_factor
        +float gpu_cost_per_hour
        +__post_init__()
        +replicas_needed() int
        +hourly_cost_usd() float
        +to_dict() dict
    }

    class PodMonitorSpec {
        +str name
        +str namespace
        +str port_name
        +str scrape_interval
        +__post_init__()
        +to_manifest() dict
        +to_dict() dict
    }

    VLLMDeploymentSpec --> VLLMEngineConfig
    VLLMHPASpec ..> VLLMDeploymentSpec
    PodMonitorSpec ..> VLLMDeploymentSpec
```

---

## Sequence Diagram — HPA Scale-Up Event

```mermaid
sequenceDiagram
    participant P as Prometheus
    participant MA as Metrics Adapter
    participant HPA as HPA Controller
    participant API as K8s API Server
    participant D as vLLM Deployment

    P->>D: Scrape /metrics (every 15s)
    D-->>P: vllm_request_rate = 45 req/s
    P->>MA: Store metric
    HPA->>MA: Query vllm_request_rate
    MA-->>HPA: avg = 45 req/s (per pod, target = 10)
    HPA->>HPA: desired_replicas = ceil(45/10) = 5 (currently 2)
    HPA->>API: Patch Deployment replicas=5
    API->>D: Scale up to 5 pods
    D->>D: New pods pull image, start vLLM engine
    Note over D: ~60-90s to be ready (model load)
    D-->>API: Pods Ready
    API-->>HPA: Scale complete
```

---

## Flow Diagram — Capacity Planning Decision

```mermaid
flowchart TD
    A[target_rps, single_replica_rps] --> B[replicas = ceil\ntarget × safety / replica_rps]
    B --> C{replicas × gpu_cost > budget?}
    C -->|Yes| D[Consider quantization\nto improve replica_rps]
    D --> E[Re-run with AWQ INT4\nreplica_rps × 2]
    E --> B
    C -->|No| F[Deploy replicas]
    F --> G[Set HPA min=1, max=replicas×2]
    G --> H[Monitor vllm_gpu_cache_usage_perc]
    H --> I{cache > 80%?}
    I -->|Yes| J[Scale up or reduce max_model_len]
    I -->|No| K[Healthy serving fleet]
```

---

## Kubernetes Manifest Summary

| Resource | Kind | Purpose |
|----------|------|---------|
| `VLLMDeploymentSpec` | `apps/v1/Deployment` | Pod template with GPU limits + health probes |
| `VLLMServiceSpec` | `v1/Service` | ClusterIP/LoadBalancer for routing |
| `VLLMHPASpec` | `autoscaling/v2/HorizontalPodAutoscaler` | Custom metric scale-up/down |
| `PodMonitorSpec` | `monitoring.coreos.com/v1/PodMonitor` | Prometheus scraping |

---

## Key Takeaways

1. **GPU limits** must match `engine.total_parallel_size()` — tensor_parallel × pipeline_parallel.
2. **CPU-based HPA is wrong for LLMs** — always use `vllm_request_rate` or `num_requests_running`.
3. **Safety factor 1.2** = 20% headroom for traffic spikes before new replicas come online.
4. **PodMonitor** auto-discovers new replicas as they scale — no manual Prometheus config.
5. **Model load time** (~60–90s) limits HPA reaction speed; over-provision min_replicas for latency-critical services.
