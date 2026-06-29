# Day 97 — GPU Utilization & Cost: MIG, Fractional GPUs, Spot, Cost Optimization

## WHY

A single A100-80GB costs ~$3/hr on AWS. Most inference workloads use only 10–30% of a GPU. Without optimization:

- **$3/hr × 10% utilization = $30 effective cost per GPU-hour of useful work**
- A 7B model serving 1 req/s uses < 5% of an A100

MIG, fractional GPU techniques, and spot instances can cut costs by **60–85%** while maintaining the same throughput.

---

## HOW

### MIG (Multi-Instance GPU)

Available on A100 and H100 GPUs. Partitions the physical GPU into up to 7 isolated GPU instances, each with:
- Dedicated SM partition (guaranteed compute)
- Dedicated HBM memory slice
- Dedicated L2 cache bandwidth
- Hardware-level isolation (no interference between instances)

```
A100-40GB partitioned as 7 × MIG_1g.5gb:
  Each instance: ~6.2 GPC SMs + 5 GB HBM
  Cost: $3/hr ÷ 7 ≈ $0.43/hr per model
```

### CUDA MPS (Software Multiplexing)

Multiple processes share the GPU via CUDA Multi-Process Service. Less isolation than MIG (shared L2 cache, no memory guarantees) but works on older GPUs (T4, V100).

### Spot / Preemptible GPUs

Cloud providers offer unused GPU capacity at 60–80% discount. Interruption risk managed by:
1. Checkpoint every N steps
2. Handle SIGTERM → save state immediately
3. Resume from last checkpoint on restart

```
Spot savings: on_demand × spot_discount = 70% discount
$3.00/hr → $0.90/hr for identical hardware
```

### GPU Utilization Metrics

| Metric | Healthy | Under-utilized | Fix |
|--------|---------|---------------|-----|
| SM Efficiency | > 70% | < 50% | Increase batch size |
| Memory BW | > 60% | < 40% | Larger model or FP16 |
| Tensor Core | > 50% | < 30% | Enable TF32 / BF16 |

---

## Class Diagram

```mermaid
classDiagram
    class MIGProfile {
        <<enumeration>>
        MIG_1g_5gb = "1g.5gb"
        MIG_2g_10gb = "2g.10gb"
        MIG_3g_20gb = "3g.20gb"
        MIG_4g_20gb = "4g.20gb"
        MIG_7g_40gb = "7g.40gb"
    }

    class GPUInstance {
        <<enumeration>>
        A100_40GB
        A100_80GB
        H100_80GB
        T4
        L4
        V100
    }

    class MIGConfig {
        +GPUInstance gpu_instance
        +MIGProfile profile
        +int num_instances
        +__post_init__()
        +memory_per_instance_gb() float
        +to_dict() dict
    }

    class SpotConfig {
        +str instance_type
        +float on_demand_price_usd
        +float spot_discount
        +__post_init__()
        +spot_price() float
        +savings_usd_per_hour() float
        +to_dict() dict
    }

    class GPUCostModel {
        +GPUInstance gpu_instance
        +float cost_per_hour_usd
        +float utilization_target
        +__post_init__()
        +effective_cost_per_hour() float
        +to_dict() dict
    }

    class GPUUtilizationReport {
        +GPUInstance gpu_instance
        +float sm_efficiency
        +float memory_bandwidth_util
        +float tensor_core_util
        +__post_init__()
        +is_underutilized() bool
        +optimization_hints() list~str~
        +to_dict() dict
    }

    MIGConfig --> GPUInstance
    MIGConfig --> MIGProfile
    GPUCostModel --> GPUInstance
    GPUUtilizationReport --> GPUInstance
```

---

## Sequence Diagram — Spot Instance Checkpoint-Resume

```mermaid
sequenceDiagram
    participant TC as Training Code
    participant SP as Spot Instance
    participant CP as Checkpoint Store (S3)
    participant NI as New Instance

    TC->>SP: Training running (step 1500)
    TC->>CP: Save checkpoint (step 1000, every 500 steps)
    SP->>TC: SIGTERM received (preemption in 2 min)
    TC->>TC: Interrupt handler fires
    TC->>CP: Emergency checkpoint (step 1500)
    SP->>SP: Instance terminated

    Note over NI: New spot instance acquired
    NI->>CP: Load latest checkpoint (step 1500)
    NI->>NI: Resume training from step 1500
    NI->>TC: Continue training
```

---

## Cost Optimization Matrix

| Workload | Recommended Approach | Savings |
|----------|---------------------|---------|
| Small models (< 5GB) inference | MIG 1g.5gb × 7 | 7× density |
| Development / experiments | Spot + checkpointing | 70% |
| Latency-sensitive production | On-demand + utilization monitoring | 0% |
| Batch inference (flexible SLA) | Spot + queue depth autoscaling | 70% |
| Fine-tuning | Spot + ZeRO-3 + checkpointing | 70% |

---

## Key Takeaways

1. **MIG** is the best tool for multi-tenant inference — hardware-level isolation, guaranteed memory.
2. **Spot instances** save 70% for fault-tolerant workloads (training, batch inference).
3. **Effective cost = nominal cost / utilization** — 20% utilization means 5× higher effective cost.
4. SM efficiency < 50% → increase batch size; tensor core < 30% → enable BF16/TF32.
5. Combine MIG + AWQ quantization for maximum model density per dollar.
