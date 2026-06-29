"""
llm/ — Phase 13: Scaling & Inference Optimization (Days 91–99)

Modules
-------
distributed    (Day 91) — Data/model/pipeline/tensor parallelism; DDP, FSDP, ZeRO
ray_train      (Day 92) — Ray Train multi-GPU job configuration
training_opt   (Day 93) — Mixed precision, gradient checkpointing, data loading
inference_opt  (Day 94) — KV cache, PagedAttention, continuous batching
quantization   (Day 95) — PTQ/QAT, GPTQ/AWQ, knowledge distillation
compilation    (Day 96) — torch.compile, ONNX Runtime, TensorRT-LLM
gpu_cost       (Day 97) — MIG, spot pricing, GPU cost & utilization
vllm_config    (Day 98) — vLLM engine/server config, LoRA, sampling, benchmarks
vllm_k8s       (Day 99) — vLLM on Kubernetes, HPA, capacity planning, PodMonitor
"""
