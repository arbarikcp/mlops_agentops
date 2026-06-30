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

Phase 14: LLMOps Core (Days 100–108)
-------------------------------------
llmops_core        (Day 100) — Prompts-as-artifacts, non-determinism config, cost-as-metric
llm_serving         (Day 101) — KServe LLMInferenceService CRD, Ray Serve deployment graphs
prompt_registry     (Day 102) — Prompt versioning, lifecycle status, hash-based A/B testing
llm_eval            (Day 103) — Offline eval: reference-based/free metrics, LLM-as-judge
ragas_eval          (Day 104) — RAGAS: faithfulness, context relevance, answer correctness
finetuning_ops      (Day 105) — LoRA/QLoRA fine-tuning, dataset versioning, eval-gated promotion
llm_observability   (Day 106) — OTel GenAI span attributes, Langfuse/Phoenix/LangSmith comparison
llm_monitoring      (Day 107) — Quality/hallucination drift, online eval sampling, eval economics
llm_gateway         (Day 108) — Model routing, quota enforcement, semantic caching, cost governance
"""
