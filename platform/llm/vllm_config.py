"""
vllm_config.py — vLLM Single-Node Configuration (Day 98)

Covers vLLM engine config, server config, LoRA serving,
sampling parameters, and benchmark result dataclasses.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


@dataclass
class SamplingParams:
    """Sampling parameters for vLLM generation requests."""

    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stop: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}"
            )
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
        }


@dataclass
class LoRAConfig:
    """LoRA adapter configuration for multi-LoRA serving."""

    lora_id: str
    base_model: str
    adapter_path: str
    rank: int = 16

    def __post_init__(self) -> None:
        if not self.lora_id:
            raise ValueError("lora_id must be non-empty")
        if not self.base_model:
            raise ValueError("base_model must be non-empty")
        if not self.adapter_path:
            raise ValueError("adapter_path must be non-empty")
        if self.rank <= 0 or not _is_power_of_two(self.rank):
            raise ValueError(
                f"rank must be > 0 and a power of 2, got {self.rank}"
            )

    def to_dict(self) -> dict:
        return {
            "lora_id": self.lora_id,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "rank": self.rank,
        }


@dataclass
class VLLMEngineConfig:
    """vLLM AsyncLLMEngine configuration."""

    model: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_model_len: int = 4096
    quantization: str = ""

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty")
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}"
            )
        if self.pipeline_parallel_size < 1:
            raise ValueError(
                f"pipeline_parallel_size must be >= 1, "
                f"got {self.pipeline_parallel_size}"
            )
        if not (0 < self.gpu_memory_utilization <= 1.0):
            raise ValueError(
                f"gpu_memory_utilization must be in (0, 1], "
                f"got {self.gpu_memory_utilization}"
            )

    def total_parallel_size(self) -> int:
        """Total parallelism = tensor_parallel * pipeline_parallel."""
        return self.tensor_parallel_size * self.pipeline_parallel_size

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "quantization": self.quantization,
        }


@dataclass
class VLLMServerConfig:
    """vLLM OpenAI-compatible API server configuration."""

    engine: VLLMEngineConfig
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""
    enable_lora: bool = False
    lora_modules: list[LoRAConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"port must be in [1, 65535], got {self.port}")

    def to_launch_args(self) -> list[str]:
        """Build CLI arguments for vllm.entrypoints.openai.api_server."""
        args = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.engine.model,
            "--tensor-parallel-size",
            str(self.engine.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.engine.pipeline_parallel_size),
            "--gpu-memory-utilization",
            str(self.engine.gpu_memory_utilization),
            "--max-num-seqs",
            str(self.engine.max_num_seqs),
            "--max-model-len",
            str(self.engine.max_model_len),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.engine.quantization:
            args += ["--quantization", self.engine.quantization]
        if self.api_key:
            args += ["--api-key", self.api_key]
        if self.enable_lora:
            args.append("--enable-lora")
        for lora in self.lora_modules:
            args += ["--lora-modules", f"{lora.lora_id}={lora.adapter_path}"]
        return args

    def to_dict(self) -> dict:
        return {
            "engine": self.engine.to_dict(),
            "host": self.host,
            "port": self.port,
            "api_key": self.api_key,
            "enable_lora": self.enable_lora,
            "lora_modules": [lm.to_dict() for lm in self.lora_modules],
        }


@dataclass
class VLLMBenchmarkResult:
    """Benchmark result from a vLLM throughput test."""

    model: str
    num_prompts: int
    total_time_s: float
    throughput_req_per_s: float
    mean_latency_ms: float
    p99_latency_ms: float

    def __post_init__(self) -> None:
        if self.num_prompts < 1:
            raise ValueError(f"num_prompts must be >= 1, got {self.num_prompts}")

    def tokens_per_second(self, avg_output_tokens: int) -> float:
        """Total token throughput = req/s * avg_output_tokens."""
        return self.throughput_req_per_s * avg_output_tokens

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "num_prompts": self.num_prompts,
            "total_time_s": self.total_time_s,
            "throughput_req_per_s": self.throughput_req_per_s,
            "mean_latency_ms": self.mean_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
        }
