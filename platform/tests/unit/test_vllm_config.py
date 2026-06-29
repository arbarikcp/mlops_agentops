"""Unit tests for platform/llm/vllm_config.py (Day 98)."""

import pytest
from llm.vllm_config import (
    LoRAConfig,
    SamplingParams,
    VLLMBenchmarkResult,
    VLLMEngineConfig,
    VLLMServerConfig,
)


# ── SamplingParams ─────────────────────────────────────────────────────────

class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams()
        assert p.temperature == 1.0
        assert p.top_p == 1.0
        assert p.max_tokens == 256

    def test_greedy_decoding(self):
        p = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=128)
        assert p.temperature == 0.0

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-0.1)

    def test_invalid_top_p(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)

    def test_top_p_exactly_one(self):
        p = SamplingParams(top_p=1.0)
        assert p.top_p == 1.0

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=0)

    def test_to_dict(self):
        p = SamplingParams(temperature=0.7, max_tokens=512, stop=["</s>"])
        d = p.to_dict()
        assert d["temperature"] == 0.7
        assert d["stop"] == ["</s>"]


# ── LoRAConfig ─────────────────────────────────────────────────────────────

class TestLoRAConfig:
    def test_basic(self):
        lora = LoRAConfig(
            lora_id="sql-lora",
            base_model="llama-7b",
            adapter_path="/adapters/sql",
            rank=16,
        )
        assert lora.rank == 16

    def test_power_of_two_ranks(self):
        for rank in [1, 2, 4, 8, 16, 32, 64]:
            lora = LoRAConfig(
                lora_id="x", base_model="y", adapter_path="z", rank=rank
            )
            assert lora.rank == rank

    def test_invalid_rank_not_power_of_two(self):
        with pytest.raises(ValueError, match="rank"):
            LoRAConfig(lora_id="x", base_model="y", adapter_path="z", rank=12)

    def test_invalid_rank_zero(self):
        with pytest.raises(ValueError, match="rank"):
            LoRAConfig(lora_id="x", base_model="y", adapter_path="z", rank=0)

    def test_empty_lora_id_raises(self):
        with pytest.raises(ValueError, match="lora_id"):
            LoRAConfig(lora_id="", base_model="y", adapter_path="z")

    def test_to_dict(self):
        lora = LoRAConfig(lora_id="id", base_model="base", adapter_path="/path", rank=32)
        d = lora.to_dict()
        assert d["rank"] == 32


# ── VLLMEngineConfig ───────────────────────────────────────────────────────

class TestVLLMEngineConfig:
    def test_defaults(self):
        cfg = VLLMEngineConfig(model="meta-llama/Llama-2-7b")
        assert cfg.tensor_parallel_size == 1
        assert cfg.gpu_memory_utilization == 0.9

    def test_total_parallel_size(self):
        cfg = VLLMEngineConfig(
            model="llama-70b",
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
        )
        assert cfg.total_parallel_size() == 8

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            VLLMEngineConfig(model="")

    def test_invalid_tensor_parallel(self):
        with pytest.raises(ValueError, match="tensor_parallel"):
            VLLMEngineConfig(model="m", tensor_parallel_size=0)

    def test_invalid_gpu_util(self):
        with pytest.raises(ValueError, match="gpu_memory_utilization"):
            VLLMEngineConfig(model="m", gpu_memory_utilization=0.0)

    def test_to_dict(self):
        cfg = VLLMEngineConfig(model="mistral-7b", tensor_parallel_size=2)
        d = cfg.to_dict()
        assert d["model"] == "mistral-7b"
        assert d["tensor_parallel_size"] == 2


# ── VLLMServerConfig ───────────────────────────────────────────────────────

class TestVLLMServerConfig:
    def _engine(self):
        return VLLMEngineConfig(model="meta-llama/Llama-2-7b", tensor_parallel_size=1)

    def test_to_launch_args_basic(self):
        cfg = VLLMServerConfig(engine=self._engine())
        args = cfg.to_launch_args()
        assert "python" in args
        assert "--model" in args
        assert "meta-llama/Llama-2-7b" in args
        assert "--port" in args
        assert "8000" in args

    def test_to_launch_args_with_api_key(self):
        cfg = VLLMServerConfig(engine=self._engine(), api_key="secret")
        args = cfg.to_launch_args()
        assert "--api-key" in args
        assert "secret" in args

    def test_to_launch_args_with_lora(self):
        lora = LoRAConfig(
            lora_id="sql-lora", base_model="llama-7b", adapter_path="/adapters/sql"
        )
        cfg = VLLMServerConfig(
            engine=self._engine(),
            enable_lora=True,
            lora_modules=[lora],
        )
        args = cfg.to_launch_args()
        assert "--enable-lora" in args
        assert "--lora-modules" in args

    def test_invalid_port(self):
        with pytest.raises(ValueError, match="port"):
            VLLMServerConfig(engine=self._engine(), port=0)

    def test_invalid_port_too_large(self):
        with pytest.raises(ValueError, match="port"):
            VLLMServerConfig(engine=self._engine(), port=70000)

    def test_to_dict(self):
        cfg = VLLMServerConfig(engine=self._engine())
        d = cfg.to_dict()
        assert "engine" in d
        assert d["port"] == 8000


# ── VLLMBenchmarkResult ────────────────────────────────────────────────────

class TestVLLMBenchmarkResult:
    def test_tokens_per_second(self):
        r = VLLMBenchmarkResult(
            model="llama-7b",
            num_prompts=100,
            total_time_s=10.0,
            throughput_req_per_s=20.0,
            mean_latency_ms=50.0,
            p99_latency_ms=200.0,
        )
        assert r.tokens_per_second(256) == pytest.approx(20.0 * 256)

    def test_invalid_num_prompts(self):
        with pytest.raises(ValueError, match="num_prompts"):
            VLLMBenchmarkResult(
                model="m",
                num_prompts=0,
                total_time_s=1.0,
                throughput_req_per_s=1.0,
                mean_latency_ms=1.0,
                p99_latency_ms=1.0,
            )

    def test_to_dict(self):
        r = VLLMBenchmarkResult(
            model="gpt2",
            num_prompts=50,
            total_time_s=5.0,
            throughput_req_per_s=10.0,
            mean_latency_ms=100.0,
            p99_latency_ms=300.0,
        )
        d = r.to_dict()
        assert d["model"] == "gpt2"
        assert d["throughput_req_per_s"] == 10.0
