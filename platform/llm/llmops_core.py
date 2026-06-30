"""
llmops_core.py — LLMOps vs MLOps Foundations (Day 100)

Covers prompts-as-artifacts (versioned, content-hashed), non-determinism
tracking (temperature/seed config), cost-as-metric (token cost is a
first-class production metric), and a structured comparison of LLMOps
vs classical MLOps across key operational dimensions.
No external SDK imports — pure Python dataclasses + stdlib hashlib.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class ArtifactType(str, Enum):
    """Kinds of artifacts that must be versioned in an LLMOps system."""

    PROMPT = "prompt"
    MODEL_WEIGHTS = "model_weights"
    EVAL_DATASET = "eval_dataset"
    SYSTEM_CONFIG = "system_config"


@dataclass
class PromptArtifact:
    """A versioned prompt, treated as a first-class code artifact."""

    content: str
    version: str
    artifact_type: ArtifactType = ArtifactType.PROMPT
    author: str = ""
    content_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("content must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        self.content_hash = hashlib.sha256(
            self.content.encode()
        ).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "version": self.version,
            "artifact_type": self.artifact_type.value,
            "author": self.author,
            "content_hash": self.content_hash,
        }


@dataclass
class NonDeterminismConfig:
    """Sampling configuration that governs LLM output non-determinism."""

    temperature: float
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be >= 0, got {self.temperature}"
            )
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")

    def is_deterministic(self) -> bool:
        return self.temperature == 0 and self.seed is not None

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }


@dataclass
class CostMetric:
    """Per-request token cost, tracked as a first-class production metric."""

    prompt_tokens: int
    completion_tokens: int
    price_per_1k_prompt: float
    price_per_1k_completion: float

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens must be >= 0")
        if self.price_per_1k_prompt < 0:
            raise ValueError("price_per_1k_prompt must be >= 0")
        if self.price_per_1k_completion < 0:
            raise ValueError("price_per_1k_completion must be >= 0")

    def total_cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1000.0 * self.price_per_1k_prompt
            + self.completion_tokens / 1000.0 * self.price_per_1k_completion
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "price_per_1k_prompt": self.price_per_1k_prompt,
            "price_per_1k_completion": self.price_per_1k_completion,
            "total_cost_usd": self.total_cost_usd(),
        }


class LLMOpsVsMLOps:
    """Static comparison of LLMOps vs classical MLOps operational dimensions."""

    @staticmethod
    def compare() -> dict[str, dict]:
        return {
            "artifact_versioning": {
                "mlops": "Model weights + training data versioned (DVC/MLflow)",
                "llmops": "Prompts ALSO versioned as code artifacts (content hash, PR review)",
            },
            "determinism": {
                "mlops": "Same input + model version -> same output (deterministic inference)",
                "llmops": "Same input may yield different outputs (temperature/sampling); reproducibility requires tracking seed+temperature",
            },
            "primary_cost_driver": {
                "mlops": "Training compute (mostly fixed, amortized) + serving infra",
                "llmops": "Per-request token cost (variable, can spike 10x with bad prompts)",
            },
            "eval_method": {
                "mlops": "Fixed metrics: accuracy, AUC, F1 against labeled test set",
                "llmops": "Reference-based + reference-free + LLM-as-judge; no single ground truth for free text",
            },
            "failure_mode": {
                "mlops": "Accuracy degradation, data drift, infra errors",
                "llmops": "Hallucination, prompt injection, cost blowup, in addition to classical drift/infra failures",
            },
        }
