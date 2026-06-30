"""
finetuning_ops.py — Fine-Tuning Ops: LoRA/QLoRA, Dataset Versioning,
Eval-Gated Promotion (Day 105)

Covers LoRA (low-rank adapter training), QLoRA (LoRA + 4-bit quantized
base model), fine-tuning dataset versioning, and eval-gated promotion
(a fine-tuned model only ships if it beats baseline on held-out eval).

NOTE: the LoRA dataclass here is named `LoRAFinetuneConfig`, NOT
`LoRAConfig` — that name is already used in vllm_config.py for a
different purpose (multi-LoRA serving, not fine-tuning).
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class FineTuneMethod(str, Enum):
    """Fine-tuning strategy."""

    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"


@dataclass
class LoRAFinetuneConfig:
    """LoRA adapter hyperparameters for fine-tuning (rank/alpha/target modules)."""

    rank: int = 16
    alpha: int = 32
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.rank <= 0 or not _is_power_of_two(self.rank):
            raise ValueError(
                f"rank must be a positive power of 2, got {self.rank}"
            )
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if not (0 <= self.dropout < 1):
            raise ValueError("dropout must be in [0, 1)")

    def scaling_factor(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "target_modules": self.target_modules,
            "dropout": self.dropout,
            "scaling_factor": self.scaling_factor(),
        }


@dataclass
class QLoRAConfig:
    """LoRA combined with a quantized frozen base model."""

    lora: LoRAFinetuneConfig
    quant_bits: int = 4
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.quant_bits not in (4, 8):
            raise ValueError("quant_bits must be 4 or 8")

    def to_dict(self) -> dict:
        return {
            "lora": self.lora.to_dict(),
            "quant_bits": self.quant_bits,
            "compute_dtype": self.compute_dtype,
        }


@dataclass
class FineTuneDatasetVersion:
    """A versioned fine-tuning dataset."""

    name: str
    version: str
    num_examples: int
    source_uri: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if not self.source_uri:
            raise ValueError("source_uri must be non-empty")
        if self.num_examples <= 0:
            raise ValueError("num_examples must be > 0")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "num_examples": self.num_examples,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
        }


@dataclass
class FineTuneJob:
    """A fine-tuning job specification."""

    base_model: str
    dataset: FineTuneDatasetVersion
    method: FineTuneMethod
    lora_config: LoRAFinetuneConfig | None = None
    num_epochs: int = 3
    learning_rate: float = 2e-4

    def __post_init__(self) -> None:
        if not self.base_model:
            raise ValueError("base_model must be non-empty")
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.method != FineTuneMethod.FULL and self.lora_config is None:
            raise ValueError(
                "lora_config is required when method is LORA or QLORA"
            )

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "dataset": self.dataset.to_dict(),
            "method": self.method.value,
            "lora_config": self.lora_config.to_dict()
            if self.lora_config
            else None,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
        }


@dataclass
class EvalGatedPromotion:
    """Gate: only promote a fine-tuned model if it beats baseline by tolerance."""

    candidate_score: float
    baseline_score: float
    min_improvement: float = 0.0

    def __post_init__(self) -> None:
        if not (0 <= self.candidate_score <= 1):
            raise ValueError("candidate_score must be in [0, 1]")
        if not (0 <= self.baseline_score <= 1):
            raise ValueError("baseline_score must be in [0, 1]")

    def improvement(self) -> float:
        return self.candidate_score - self.baseline_score

    def should_promote(self) -> bool:
        return self.improvement() >= self.min_improvement

    def to_dict(self) -> dict:
        return {
            "candidate_score": self.candidate_score,
            "baseline_score": self.baseline_score,
            "min_improvement": self.min_improvement,
            "improvement": self.improvement(),
            "should_promote": self.should_promote(),
        }
