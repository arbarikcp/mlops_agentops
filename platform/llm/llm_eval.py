"""
llm_eval.py — LLM Eval I: Offline, Reference-Based/Free, LLM-as-Judge (Day 103)

Covers offline evaluation against a fixed golden dataset, reference-based
metrics (exact-match, ROUGE-L), reference-free metrics, and LLM-as-judge
scoring against a rubric. Provides dataset, judge config, per-example
result, and aggregate report dataclasses.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvalMethod(str, Enum):
    """Evaluation methods for free-text LLM outputs."""

    EXACT_MATCH = "exact_match"
    ROUGE_L = "rouge_l"
    LLM_JUDGE = "llm_judge"
    REFERENCE_FREE = "reference_free"


@dataclass
class EvalExample:
    """A single (input, expected_output, rubric) eval example."""

    input: str
    expected_output: str = ""
    rubric: str = ""

    def __post_init__(self) -> None:
        if not self.input:
            raise ValueError("input must be non-empty")

    def to_dict(self) -> dict:
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "rubric": self.rubric,
        }


@dataclass
class EvalDataset:
    """A golden dataset of eval examples."""

    name: str
    examples: list[EvalExample]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.examples:
            raise ValueError("examples must be non-empty")

    def size(self) -> int:
        return len(self.examples)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "examples": [e.to_dict() for e in self.examples],
        }


@dataclass
class JudgeConfig:
    """Configuration for an LLM-as-judge scorer."""

    judge_model: str
    rubric_template: str
    score_range: tuple[int, int] = (1, 5)

    def __post_init__(self) -> None:
        if not self.judge_model:
            raise ValueError("judge_model must be non-empty")
        if not self.rubric_template:
            raise ValueError("rubric_template must be non-empty")
        if self.score_range[0] >= self.score_range[1]:
            raise ValueError("score_range[0] must be < score_range[1]")

    def to_dict(self) -> dict:
        return {
            "judge_model": self.judge_model,
            "rubric_template": self.rubric_template,
            "score_range": list(self.score_range),
        }


@dataclass
class EvalResult:
    """Result of evaluating a single example."""

    example_input: str
    method: EvalMethod
    score: float
    max_score: float = 1.0
    reasoning: str = ""

    def __post_init__(self) -> None:
        if self.max_score <= 0:
            raise ValueError("max_score must be > 0")
        if not (0 <= self.score <= self.max_score):
            raise ValueError("score must be in [0, max_score]")

    def normalized_score(self) -> float:
        return self.score / self.max_score

    def passed(self, threshold: float = 0.7) -> bool:
        return self.normalized_score() >= threshold

    def to_dict(self) -> dict:
        return {
            "example_input": self.example_input,
            "method": self.method.value,
            "score": self.score,
            "max_score": self.max_score,
            "reasoning": self.reasoning,
            "normalized_score": self.normalized_score(),
        }


@dataclass
class EvalReport:
    """Aggregate report over a set of EvalResults."""

    dataset_name: str
    results: list[EvalResult]

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be non-empty")

    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.normalized_score() for r in self.results) / len(self.results)

    def pass_rate(self, threshold: float = 0.7) -> float:
        if not self.results:
            return 0.0
        passed = sum(1 for r in self.results if r.passed(threshold))
        return passed / len(self.results)

    def worst_examples(self, n: int = 5) -> list[EvalResult]:
        return sorted(self.results, key=lambda r: r.normalized_score())[:n]

    def to_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "results": [r.to_dict() for r in self.results],
            "mean_score": self.mean_score(),
            "pass_rate": self.pass_rate(),
        }
