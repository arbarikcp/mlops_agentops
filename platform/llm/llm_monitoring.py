"""
llm_monitoring.py — LLM Monitoring in Prod: Quality/Hallucination Drift,
Online Eval, Full-Traffic Economics (Day 107)

Covers rolling-window quality drift detection, hallucination drift
monitoring (faithfulness trending down), online eval sampling strategies
(can't eval 100% of prod traffic), and the cost economics of sampled vs
full-traffic evaluation.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class QualityDriftWindow:
    """Rolling window of recent eval scores compared against a historical baseline."""

    metric_name: str
    historical_mean: float
    recent_scores: list[float]
    drift_threshold: float = 0.1

    def __post_init__(self) -> None:
        if not self.metric_name:
            raise ValueError("metric_name must be non-empty")
        if not self.recent_scores:
            raise ValueError("recent_scores must be non-empty")
        if self.historical_mean <= 0:
            raise ValueError("historical_mean must be > 0")
        if not (0 < self.drift_threshold < 1):
            raise ValueError("drift_threshold must be in (0, 1)")

    def recent_mean(self) -> float:
        return sum(self.recent_scores) / len(self.recent_scores)

    def has_drifted(self) -> bool:
        return (
            self.historical_mean - self.recent_mean()
        ) / self.historical_mean > self.drift_threshold

    def drift_magnitude(self) -> float:
        return (self.historical_mean - self.recent_mean()) / self.historical_mean

    def to_dict(self) -> dict:
        return {
            "metric_name": self.metric_name,
            "historical_mean": self.historical_mean,
            "recent_mean": self.recent_mean(),
            "drift_threshold": self.drift_threshold,
            "has_drifted": self.has_drifted(),
            "drift_magnitude": self.drift_magnitude(),
        }


@dataclass
class HallucinationDriftMonitor:
    """Monitors faithfulness drift specifically and raises hallucination alerts."""

    faithfulness_window: QualityDriftWindow
    alert_threshold: float = 0.15

    def __post_init__(self) -> None:
        if self.alert_threshold <= 0:
            raise ValueError("alert_threshold must be > 0")

    def is_alerting(self) -> bool:
        return (
            self.faithfulness_window.has_drifted()
            and self.faithfulness_window.drift_magnitude() > self.alert_threshold
        )

    def to_dict(self) -> dict:
        return {
            "faithfulness_window": self.faithfulness_window.to_dict(),
            "alert_threshold": self.alert_threshold,
            "is_alerting": self.is_alerting(),
        }


class SamplingStrategy(str, Enum):
    """Strategies for sampling production traffic for online eval."""

    FIXED_RATE = "fixed_rate"
    EVERY_NTH = "every_nth"
    ADAPTIVE = "adaptive"


@dataclass
class OnlineEvalSampler:
    """Decides which production requests get sampled for online eval."""

    strategy: SamplingStrategy
    sample_rate: float = 0.05
    every_nth: int = 20

    def __post_init__(self) -> None:
        if not (0 < self.sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")
        if self.every_nth < 1:
            raise ValueError("every_nth must be >= 1")

    def should_sample(self, request_index: int) -> bool:
        if self.strategy == SamplingStrategy.EVERY_NTH:
            return request_index % self.every_nth == 0
        # FIXED_RATE and ADAPTIVE both use hash-based fixed-rate sampling
        return hash(request_index) % 100 < self.sample_rate * 100

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy.value,
            "sample_rate": self.sample_rate,
            "every_nth": self.every_nth,
        }


@dataclass
class EvalEconomics:
    """Cost economics of online eval sampling vs full-traffic evaluation."""

    total_requests_per_day: int
    sample_rate: float
    cost_per_eval_usd: float

    def __post_init__(self) -> None:
        if self.total_requests_per_day < 0:
            raise ValueError("total_requests_per_day must be >= 0")
        if not (0 < self.sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")
        if self.cost_per_eval_usd < 0:
            raise ValueError("cost_per_eval_usd must be >= 0")

    def daily_eval_count(self) -> int:
        return int(self.total_requests_per_day * self.sample_rate)

    def daily_cost_usd(self) -> float:
        return self.daily_eval_count() * self.cost_per_eval_usd

    def full_traffic_cost_usd(self) -> float:
        return self.total_requests_per_day * self.cost_per_eval_usd

    def savings_usd(self) -> float:
        return self.full_traffic_cost_usd() - self.daily_cost_usd()

    def to_dict(self) -> dict:
        return {
            "total_requests_per_day": self.total_requests_per_day,
            "sample_rate": self.sample_rate,
            "cost_per_eval_usd": self.cost_per_eval_usd,
            "daily_eval_count": self.daily_eval_count(),
            "daily_cost_usd": self.daily_cost_usd(),
            "full_traffic_cost_usd": self.full_traffic_cost_usd(),
            "savings_usd": self.savings_usd(),
        }
