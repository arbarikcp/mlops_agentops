"""Progressive delivery manifest builders — canary steps, Argo Rollouts, AnalysisTemplate.

Day 75 — canary weight steps, Argo Rollout CRD builder, and AnalysisTemplate
(Prometheus-based AUC / PSI gate) for progressive ML model promotion.

Classes:
  CanaryStep        — one step in a canary rollout (weight + optional pause + analysis)
  RolloutStrategy   — ordered list of canary steps with validation
  AnalysisMetric    — one Prometheus-backed metric in an AnalysisTemplate
  AnalysisTemplate  — Argo Rollouts AnalysisTemplate CRD manifest builder
  ArgoRollout       — Argo Rollouts Rollout CRD manifest builder

See: docs/phase11/day75_progressive_delivery.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── CanaryStep ────────────────────────────────────────────────────────────────

@dataclass
class CanaryStep:
    """One step in a canary rollout sequence.

    Attributes:
        weight:            Traffic percentage for this step (1–100).
        pause_minutes:     How long to pause after reaching this weight (0 = no pause).
        analysis_template: AnalysisTemplate name to run at this step (empty = skip).
    """

    weight: int
    pause_minutes: int = 0
    analysis_template: str = ""

    def __post_init__(self) -> None:
        if not (1 <= self.weight <= 100):
            raise ValueError(f"CanaryStep.weight must be 1–100, got {self.weight}")
        if self.pause_minutes < 0:
            raise ValueError("CanaryStep.pause_minutes must be >= 0")

    def to_dict(self) -> list[dict[str, Any]]:
        """Return list of Argo Rollouts step dicts for this step."""
        steps: list[dict[str, Any]] = [{"setWeight": self.weight}]
        if self.pause_minutes > 0:
            steps.append({"pause": {"duration": f"{self.pause_minutes}m"}})
        if self.analysis_template:
            steps.append({"analysis": {"templates": [{"templateName": self.analysis_template}]}})
        return steps


# ── RolloutStrategy ───────────────────────────────────────────────────────────

@dataclass
class RolloutStrategy:
    """Ordered list of canary steps forming a full rollout strategy.

    Attributes:
        steps:           Ordered CanaryStep objects (must end with weight=100).
        max_surge:       Extra pods during rollout.
        max_unavailable: Max pods unavailable during rollout.
    """

    steps: list[CanaryStep] = field(default_factory=list)
    max_surge: int = 1
    max_unavailable: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.steps:
            raise ValueError("RolloutStrategy.steps cannot be empty")
        if self.steps[-1].weight != 100:
            raise ValueError("Last CanaryStep must have weight=100 (full promotion)")
        if self.max_surge < 0:
            raise ValueError("max_surge must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        all_steps: list[dict[str, Any]] = []
        for step in self.steps:
            all_steps.extend(step.to_dict())
        return {
            "canary": {
                "steps": all_steps,
                "canaryMetadata": {"labels": {"model-variant": "canary"}},
                "stableMetadata": {"labels": {"model-variant": "stable"}},
            }
        }


# ── AnalysisMetric ────────────────────────────────────────────────────────────

@dataclass
class AnalysisMetric:
    """One Prometheus-backed metric in an AnalysisTemplate.

    Attributes:
        name:               Metric name (slug).
        prometheus_query:   PromQL query string.
        success_condition:  Go expression on `result` (e.g. "result >= 0.78").
        failure_limit:      How many failures before the analysis fails.
        interval_m:         How often to re-evaluate (minutes).
        prometheus_address: Prometheus server URL.
    """

    name: str
    prometheus_query: str
    success_condition: str
    failure_limit: int = 1
    interval_m: int = 5
    prometheus_address: str = "http://prometheus.monitoring.svc:9090"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AnalysisMetric.name cannot be empty")
        if not self.prometheus_query:
            raise ValueError("AnalysisMetric.prometheus_query cannot be empty")
        if not self.success_condition:
            raise ValueError("AnalysisMetric.success_condition cannot be empty")
        if self.failure_limit < 0:
            raise ValueError("failure_limit must be >= 0")
        if self.interval_m < 1:
            raise ValueError("interval_m must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interval": f"{self.interval_m}m",
            "successCondition": self.success_condition,
            "failureLimit": self.failure_limit,
            "provider": {
                "prometheus": {
                    "address": self.prometheus_address,
                    "query": self.prometheus_query,
                }
            },
        }


# ── AnalysisTemplate ──────────────────────────────────────────────────────────

@dataclass
class AnalysisTemplate:
    """Argo Rollouts AnalysisTemplate CRD manifest builder.

    Attributes:
        name:      Template name (referenced in CanaryStep.analysis_template).
        namespace: Kubernetes namespace.
        metrics:   List of AnalysisMetric objects.
    """

    name: str
    namespace: str = "ml-serving"
    metrics: list[AnalysisMetric] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AnalysisTemplate.name cannot be empty")
        if not self.metrics:
            raise ValueError("AnalysisTemplate.metrics cannot be empty")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "AnalysisTemplate",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {"metrics": [m.to_dict() for m in self.metrics]},
        }

    @staticmethod
    def ml_quality_gate(
        namespace: str = "ml-serving",
        min_auc: float = 0.78,
        max_psi: float = 0.2,
    ) -> "AnalysisTemplate":
        """Standard ML quality gate: AUC + PSI checks on canary traffic."""
        return AnalysisTemplate(
            name="ml-quality-gate",
            namespace=namespace,
            metrics=[
                AnalysisMetric(
                    name="model-auc",
                    prometheus_query=(
                        'avg_over_time(ml_model_auc{model_variant="canary"}[10m])'
                    ),
                    success_condition=f"result >= {min_auc}",
                    failure_limit=1,
                    interval_m=5,
                ),
                AnalysisMetric(
                    name="psi-check",
                    prometheus_query=(
                        'ml_prediction_psi_score{model_variant="canary"}'
                    ),
                    success_condition=f"result < {max_psi}",
                    failure_limit=1,
                    interval_m=5,
                ),
            ],
        )


# ── ArgoRollout ───────────────────────────────────────────────────────────────

@dataclass
class ArgoRollout:
    """Argo Rollouts Rollout CRD manifest builder.

    Attributes:
        name:       Rollout name.
        namespace:  Kubernetes namespace.
        image:      Container image (repo:tag).
        replicas:   Desired replica count.
        strategy:   RolloutStrategy defining canary steps.
        port:       Container port.
        app_label:  Value for the `app` label selector.
    """

    name: str
    namespace: str = "ml-serving"
    image: str = ""
    replicas: int = 3
    strategy: RolloutStrategy = field(default_factory=lambda: RolloutStrategy(
        steps=[
            CanaryStep(weight=10, pause_minutes=30, analysis_template="ml-quality-gate"),
            CanaryStep(weight=50, pause_minutes=15),
            CanaryStep(weight=100),
        ]
    ))
    port: int = 8080
    app_label: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ArgoRollout.name cannot be empty")
        if self.replicas < 1:
            raise ValueError("replicas must be >= 1")
        if not self.app_label:
            self.app_label = self.name

    def to_manifest(self) -> dict[str, Any]:
        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Rollout",
            "metadata": {"name": self.name, "namespace": self.namespace},
            "spec": {
                "replicas": self.replicas,
                "strategy": self.strategy.to_dict(),
                "selector": {"matchLabels": {"app": self.app_label}},
                "template": {
                    "metadata": {"labels": {"app": self.app_label}},
                    "spec": {
                        "containers": [{
                            "name": self.name,
                            "image": self.image,
                            "ports": [{"containerPort": self.port}],
                        }]
                    },
                },
            },
        }
