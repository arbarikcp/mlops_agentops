"""Chaos experiment framework for ML system failure injection.

Day 71 — defines the data model for chaos scenarios, experiments,
and results. Actual injection is via kubectl/docker CLI commands;
this module defines, validates, and dry-runs experiments safely.

Classes:
  FailureType    — enumeration of failure categories
  ChaosScenario  — one failure scenario (inject + recovery commands)
  ChaosResult    — outcome of one experiment dry-run
  ChaosExperiment — pairs a scenario with a steady-state definition

See: docs/phase10/day71_chaos_infra.md
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureType(str, Enum):
    """Category of failure being injected."""
    PROCESS_KILL = "process_kill"
    NETWORK_PARTITION = "network_partition"
    RESOURCE_EXHAUST = "resource_exhaust"
    NODE_DRAIN = "node_drain"
    BAD_ARTIFACT = "bad_artifact"
    STALE_DATA = "stale_data"


# ── ChaosScenario ─────────────────────────────────────────────────────────────

@dataclass
class ChaosScenario:
    """One reproducible failure scenario.

    Attributes:
        name:                Short slug (e.g. "mlflow-down").
        target:              Component being broken (e.g. "mlflow-pod").
        failure_type:        Category from FailureType enum.
        hypothesis:          What we expect the system to do.
        inject_cmd:          Shell command(s) that inject the failure.
        recovery_cmd:        Shell command(s) that restore normal state.
        expected_slo_impact: Human description of expected SLO effect.
        blast_radius:        "low" / "medium" / "high".
    """

    name: str
    target: str
    failure_type: FailureType
    hypothesis: str
    inject_cmd: list[str]
    recovery_cmd: list[str]
    expected_slo_impact: str = ""
    blast_radius: str = "low"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ChaosScenario.name cannot be empty")
        if not self.inject_cmd:
            raise ValueError("ChaosScenario.inject_cmd must have at least one command")
        if not self.recovery_cmd:
            raise ValueError("ChaosScenario.recovery_cmd must have at least one command")
        valid_radii = {"low", "medium", "high"}
        if self.blast_radius not in valid_radii:
            raise ValueError(f"blast_radius must be one of {valid_radii}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target,
            "failure_type": self.failure_type.value,
            "hypothesis": self.hypothesis,
            "inject_cmd": self.inject_cmd,
            "recovery_cmd": self.recovery_cmd,
            "expected_slo_impact": self.expected_slo_impact,
            "blast_radius": self.blast_radius,
        }


# ── ChaosResult ───────────────────────────────────────────────────────────────

@dataclass
class ChaosResult:
    """Outcome of one chaos experiment dry-run.

    Attributes:
        scenario_name:        Matches ChaosScenario.name.
        passed:               True if hypothesis was confirmed.
        hypothesis_confirmed: Same as passed; kept explicit for clarity.
        slo_breached:         True if steady-state metrics were violated.
        recovery_time_s:      Seconds from inject to full recovery.
        notes:                Free-form observations.
    """

    scenario_name: str
    passed: bool
    hypothesis_confirmed: bool
    slo_breached: bool = False
    recovery_time_s: float = 0.0
    notes: str = ""


# ── ChaosExperiment ───────────────────────────────────────────────────────────

@dataclass
class ChaosExperiment:
    """Pairs a ChaosScenario with a steady-state definition.

    Attributes:
        scenario:      The failure scenario to exercise.
        steady_state:  Dict of metric_name → acceptable value/range.
                       Example: {"error_rate_pct": 1.0, "p99_ms": 500}
    """

    scenario: ChaosScenario
    steady_state: dict[str, float] = field(default_factory=dict)

    def run_dry(self) -> ChaosResult:
        """Validate the experiment structure without running real commands.

        In a real environment, this would execute inject_cmd, observe
        metrics, then execute recovery_cmd. In dry-run mode it validates
        that commands are non-empty and returns a structural result.
        """
        errors: list[str] = []

        if not self.scenario.inject_cmd:
            errors.append("inject_cmd is empty")
        if not self.scenario.recovery_cmd:
            errors.append("recovery_cmd is empty")
        for cmd in self.scenario.inject_cmd + self.scenario.recovery_cmd:
            if not cmd.strip():
                errors.append(f"blank command in scenario '{self.scenario.name}'")

        passed = len(errors) == 0
        return ChaosResult(
            scenario_name=self.scenario.name,
            passed=passed,
            hypothesis_confirmed=passed,
            slo_breached=False,
            recovery_time_s=0.0,
            notes="; ".join(errors) if errors else "dry-run OK — commands validated",
        )

    def validate_hypothesis(self, observed: dict[str, float]) -> bool:
        """Check whether observed metrics satisfy the steady-state definition.

        Args:
            observed: Dict of metric_name → observed value.

        Returns:
            True if all observed values are within steady-state limits
            (observed value ≤ steady_state limit for each metric).
        """
        for metric, limit in self.steady_state.items():
            if observed.get(metric, float("inf")) > limit:
                return False
        return True


# ── Pre-built ML infra scenarios ──────────────────────────────────────────────

def mlflow_down_scenario() -> ChaosScenario:
    """MLflow tracking server killed mid-training."""
    return ChaosScenario(
        name="mlflow-down",
        target="mlflow-pod",
        failure_type=FailureType.PROCESS_KILL,
        hypothesis="Training continues and completes; only logging is dropped",
        inject_cmd=["kubectl delete pod -l app=mlflow -n mlops-infra"],
        recovery_cmd=["kubectl rollout status deployment/mlflow -n mlops-infra"],
        expected_slo_impact="Experiment not logged; AUC SLO unaffected",
        blast_radius="low",
    )


def minio_down_scenario() -> ChaosScenario:
    """MinIO artifact store killed during serving."""
    return ChaosScenario(
        name="minio-down",
        target="mlops-minio",
        failure_type=FailureType.PROCESS_KILL,
        hypothesis="Serving continues (model in memory); new pod starts fail at init-container",
        inject_cmd=["docker stop mlops-minio"],
        recovery_cmd=["docker start mlops-minio"],
        expected_slo_impact="Existing pods unaffected; new deployments blocked",
        blast_radius="medium",
    )


def kserve_crashloop_scenario() -> ChaosScenario:
    """KServe model pod put into CrashLoopBackOff via bad image tag."""
    return ChaosScenario(
        name="kserve-crashloop",
        target="credit-risk-predictor",
        failure_type=FailureType.BAD_ARTIFACT,
        hypothesis="Rolling update keeps ≥2 old pods serving; readiness probe blocks broken pods",
        inject_cmd=[
            "helm upgrade credit-risk infra/helm/credit-risk "
            "--set image.tag=nonexistent --namespace ml-serving",
        ],
        recovery_cmd=["helm rollback credit-risk 1 --namespace ml-serving"],
        expected_slo_impact="Error rate stays < 1% due to rolling update maxUnavailable=1",
        blast_radius="medium",
    )


def gpu_node_gone_scenario() -> ChaosScenario:
    """GPU node drained to simulate autoscaler removal."""
    return ChaosScenario(
        name="gpu-node-gone",
        target="gpu-node-1",
        failure_type=FailureType.NODE_DRAIN,
        hypothesis="Training Job enters Pending in Kueue; Karpenter provisions replacement in < 90s",
        inject_cmd=[
            "kubectl cordon gpu-node-1",
            "kubectl drain gpu-node-1 --ignore-daemonsets --delete-emptydir-data",
        ],
        recovery_cmd=["kubectl uncordon gpu-node-1"],
        expected_slo_impact="GPU SLO temporarily degraded; job not lost",
        blast_radius="high",
    )


def queue_backlog_scenario() -> ChaosScenario:
    """Submit 20 GPU jobs when only 8 GPU quota is available."""
    return ChaosScenario(
        name="queue-backlog",
        target="kueue-cluster-queue",
        failure_type=FailureType.RESOURCE_EXHAUST,
        hypothesis="Kueue admits 8 jobs; 12 remain pending; BestEffortFIFO ordering respected",
        inject_cmd=[
            "for i in $(seq 1 20); do "
            "kubectl apply -f platform/infra/k8s/kueue/sample-gpu-job.yaml; done",
        ],
        recovery_cmd=["kubectl delete jobs -l app=chaos-gpu-test -n ml-training"],
        expected_slo_impact="Queue depth alert fires; no job corruption",
        blast_radius="medium",
    )
