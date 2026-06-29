"""ML-specific incident drill definitions and dry-run framework.

Day 72 — three ML-specific incident categories (bad artifact, stale features,
broken retriever). Each incident has structured symptoms, detection signals,
recovery steps, and prevention controls.

Classes:
  IncidentCategory   — enumeration of ML incident types
  MLIncident         — structured incident definition
  IncidentDrillResult — outcome of a drill dry-run
  MLIncidentDrill    — pairs an incident with drill execution

Pre-built incidents:
  bad_artifact_incident()   — bad model promoted to production
  stale_features_incident() — materialization lag causing train-serve skew
  broken_retriever_incident() — embedding OOM kills RAG retrieval

See: docs/phase10/day72_ml_incidents.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IncidentCategory(str, Enum):
    """Category of ML-specific incident."""
    BAD_ARTIFACT = "bad_artifact"
    STALE_DATA = "stale_data"
    BROKEN_DEPENDENCY = "broken_dependency"


# ── MLIncident ────────────────────────────────────────────────────────────────

@dataclass
class MLIncident:
    """Structured description of one ML system incident.

    Attributes:
        name:                  Short slug (e.g. "bad-artifact-pushed").
        category:              IncidentCategory enum value.
        symptoms:              Observable signals during the incident.
        detection_signal:      Primary metric / alert that fires.
        expected_behavior:     What a well-designed system does automatically.
        actual_behavior:       What happens in the incident without controls.
        recovery_steps:        Ordered list of steps to restore steady state.
        prevention_controls:   Engineering controls to prevent recurrence.
    """

    name: str
    category: IncidentCategory
    symptoms: list[str]
    detection_signal: str
    expected_behavior: str
    actual_behavior: str
    recovery_steps: list[str] = field(default_factory=list)
    prevention_controls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MLIncident.name cannot be empty")
        if not self.symptoms:
            raise ValueError("MLIncident.symptoms must have at least one entry")
        if not self.detection_signal:
            raise ValueError("MLIncident.detection_signal cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "symptoms": self.symptoms,
            "detection_signal": self.detection_signal,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
            "recovery_steps": self.recovery_steps,
            "prevention_controls": self.prevention_controls,
        }


# ── IncidentDrillResult ───────────────────────────────────────────────────────

@dataclass
class IncidentDrillResult:
    """Outcome of one incident drill execution.

    Attributes:
        incident_name:     Matches MLIncident.name.
        detected:          Whether the alert fired as expected.
        detection_time_s:  Seconds from inject to alert.
        recovered:         Whether system returned to steady state.
        recovery_time_s:   Seconds from alert to recovery.
        notes:             Free-form observations.
    """

    incident_name: str
    detected: bool
    detection_time_s: float = 0.0
    recovered: bool = False
    recovery_time_s: float = 0.0
    notes: str = ""

    @property
    def passed(self) -> bool:
        return self.detected and self.recovered

    @property
    def total_time_s(self) -> float:
        return self.detection_time_s + self.recovery_time_s


# ── MLIncidentDrill ───────────────────────────────────────────────────────────

@dataclass
class MLIncidentDrill:
    """Pairs an MLIncident with drill execution logic.

    Attributes:
        incident:          The ML incident to drill.
        alert_rules:       Set of Prometheus alert rule names expected to fire.
        runbook_path:      Path to the runbook document.
    """

    incident: MLIncident
    alert_rules: list[str] = field(default_factory=list)
    runbook_path: str = ""

    def run_dry(self) -> IncidentDrillResult:
        """Validate drill structure without real execution.

        Checks:
          - incident has recovery steps
          - incident has prevention controls
          - at least one alert rule is defined
        """
        issues: list[str] = []

        if not self.incident.recovery_steps:
            issues.append("no recovery_steps defined")
        if not self.incident.prevention_controls:
            issues.append("no prevention_controls defined")
        if not self.alert_rules:
            issues.append("no alert_rules wired — detection coverage unknown")
        if not self.runbook_path:
            issues.append("no runbook_path set — on-call has no runbook")

        passed = len(issues) == 0
        return IncidentDrillResult(
            incident_name=self.incident.name,
            detected=passed,
            recovered=passed,
            notes="; ".join(issues) if issues else "dry-run OK — drill structure valid",
        )

    def check_alert_coverage(self) -> bool:
        """Return True if at least one alert rule is defined for this incident."""
        return len(self.alert_rules) > 0


# ── Pre-built ML incident definitions ─────────────────────────────────────────

def bad_artifact_incident() -> MLIncident:
    """Bad model pushed to production registry."""
    return MLIncident(
        name="bad-artifact-pushed",
        category=IncidentCategory.BAD_ARTIFACT,
        symptoms=[
            "Prediction PSI (population stability index) spike > 0.2",
            "AUC on shadow traffic drops vs baseline",
            "Approval rate changes significantly in 15-min window",
        ],
        detection_signal="model_prediction_psi_score > 0.2 for 5m",
        expected_behavior=(
            "AUC guard in CI blocks promotion; if bypassed, PSI alert fires "
            "within 15 min; canary limits blast radius to 10% traffic"
        ),
        actual_behavior=(
            "Without CI gate: bad model promoted; serving full traffic; "
            "PSI spikes after data drift; no automatic rollback"
        ),
        recovery_steps=[
            "Identify last good model version: mlflow models search-model-versions",
            "Roll back alias: mlflow models set-alias --alias production --version <prev>",
            "Verify KServe downloads previous version",
            "Confirm PSI drops below 0.1 in Grafana",
            "Open postmortem; add AUC gate check to CI pipeline",
        ],
        prevention_controls=[
            "AUCGuard in CI pipeline blocks promotion if regression > 1%",
            "DataContractChecker.check_label_dist() before training",
            "10% canary for 30 min before full promotion",
            "Automatic rollback trigger if PSI > 0.25 for 10 min",
        ],
    )


def stale_features_incident() -> MLIncident:
    """Materialization failure causes stale features in serving."""
    return MLIncident(
        name="stale-features",
        category=IncidentCategory.STALE_DATA,
        symptoms=[
            "feature_freshness_lag_s > 3600 in Prometheus",
            "Prediction confidence drops (scores cluster near 0.5)",
            "Dagster materialization job shows FAILURE status",
        ],
        detection_signal="feature_freshness_lag_s > 3600 for 10m",
        expected_behavior=(
            "Dagster sensor fires Slack alert on materialization failure; "
            "feature store freshness metric checked before serving; "
            "API returns X-Feature-Age-Seconds header"
        ),
        actual_behavior=(
            "Materialization fails silently (no alert); feature store returns "
            "6-hour-old values; model makes decisions on stale account state"
        ),
        recovery_steps=[
            "Check Dagster: dagster job status --job materialize_credit_features",
            "Re-trigger materialization: dagster job launch ...",
            "Verify Redis freshness: redis-cli hgetall <entity_key>",
            "Confirm feature_freshness_lag_s drops below 60s",
        ],
        prevention_controls=[
            "Dagster sensor: materialization failure → immediate Slack alert",
            "FeatureMonitor.check_freshness() on every prediction batch",
            "Hard cap: if feature age > 2h, return StaleFeatureError",
            "Prometheus alert: feature_freshness_lag_s > 3600 for 10m",
        ],
    )


def broken_retriever_incident() -> MLIncident:
    """Embedding model OOM kills RAG retrieval context."""
    return MLIncident(
        name="broken-retriever",
        category=IncidentCategory.BROKEN_DEPENDENCY,
        symptoms=[
            "retrieval_context_size = 0 for > 3 min",
            "retrieval_empty_rate > 5%",
            "kubectl events show OOMKilled for embedding-sidecar",
            "LLM responses contain no citations",
        ],
        detection_signal="retrieval_empty_rate > 0.05 for 5m",
        expected_behavior=(
            "RAG service detects empty context and returns null answer with "
            "reason=retrieval_unavailable; liveness probe restarts OOM pod; "
            "alert fires to on-call"
        ),
        actual_behavior=(
            "OOM pod crashes; retrieval returns []; LLM generates hallucinated "
            "policy text with no grounding; no immediate user-visible error"
        ),
        recovery_steps=[
            "kubectl get events -n llm-serving --field-selector reason=OOMKilling",
            "Increase memory: kubectl set resources deployment/embedding-sidecar --limits=memory=4Gi",
            "Verify pod healthy: kubectl rollout status deployment/embedding-sidecar",
            "Re-test: curl http://rag-service/retrieve -d '{\"query\": \"test\"}'",
            "Confirm retrieval_context_size returns to > 0",
        ],
        prevention_controls=[
            "Set embedding pod limits.memory = 2x observed peak via DCGM",
            "Liveness probe: POST /embed with probe token every 10s",
            "RAG fallback: empty context → refuse answer (not hallucinate)",
            "Alert: retrieval_context_size_avg < 1 for 5 min → page",
            "Load test embedding model under concurrent requests before deploy",
        ],
    )
