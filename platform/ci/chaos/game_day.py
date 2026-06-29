"""Game day orchestration, runbook validation, and postmortem structure.

Day 73 — game day ties together chaos scenarios into a structured exercise,
validates that runbooks are complete, and provides a postmortem template.

Classes:
  Runbook       — structured runbook with completion validation
  Postmortem    — blameless postmortem with action items
  GameDayReport — outcome of running all game day scenarios
  GameDay       — orchestrates scenarios into a game day exercise

See: docs/phase10/day73_game_day.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ci.chaos.chaos_engine import ChaosExperiment, ChaosResult, ChaosScenario
from ci.chaos.ml_incidents import IncidentDrillResult, MLIncidentDrill


# ── Runbook ───────────────────────────────────────────────────────────────────

@dataclass
class Runbook:
    """Structured runbook for one incident type.

    Attributes:
        incident_name:          Matches MLIncident.name or ChaosScenario.name.
        alert_name:             Prometheus alert that triggers this runbook.
        immediate_steps:        Actions in the first 5 minutes.
        investigation_steps:    Root-cause investigation commands.
        recovery_steps:         Steps to restore steady state.
        escalation_criteria:    When to escalate beyond on-call.
        runbook_path:           Path to the Markdown runbook file.
    """

    incident_name: str
    alert_name: str
    immediate_steps: list[str] = field(default_factory=list)
    investigation_steps: list[str] = field(default_factory=list)
    recovery_steps: list[str] = field(default_factory=list)
    escalation_criteria: list[str] = field(default_factory=list)
    runbook_path: str = ""

    def __post_init__(self) -> None:
        if not self.incident_name:
            raise ValueError("Runbook.incident_name cannot be empty")
        if not self.alert_name:
            raise ValueError("Runbook.alert_name cannot be empty")

    def is_complete(self) -> bool:
        """Return True if all required sections are populated."""
        return (
            bool(self.immediate_steps)
            and bool(self.recovery_steps)
            and bool(self.escalation_criteria)
        )

    def missing_sections(self) -> list[str]:
        """Return list of unpopulated required sections."""
        missing = []
        if not self.immediate_steps:
            missing.append("immediate_steps")
        if not self.recovery_steps:
            missing.append("recovery_steps")
        if not self.escalation_criteria:
            missing.append("escalation_criteria")
        return missing


# ── Postmortem ────────────────────────────────────────────────────────────────

@dataclass
class ActionItem:
    """One action item from a postmortem.

    Attributes:
        description: What to do.
        owner:       Team or person responsible.
        due_date:    ISO date string (e.g. "2026-07-06").
        done:        Completion flag.
    """

    description: str
    owner: str
    due_date: str
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "owner": self.owner,
            "due_date": self.due_date,
            "done": self.done,
        }


@dataclass
class Postmortem:
    """Blameless postmortem for one ML incident.

    Attributes:
        incident_name:        Identifier matching MLIncident.name.
        date:                 ISO date of the incident.
        summary:              One-paragraph narrative.
        timeline:             List of {"time": ..., "event": ...} dicts.
        root_cause:           Technical root cause description.
        contributing_factors: System conditions that made the incident worse.
        action_items:         List of ActionItem objects.
    """

    incident_name: str
    date: str
    summary: str
    timeline: list[dict[str, str]] = field(default_factory=list)
    root_cause: str = ""
    contributing_factors: list[str] = field(default_factory=list)
    action_items: list[ActionItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.incident_name:
            raise ValueError("Postmortem.incident_name cannot be empty")
        if not self.summary:
            raise ValueError("Postmortem.summary cannot be empty")

    def is_blameless(self) -> bool:
        """Return True if summary and root_cause avoid blaming individuals.

        Checks for first-person singular language used in a blaming context.
        """
        blame_phrases = ["his fault", "her fault", "their fault", "you should have"]
        text = (self.summary + " " + self.root_cause).lower()
        return not any(phrase in text for phrase in blame_phrases)

    def open_action_items(self) -> list[ActionItem]:
        return [a for a in self.action_items if not a.done]

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_name": self.incident_name,
            "date": self.date,
            "summary": self.summary,
            "timeline": self.timeline,
            "root_cause": self.root_cause,
            "contributing_factors": self.contributing_factors,
            "action_items": [a.to_dict() for a in self.action_items],
            "is_blameless": self.is_blameless(),
        }


# ── GameDayReport ─────────────────────────────────────────────────────────────

@dataclass
class GameDayReport:
    """Outcome of a complete game day exercise.

    Attributes:
        name:                 Game day name (e.g. "Phase 10 Game Day").
        chaos_results:        Results from ChaosExperiment dry-runs.
        drill_results:        Results from MLIncidentDrill dry-runs.
        runbook_gaps:         Runbook names that failed is_complete().
        action_items:         Aggregate action items from all scenarios.
    """

    name: str
    chaos_results: list[ChaosResult] = field(default_factory=list)
    drill_results: list[IncidentDrillResult] = field(default_factory=list)
    runbook_gaps: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)

    @property
    def total_scenarios(self) -> int:
        return len(self.chaos_results) + len(self.drill_results)

    @property
    def scenarios_passed(self) -> int:
        chaos_passed = sum(1 for r in self.chaos_results if r.passed)
        drill_passed = sum(1 for r in self.drill_results if r.passed)
        return chaos_passed + drill_passed

    @property
    def slo_breaches(self) -> int:
        return sum(1 for r in self.chaos_results if r.slo_breached)

    def pass_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.scenarios_passed / self.total_scenarios

    def summary(self) -> str:
        return (
            f"{self.name}: {self.scenarios_passed}/{self.total_scenarios} passed "
            f"({self.pass_rate():.0%}), {self.slo_breaches} SLO breaches, "
            f"{len(self.runbook_gaps)} runbook gaps"
        )


# ── GameDay ───────────────────────────────────────────────────────────────────

@dataclass
class GameDay:
    """Orchestrates chaos experiments and ML incident drills into one game day.

    Attributes:
        name:           Game day identifier.
        environment:    "staging" or "production".
        experiments:    List of ChaosExperiment objects.
        drills:         List of MLIncidentDrill objects.
        runbooks:       List of Runbook objects (one per incident).
        rollback_plan:  Description of how to abort safely.
    """

    name: str
    environment: str = "staging"
    experiments: list[ChaosExperiment] = field(default_factory=list)
    drills: list[MLIncidentDrill] = field(default_factory=list)
    runbooks: list[Runbook] = field(default_factory=list)
    rollback_plan: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GameDay.name cannot be empty")
        valid_envs = {"staging", "production"}
        if self.environment not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}")
        if not self.rollback_plan:
            raise ValueError("GameDay.rollback_plan cannot be empty — safety first")

    def run_dry(self) -> GameDayReport:
        """Dry-run all experiments and drills; validate all runbooks.

        Returns a GameDayReport with aggregated results.
        """
        chaos_results = [e.run_dry() for e in self.experiments]
        drill_results = [d.run_dry() for d in self.drills]

        runbook_gaps = [
            r.incident_name for r in self.runbooks if not r.is_complete()
        ]

        action_items: list[str] = []
        for result in chaos_results:
            if not result.passed:
                action_items.append(
                    f"Fix experiment '{result.scenario_name}': {result.notes}"
                )
        for result in drill_results:
            if not result.passed:
                action_items.append(
                    f"Fix drill '{result.incident_name}': {result.notes}"
                )
        for gap in runbook_gaps:
            action_items.append(f"Complete runbook for '{gap}'")

        return GameDayReport(
            name=self.name,
            chaos_results=chaos_results,
            drill_results=drill_results,
            runbook_gaps=runbook_gaps,
            action_items=action_items,
        )
