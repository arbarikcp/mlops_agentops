"""Tests for ci/chaos/game_day.py — Runbook, Postmortem, GameDay, GameDayReport."""
from __future__ import annotations

import pytest

from ci.chaos.chaos_engine import ChaosExperiment, mlflow_down_scenario
from ci.chaos.game_day import (
    ActionItem,
    GameDay,
    GameDayReport,
    Postmortem,
    Runbook,
)
from ci.chaos.ml_incidents import MLIncidentDrill, bad_artifact_incident


# ── Runbook ───────────────────────────────────────────────────────────────────

class TestRunbook:
    def make(self, **kwargs) -> Runbook:
        defaults = dict(
            incident_name="bad-artifact-pushed",
            alert_name="ModelPredictionPSI",
            immediate_steps=["Check Grafana"],
            investigation_steps=["Check MLflow"],
            recovery_steps=["Roll back alias"],
            escalation_criteria=["If PSI stays high > 30m"],
            runbook_path="docs/runbooks/bad-artifact-pushed.md",
        )
        defaults.update(kwargs)
        return Runbook(**defaults)

    def test_empty_incident_name_raises(self) -> None:
        with pytest.raises(ValueError, match="incident_name"):
            self.make(incident_name="")

    def test_empty_alert_name_raises(self) -> None:
        with pytest.raises(ValueError, match="alert_name"):
            self.make(alert_name="")

    def test_complete_runbook(self) -> None:
        r = self.make()
        assert r.is_complete() is True

    def test_missing_immediate_steps(self) -> None:
        r = self.make(immediate_steps=[])
        assert r.is_complete() is False
        assert "immediate_steps" in r.missing_sections()

    def test_missing_recovery_steps(self) -> None:
        r = self.make(recovery_steps=[])
        assert r.is_complete() is False

    def test_missing_escalation_criteria(self) -> None:
        r = self.make(escalation_criteria=[])
        assert r.is_complete() is False
        assert "escalation_criteria" in r.missing_sections()

    def test_no_missing_sections_when_complete(self) -> None:
        assert self.make().missing_sections() == []


# ── ActionItem ────────────────────────────────────────────────────────────────

class TestActionItem:
    def test_to_dict(self) -> None:
        a = ActionItem("Add alert", "ML Eng", "2026-07-06")
        d = a.to_dict()
        assert d["description"] == "Add alert"
        assert d["owner"] == "ML Eng"
        assert d["due_date"] == "2026-07-06"
        assert d["done"] is False

    def test_done_flag(self) -> None:
        a = ActionItem("Deploy fix", "DevOps", "2026-07-10", done=True)
        assert a.to_dict()["done"] is True


# ── Postmortem ────────────────────────────────────────────────────────────────

class TestPostmortem:
    def make(self, **kwargs) -> Postmortem:
        defaults = dict(
            incident_name="bad-artifact-pushed",
            date="2026-06-29",
            summary="A bad model was promoted to production causing PSI spike.",
            root_cause="AUC guard was not enforced in the CI pipeline.",
        )
        defaults.update(kwargs)
        return Postmortem(**defaults)

    def test_empty_incident_name_raises(self) -> None:
        with pytest.raises(ValueError, match="incident_name"):
            self.make(incident_name="")

    def test_empty_summary_raises(self) -> None:
        with pytest.raises(ValueError, match="summary"):
            self.make(summary="")

    def test_blameless(self) -> None:
        pm = self.make()
        assert pm.is_blameless() is True

    def test_blame_detected_in_summary(self) -> None:
        pm = self.make(summary="It was his fault the model was deployed.")
        assert pm.is_blameless() is False

    def test_blame_detected_in_root_cause(self) -> None:
        pm = self.make(root_cause="Their fault — no review.")
        assert pm.is_blameless() is False

    def test_open_action_items(self) -> None:
        pm = self.make()
        pm.action_items = [
            ActionItem("Fix CI gate", "ML", "2026-07-01", done=False),
            ActionItem("Add alert", "Ops", "2026-07-05", done=True),
        ]
        open_items = pm.open_action_items()
        assert len(open_items) == 1
        assert open_items[0].description == "Fix CI gate"

    def test_to_dict(self) -> None:
        pm = self.make()
        d = pm.to_dict()
        assert d["is_blameless"] is True
        assert "action_items" in d


# ── GameDayReport ─────────────────────────────────────────────────────────────

class TestGameDayReport:
    def test_pass_rate_all_pass(self) -> None:
        from ci.chaos.chaos_engine import ChaosResult
        from ci.chaos.ml_incidents import IncidentDrillResult
        report = GameDayReport(
            name="test",
            chaos_results=[ChaosResult("s1", passed=True, hypothesis_confirmed=True)],
            drill_results=[IncidentDrillResult("d1", detected=True, recovered=True)],
        )
        assert report.pass_rate() == pytest.approx(1.0)
        assert report.total_scenarios == 2
        assert report.scenarios_passed == 2

    def test_pass_rate_partial(self) -> None:
        from ci.chaos.chaos_engine import ChaosResult
        report = GameDayReport(
            name="test",
            chaos_results=[
                ChaosResult("s1", passed=True, hypothesis_confirmed=True),
                ChaosResult("s2", passed=False, hypothesis_confirmed=False),
            ],
        )
        assert report.pass_rate() == pytest.approx(0.5)

    def test_slo_breach_count(self) -> None:
        from ci.chaos.chaos_engine import ChaosResult
        report = GameDayReport(
            name="test",
            chaos_results=[
                ChaosResult("s1", passed=True, hypothesis_confirmed=True, slo_breached=True),
                ChaosResult("s2", passed=True, hypothesis_confirmed=True, slo_breached=False),
            ],
        )
        assert report.slo_breaches == 1

    def test_pass_rate_empty(self) -> None:
        report = GameDayReport(name="empty")
        assert report.pass_rate() == 0.0

    def test_summary_string(self) -> None:
        report = GameDayReport(name="Phase 10 Game Day")
        s = report.summary()
        assert "Phase 10 Game Day" in s


# ── GameDay ───────────────────────────────────────────────────────────────────

class TestGameDay:
    def make_runbook(self, name: str = "bad-artifact-pushed") -> Runbook:
        return Runbook(
            incident_name=name,
            alert_name="SomeAlert",
            immediate_steps=["Step 1"],
            recovery_steps=["Recover"],
            escalation_criteria=["Escalate if > 30m"],
        )

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            GameDay(name="", rollback_plan="run helm rollback")

    def test_invalid_environment_raises(self) -> None:
        with pytest.raises(ValueError, match="environment"):
            GameDay(name="gd", environment="dev", rollback_plan="rollback")

    def test_empty_rollback_plan_raises(self) -> None:
        with pytest.raises(ValueError, match="rollback_plan"):
            GameDay(name="gd", rollback_plan="")

    def test_valid_staging(self) -> None:
        gd = GameDay(name="gd", environment="staging", rollback_plan="helm rollback")
        assert gd.environment == "staging"

    def test_valid_production(self) -> None:
        gd = GameDay(name="gd", environment="production", rollback_plan="helm rollback")
        assert gd.environment == "production"

    def test_dry_run_no_scenarios(self) -> None:
        gd = GameDay(name="empty-gd", rollback_plan="run helm rollback")
        report = gd.run_dry()
        assert report.total_scenarios == 0
        assert report.pass_rate() == 0.0

    def test_dry_run_with_passing_experiment(self) -> None:
        gd = GameDay(
            name="test-gd",
            rollback_plan="helm rollback credit-risk 1",
            experiments=[ChaosExperiment(scenario=mlflow_down_scenario())],
        )
        report = gd.run_dry()
        assert report.total_scenarios == 1
        assert report.scenarios_passed == 1

    def test_dry_run_with_passing_drill(self) -> None:
        gd = GameDay(
            name="test-gd",
            rollback_plan="helm rollback",
            drills=[
                MLIncidentDrill(
                    incident=bad_artifact_incident(),
                    alert_rules=["ModelPredictionPSI"],
                    runbook_path="docs/runbooks/bad-artifact-pushed.md",
                )
            ],
        )
        report = gd.run_dry()
        assert report.total_scenarios == 1
        assert report.scenarios_passed == 1

    def test_runbook_gap_detected(self) -> None:
        incomplete_runbook = Runbook(
            incident_name="bad-artifact-pushed",
            alert_name="ModelPSI",
            immediate_steps=[],  # missing
            recovery_steps=["Rollback"],
            escalation_criteria=["Escalate"],
        )
        gd = GameDay(
            name="gd",
            rollback_plan="helm rollback",
            runbooks=[incomplete_runbook],
        )
        report = gd.run_dry()
        assert "bad-artifact-pushed" in report.runbook_gaps

    def test_complete_runbook_not_a_gap(self) -> None:
        gd = GameDay(
            name="gd",
            rollback_plan="helm rollback",
            runbooks=[self.make_runbook()],
        )
        report = gd.run_dry()
        assert report.runbook_gaps == []
