"""Tests for ci/chaos/ml_incidents.py — MLIncident, MLIncidentDrill, IncidentDrillResult."""
from __future__ import annotations

import pytest

from ci.chaos.ml_incidents import (
    IncidentCategory,
    IncidentDrillResult,
    MLIncident,
    MLIncidentDrill,
    bad_artifact_incident,
    broken_retriever_incident,
    stale_features_incident,
)


def make_incident(**kwargs) -> MLIncident:
    defaults = dict(
        name="test-incident",
        category=IncidentCategory.BAD_ARTIFACT,
        symptoms=["PSI spike > 0.2"],
        detection_signal="model_prediction_psi_score > 0.2 for 5m",
        expected_behavior="PSI alert fires",
        actual_behavior="model silently wrong",
        recovery_steps=["roll back alias"],
        prevention_controls=["AUC guard in CI"],
    )
    defaults.update(kwargs)
    return MLIncident(**defaults)


# ── IncidentCategory ──────────────────────────────────────────────────────────

class TestIncidentCategory:
    def test_all_values(self) -> None:
        values = {c.value for c in IncidentCategory}
        assert "bad_artifact" in values
        assert "stale_data" in values
        assert "broken_dependency" in values

    def test_is_str_enum(self) -> None:
        assert IncidentCategory.BAD_ARTIFACT == "bad_artifact"


# ── MLIncident ────────────────────────────────────────────────────────────────

class TestMLIncident:
    def test_valid_construction(self) -> None:
        inc = make_incident()
        assert inc.name == "test-incident"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            make_incident(name="")

    def test_empty_symptoms_raises(self) -> None:
        with pytest.raises(ValueError, match="symptoms"):
            make_incident(symptoms=[])

    def test_empty_detection_signal_raises(self) -> None:
        with pytest.raises(ValueError, match="detection_signal"):
            make_incident(detection_signal="")

    def test_to_dict_has_required_keys(self) -> None:
        d = make_incident().to_dict()
        for key in ("name", "category", "symptoms", "detection_signal",
                    "recovery_steps", "prevention_controls"):
            assert key in d

    def test_category_serialized_as_string(self) -> None:
        d = make_incident(category=IncidentCategory.STALE_DATA).to_dict()
        assert d["category"] == "stale_data"


# ── IncidentDrillResult ───────────────────────────────────────────────────────

class TestIncidentDrillResult:
    def test_passed_when_detected_and_recovered(self) -> None:
        r = IncidentDrillResult("test", detected=True, recovered=True)
        assert r.passed is True

    def test_not_passed_when_not_detected(self) -> None:
        r = IncidentDrillResult("test", detected=False, recovered=True)
        assert r.passed is False

    def test_not_passed_when_not_recovered(self) -> None:
        r = IncidentDrillResult("test", detected=True, recovered=False)
        assert r.passed is False

    def test_total_time(self) -> None:
        r = IncidentDrillResult("test", detected=True, recovered=True,
                                detection_time_s=30.0, recovery_time_s=120.0)
        assert r.total_time_s == pytest.approx(150.0)


# ── MLIncidentDrill ───────────────────────────────────────────────────────────

class TestMLIncidentDrill:
    def test_dry_run_passes_with_complete_incident(self) -> None:
        drill = MLIncidentDrill(
            incident=make_incident(),
            alert_rules=["ModelPredictionPSI"],
            runbook_path="docs/runbooks/bad-artifact-pushed.md",
        )
        result = drill.run_dry()
        assert result.passed is True

    def test_dry_run_fails_without_recovery_steps(self) -> None:
        drill = MLIncidentDrill(
            incident=make_incident(recovery_steps=[]),
            alert_rules=["SomeAlert"],
            runbook_path="docs/runbooks/test.md",
        )
        result = drill.run_dry()
        assert result.passed is False
        assert "recovery_steps" in result.notes

    def test_dry_run_fails_without_prevention_controls(self) -> None:
        drill = MLIncidentDrill(
            incident=make_incident(prevention_controls=[]),
            alert_rules=["SomeAlert"],
            runbook_path="docs/runbooks/test.md",
        )
        result = drill.run_dry()
        assert result.passed is False
        assert "prevention_controls" in result.notes

    def test_dry_run_fails_without_alert_rules(self) -> None:
        drill = MLIncidentDrill(
            incident=make_incident(),
            alert_rules=[],
            runbook_path="docs/runbooks/test.md",
        )
        result = drill.run_dry()
        assert result.passed is False
        assert "alert_rules" in result.notes

    def test_dry_run_fails_without_runbook_path(self) -> None:
        drill = MLIncidentDrill(
            incident=make_incident(),
            alert_rules=["SomeAlert"],
            runbook_path="",
        )
        result = drill.run_dry()
        assert result.passed is False
        assert "runbook_path" in result.notes

    def test_check_alert_coverage_true(self) -> None:
        drill = MLIncidentDrill(incident=make_incident(), alert_rules=["Alert1"])
        assert drill.check_alert_coverage() is True

    def test_check_alert_coverage_false(self) -> None:
        drill = MLIncidentDrill(incident=make_incident(), alert_rules=[])
        assert drill.check_alert_coverage() is False


# ── Pre-built incidents ───────────────────────────────────────────────────────

class TestPrebuiltIncidents:
    def test_bad_artifact(self) -> None:
        inc = bad_artifact_incident()
        assert inc.name == "bad-artifact-pushed"
        assert inc.category == IncidentCategory.BAD_ARTIFACT
        assert len(inc.symptoms) >= 2
        assert len(inc.recovery_steps) >= 3
        assert len(inc.prevention_controls) >= 3

    def test_stale_features(self) -> None:
        inc = stale_features_incident()
        assert inc.category == IncidentCategory.STALE_DATA
        assert "freshness" in inc.detection_signal

    def test_broken_retriever(self) -> None:
        inc = broken_retriever_incident()
        assert inc.category == IncidentCategory.BROKEN_DEPENDENCY
        assert "retrieval" in inc.detection_signal

    def test_all_prebuilt_have_recovery_steps(self) -> None:
        for inc in [bad_artifact_incident(), stale_features_incident(), broken_retriever_incident()]:
            assert len(inc.recovery_steps) >= 3, f"{inc.name} needs >= 3 recovery steps"

    def test_all_prebuilt_have_prevention_controls(self) -> None:
        for inc in [bad_artifact_incident(), stale_features_incident(), broken_retriever_incident()]:
            assert len(inc.prevention_controls) >= 3, f"{inc.name} needs prevention controls"
