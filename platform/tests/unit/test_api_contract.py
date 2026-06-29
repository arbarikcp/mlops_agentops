"""Tests for serving/api_contract.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from serving.api_contract import (
    ApiContractChecker,
    ApiContractVersion,
    CompatibilityReport,
    FieldSchema,
    RollbackPlan,
    build_v1_contract,
    build_v2_contract,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def checker() -> ApiContractChecker:
    return ApiContractChecker()


def _field(name: str, type_: str = "float", required: bool = True, nullable: bool = False) -> FieldSchema:
    return FieldSchema(name=name, type=type_, required=required, nullable=nullable)


def _contract(version: str, req_fields=None, resp_fields=None) -> ApiContractVersion:
    return ApiContractVersion(
        version=version,
        request_fields=req_fields or [],
        response_fields=resp_fields or [],
    )


# ── FieldSchema ───────────────────────────────────────────────────────────────

class TestFieldSchema:
    def test_to_dict_roundtrip(self) -> None:
        f = FieldSchema("applicant_id", "int", required=True, nullable=False, description="ID")
        restored = FieldSchema.from_dict(f.to_dict())
        assert restored == f

    def test_required_defaults_true(self) -> None:
        f = FieldSchema("x", "float")
        assert f.required is True

    def test_nullable_defaults_false(self) -> None:
        f = FieldSchema("x", "float")
        assert f.nullable is False


# ── ApiContractVersion ────────────────────────────────────────────────────────

class TestApiContractVersion:
    def test_to_dict_roundtrip(self) -> None:
        cv = ApiContractVersion(
            version="v1",
            request_fields=[_field("applicant_id", "int")],
            response_fields=[_field("score", "float")],
        )
        restored = ApiContractVersion.from_dict(cv.to_dict())
        assert restored.version == "v1"
        assert len(restored.request_fields) == 1

    def test_save_and_load(self, tmp_path) -> None:
        cv = build_v1_contract()
        path = tmp_path / "contract_v1.json"
        cv.save(path)
        loaded = ApiContractVersion.load(path)
        assert loaded.version == "v1"
        assert len(loaded.request_fields) == len(cv.request_fields)

    def test_deprecated_flag(self) -> None:
        cv = ApiContractVersion(version="v0", deprecated=True, sunset_date="2026-01-01")
        assert cv.deprecated is True


# ── Compatibility: compatible changes ─────────────────────────────────────────

class TestCompatibleChanges:
    def test_identical_schemas_compatible(self, checker) -> None:
        v1 = _contract("v1", req_fields=[_field("a")])
        report = checker.check_compatibility(v1, v1)
        assert report.is_backward_compatible

    def test_added_optional_field_compatible(self, checker) -> None:
        v1 = _contract("v1", req_fields=[_field("a")])
        v2 = _contract("v2", req_fields=[_field("a"), _field("b", required=False)])
        report = checker.check_compatibility(v1, v2)
        assert report.is_backward_compatible
        assert any("added optional field" in c for c in report.compatible_changes)

    def test_required_to_optional_compatible(self, checker) -> None:
        v1 = _contract("v1", req_fields=[_field("a", required=True)])
        v2 = _contract("v2", req_fields=[_field("a", required=False)])
        report = checker.check_compatibility(v1, v2)
        assert report.is_backward_compatible

    def test_nullable_relaxation_compatible(self, checker) -> None:
        v1 = _contract("v1", resp_fields=[_field("score", nullable=False)])
        v2 = _contract("v2", resp_fields=[_field("score", nullable=True)])
        report = checker.check_compatibility(v1, v2)
        assert report.is_backward_compatible


# ── Compatibility: breaking changes ──────────────────────────────────────────

class TestBreakingChanges:
    def test_removed_required_field_breaking(self, checker) -> None:
        v1 = _contract("v1", req_fields=[_field("a"), _field("b")])
        v2 = _contract("v2", req_fields=[_field("a")])
        report = checker.check_compatibility(v1, v2)
        assert not report.is_backward_compatible
        assert any("b" in msg for msg in report.breaking_changes)

    def test_type_change_breaking(self, checker) -> None:
        v1 = _contract("v1", resp_fields=[_field("score", "float")])
        v2 = _contract("v2", resp_fields=[_field("score", "int")])
        report = checker.check_compatibility(v1, v2)
        assert not report.is_backward_compatible
        assert any("type changed" in msg for msg in report.breaking_changes)

    def test_optional_to_required_breaking(self, checker) -> None:
        v1 = _contract("v1", req_fields=[_field("a", required=False)])
        v2 = _contract("v2", req_fields=[_field("a", required=True)])
        report = checker.check_compatibility(v1, v2)
        assert not report.is_backward_compatible

    def test_nullable_to_nonnullable_breaking(self, checker) -> None:
        v1 = _contract("v1", resp_fields=[_field("x", nullable=True)])
        v2 = _contract("v2", resp_fields=[_field("x", nullable=False)])
        report = checker.check_compatibility(v1, v2)
        assert not report.is_backward_compatible

    def test_adding_required_field_breaking(self, checker) -> None:
        v1 = _contract("v1", req_fields=[_field("a")])
        v2 = _contract("v2", req_fields=[_field("a"), _field("b", required=True)])
        report = checker.check_compatibility(v1, v2)
        assert not report.is_backward_compatible


# ── Compatibility: warnings ───────────────────────────────────────────────────

class TestCompatibilityWarnings:
    def test_removed_optional_field_warning(self, checker) -> None:
        v1 = _contract("v1", resp_fields=[_field("explanation", required=False)])
        v2 = _contract("v2", resp_fields=[])
        report = checker.check_compatibility(v1, v2)
        assert report.is_backward_compatible  # not a BREAKING change
        assert any("removed optional field" in w for w in report.warnings)

    def test_compatible_report_has_no_breaking(self, checker) -> None:
        v1 = _contract("v1", req_fields=[_field("a")])
        v2 = _contract("v2", req_fields=[_field("a"), _field("b", required=False)])
        report = checker.check_compatibility(v1, v2)
        assert report.breaking_changes == []


# ── v1 → v2 built-in contracts ────────────────────────────────────────────────

class TestBuiltinContracts:
    def test_v1_to_v2_is_compatible(self, checker) -> None:
        report = checker.check_compatibility(build_v1_contract(), build_v2_contract())
        assert report.is_backward_compatible

    def test_v1_has_required_fields(self) -> None:
        v1 = build_v1_contract()
        names = {f.name for f in v1.request_fields}
        assert "applicant_id" in names
        assert "features" in names

    def test_v2_has_explanation_field(self) -> None:
        v2 = build_v2_contract()
        names = {f.name for f in v2.response_fields}
        assert "explanation" in names


# ── CompatibilityReport.summary ───────────────────────────────────────────────

class TestCompatibilityReportSummary:
    def test_summary_is_string(self, checker) -> None:
        report = checker.check_compatibility(build_v1_contract(), build_v2_contract())
        assert isinstance(report.summary(), str)

    def test_summary_contains_versions(self, checker) -> None:
        report = checker.check_compatibility(build_v1_contract(), build_v2_contract())
        summary = report.summary()
        assert "v1" in summary and "v2" in summary


# ── RollbackPlan ──────────────────────────────────────────────────────────────

class TestRollbackPlan:
    def _plan(self) -> RollbackPlan:
        return RollbackPlan(
            model_version="v2.0",
            previous_stable="v1.2",
            rollback_steps=["kubectl rollout undo deployment/credit-risk"],
            rollback_ttl_minutes=7,
            canary_traffic_pct=10,
        )

    def test_valid_plan_has_no_issues(self) -> None:
        assert self._plan().validate() == []

    def test_empty_steps_issue(self) -> None:
        plan = self._plan()
        plan.rollback_steps = []
        issues = plan.validate()
        assert any("empty" in i for i in issues)

    def test_high_ttl_issue(self) -> None:
        plan = self._plan()
        plan.rollback_ttl_minutes = 20
        issues = plan.validate()
        assert any("ttl" in i.lower() for i in issues)

    def test_self_rollback_issue(self) -> None:
        plan = self._plan()
        plan.previous_stable = plan.model_version
        issues = plan.validate()
        assert any("self" in i.lower() for i in issues)

    def test_save_and_load(self, tmp_path) -> None:
        plan = self._plan()
        path = tmp_path / "rollback.json"
        plan.save(path)
        loaded = RollbackPlan.load(path)
        assert loaded.model_version == "v2.0"
        assert loaded.previous_stable == "v1.2"
        assert loaded.rollback_ttl_minutes == 7

    def test_to_dict_roundtrip(self) -> None:
        plan = self._plan()
        d = plan.to_dict()
        assert d["model_version"] == "v2.0"
        restored = RollbackPlan(**d)
        assert restored.previous_stable == "v1.2"
