"""Unit tests for infra.milestone2_gate (Day 90)."""

import pytest

from infra.milestone2_gate import (
    M2GateCheck,
    M2GateReport,
    Milestone2Gate,
    GateDimension,
    CheckStatus,
)


# ── M2GateCheck ───────────────────────────────────────────────────────────────

class TestM2GateCheck:
    def test_empty_check_id_raises(self):
        with pytest.raises(ValueError, match="check_id"):
            M2GateCheck("", GateDimension.SECURITY, "desc")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            M2GateCheck("R-01", GateDimension.SECURITY, "")

    def test_run_pass(self):
        check = M2GateCheck("R-01", GateDimension.REPRODUCIBILITY, "check1")
        check.run(lambda: True, "detail")
        assert check.status == CheckStatus.PASS

    def test_run_fail_required(self):
        check = M2GateCheck("R-01", GateDimension.REPRODUCIBILITY, "check1", required=True)
        check.run(lambda: False)
        assert check.status == CheckStatus.FAIL

    def test_run_warn_optional(self):
        check = M2GateCheck("R-01", GateDimension.REPRODUCIBILITY, "check1", required=False)
        check.run(lambda: False)
        assert check.status == CheckStatus.WARN

    def test_run_fail_on_exception(self):
        check = M2GateCheck("R-01", GateDimension.SECURITY, "check1", required=True)
        check.run(lambda: 1 / 0)
        assert check.status == CheckStatus.FAIL
        assert "Exception" in check.detail

    def test_to_dict_structure(self):
        check = M2GateCheck("SEC-01", GateDimension.SECURITY, "KMS check")
        check.run(lambda: True, "kms_arn=arn:...")
        d = check.to_dict()
        assert d["checkId"] == "SEC-01"
        assert d["dimension"] == "security"
        assert d["status"] == "pass"

    def test_detail_stored(self):
        check = M2GateCheck("M-01", GateDimension.MONITORING, "Monitor active")
        check.run(lambda: True, "monitor_status=Scheduled")
        assert check.detail == "monitor_status=Scheduled"


# ── M2GateReport ─────────────────────────────────────────────────────────────

class TestM2GateReport:
    def test_empty_gate_name_raises(self):
        with pytest.raises(ValueError, match="gate_name"):
            M2GateReport("", "prod")

    def test_empty_environment_raises(self):
        with pytest.raises(ValueError, match="environment"):
            M2GateReport("gate", "")

    def test_all_pass_is_success(self):
        report = M2GateReport("gate", "prod")
        for i in range(3):
            c = M2GateCheck(f"R-0{i}", GateDimension.REPRODUCIBILITY, f"check{i}")
            c.run(lambda: True)
            report.checks.append(c)
        assert report.is_passed is True

    def test_any_fail_is_not_success(self):
        report = M2GateReport("gate", "prod")
        c = M2GateCheck("R-01", GateDimension.REPRODUCIBILITY, "check1")
        c.run(lambda: False)
        report.checks.append(c)
        assert report.is_passed is False

    def test_warn_does_not_fail_gate(self):
        report = M2GateReport("gate", "prod")
        c = M2GateCheck("PORT-02", GateDimension.PORTABILITY, "optional", required=False)
        c.run(lambda: False)
        report.checks.append(c)
        assert report.is_passed is True

    def test_passed_checks_count(self):
        report = M2GateReport("gate", "prod")
        for i in range(5):
            c = M2GateCheck(f"C-{i}", GateDimension.SECURITY, f"c{i}")
            c.run(lambda: True)
            report.checks.append(c)
        assert len(report.passed_checks) == 5

    def test_failed_checks_count(self):
        report = M2GateReport("gate", "prod")
        c = M2GateCheck("SEC-01", GateDimension.SECURITY, "fail")
        c.run(lambda: False)
        report.checks.append(c)
        assert len(report.failed_checks) == 1

    def test_by_dimension_grouping(self):
        report = M2GateReport("gate", "prod")
        for dim in [GateDimension.SECURITY, GateDimension.REPRODUCIBILITY, GateDimension.SECURITY]:
            c = M2GateCheck(f"C-{dim}", dim, "desc")
            c.run(lambda: True)
            report.checks.append(c)
        by_dim = report.by_dimension()
        assert len(by_dim["security"]) == 2
        assert len(by_dim["reproducibility"]) == 1

    def test_dimension_pass_rate(self):
        report = M2GateReport("gate", "prod")
        c1 = M2GateCheck("S-01", GateDimension.SERVING, "c1")
        c1.run(lambda: True)
        c2 = M2GateCheck("S-02", GateDimension.SERVING, "c2")
        c2.run(lambda: False)
        report.checks.extend([c1, c2])
        rate = report.dimension_pass_rate(GateDimension.SERVING)
        assert rate == 0.5

    def test_summary_in_to_dict(self):
        report = M2GateReport("gate", "prod")
        c = M2GateCheck("R-01", GateDimension.REPRODUCIBILITY, "c")
        c.run(lambda: True)
        report.checks.append(c)
        d = report.to_dict()
        assert "summary" in d
        assert d["summary"]["gateStatus"] == "PASSED"

    def test_to_dict_has_dimension_scores(self):
        report = M2GateReport("gate", "prod")
        d = report.to_dict()
        assert "dimensionScores" in d
        assert "reproducibility" in d["dimensionScores"]


# ── Milestone2Gate ────────────────────────────────────────────────────────────

class TestMilestone2Gate:
    def test_empty_environment_raises(self):
        with pytest.raises(ValueError, match="environment"):
            Milestone2Gate("", {"model_package_arn": "arn:..."})

    def test_empty_context_raises(self):
        with pytest.raises(ValueError, match="deployment_context"):
            Milestone2Gate("prod", {})

    def test_dry_run_returns_report(self):
        report = Milestone2Gate.dry_run("prod")
        assert isinstance(report, M2GateReport)

    def test_dry_run_at_least_12_checks(self):
        report = Milestone2Gate.dry_run()
        assert len(report.checks) >= 12

    def test_dry_run_covers_all_6_dimensions(self):
        report = Milestone2Gate.dry_run()
        dimensions_covered = {c.dimension for c in report.checks}
        assert dimensions_covered == set(GateDimension)

    def test_default_context_passes_gate(self):
        ctx = Milestone2Gate.default_context()
        gate = Milestone2Gate("prod", ctx)
        report = gate.run()
        assert report.is_passed is True

    def test_missing_model_package_fails_r01(self):
        ctx = Milestone2Gate.default_context(model_package_arn="")
        gate = Milestone2Gate("prod", ctx)
        report = gate.run()
        r01 = next(c for c in report.checks if c.check_id == "R-01")
        assert r01.status == CheckStatus.FAIL
        assert not report.is_passed

    def test_high_latency_fails_s02(self):
        ctx = Milestone2Gate.default_context(endpoint_p99_ms=350.0)
        gate = Milestone2Gate("prod", ctx)
        report = gate.run()
        s02 = next(c for c in report.checks if c.check_id == "S-02")
        assert s02.status == CheckStatus.FAIL

    def test_endpoint_not_in_service_fails_s01(self):
        ctx = Milestone2Gate.default_context(endpoint_status="Updating")
        gate = Milestone2Gate("prod", ctx)
        report = gate.run()
        s01 = next(c for c in report.checks if c.check_id == "S-01")
        assert s01.status == CheckStatus.FAIL

    def test_pipeline_failed_fails_p01(self):
        ctx = Milestone2Gate.default_context(pipeline_status="Failed")
        gate = Milestone2Gate("prod", ctx)
        report = gate.run()
        p01 = next(c for c in report.checks if c.check_id == "P-01")
        assert p01.status == CheckStatus.FAIL

    def test_no_kms_fails_sec01(self):
        ctx = Milestone2Gate.default_context(kms_key_arn="")
        gate = Milestone2Gate("prod", ctx)
        report = gate.run()
        sec01 = next(c for c in report.checks if c.check_id == "SEC-01")
        assert sec01.status == CheckStatus.FAIL

    def test_low_portability_fails_port01(self):
        ctx = Milestone2Gate.default_context(portability_score=0.3)
        gate = Milestone2Gate("prod", ctx)
        report = gate.run()
        port01 = next(c for c in report.checks if c.check_id == "PORT-01")
        assert port01.status == CheckStatus.FAIL

    def test_port02_is_optional_warn(self):
        ctx = Milestone2Gate.default_context()
        ctx["serving_is_portable"] = False
        gate = Milestone2Gate("prod", ctx)
        report = gate.run()
        port02 = next(c for c in report.checks if c.check_id == "PORT-02")
        assert port02.status == CheckStatus.WARN
        # Gate still passes — PORT-02 is required=False
        assert report.is_passed is True

    def test_to_dict_full_structure(self):
        report = Milestone2Gate.dry_run("staging")
        d = report.to_dict()
        assert d["environment"] == "staging"
        assert d["summary"]["total"] >= 12
        assert len(d["checks"]) == d["summary"]["total"]
