"""Tests for ci/chaos/chaos_engine.py — ChaosScenario, ChaosExperiment, ChaosResult."""
from __future__ import annotations

import pytest

from ci.chaos.chaos_engine import (
    ChaosExperiment,
    ChaosResult,
    ChaosScenario,
    FailureType,
    gpu_node_gone_scenario,
    kserve_crashloop_scenario,
    minio_down_scenario,
    mlflow_down_scenario,
    queue_backlog_scenario,
)


def make_scenario(**kwargs) -> ChaosScenario:
    defaults = dict(
        name="test-scenario",
        target="test-pod",
        failure_type=FailureType.PROCESS_KILL,
        hypothesis="system continues",
        inject_cmd=["kubectl delete pod test"],
        recovery_cmd=["kubectl rollout status deploy/test"],
    )
    defaults.update(kwargs)
    return ChaosScenario(**defaults)


# ── FailureType ────────────────────────────────────────────────────────────────

class TestFailureType:
    def test_all_values_present(self) -> None:
        values = {ft.value for ft in FailureType}
        assert "process_kill" in values
        assert "node_drain" in values
        assert "bad_artifact" in values

    def test_is_str_enum(self) -> None:
        assert FailureType.PROCESS_KILL == "process_kill"


# ── ChaosScenario ─────────────────────────────────────────────────────────────

class TestChaosScenario:
    def test_valid_construction(self) -> None:
        s = make_scenario()
        assert s.name == "test-scenario"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            make_scenario(name="")

    def test_empty_inject_cmd_raises(self) -> None:
        with pytest.raises(ValueError, match="inject_cmd"):
            make_scenario(inject_cmd=[])

    def test_empty_recovery_cmd_raises(self) -> None:
        with pytest.raises(ValueError, match="recovery_cmd"):
            make_scenario(recovery_cmd=[])

    def test_invalid_blast_radius_raises(self) -> None:
        with pytest.raises(ValueError, match="blast_radius"):
            make_scenario(blast_radius="extreme")

    def test_valid_blast_radii(self) -> None:
        for radius in ("low", "medium", "high"):
            s = make_scenario(blast_radius=radius)
            assert s.blast_radius == radius

    def test_to_dict_keys(self) -> None:
        s = make_scenario()
        d = s.to_dict()
        assert "name" in d
        assert "failure_type" in d
        assert "inject_cmd" in d
        assert "recovery_cmd" in d

    def test_to_dict_failure_type_is_string(self) -> None:
        s = make_scenario(failure_type=FailureType.NODE_DRAIN)
        assert s.to_dict()["failure_type"] == "node_drain"

    def test_multiple_inject_commands(self) -> None:
        s = make_scenario(inject_cmd=["kubectl cordon node-1", "kubectl drain node-1"])
        assert len(s.inject_cmd) == 2


# ── ChaosExperiment dry-run ───────────────────────────────────────────────────

class TestChaosExperiment:
    def test_dry_run_valid_passes(self) -> None:
        exp = ChaosExperiment(scenario=make_scenario())
        result = exp.run_dry()
        assert result.passed is True
        assert result.hypothesis_confirmed is True

    def test_dry_run_result_type(self) -> None:
        exp = ChaosExperiment(scenario=make_scenario())
        result = exp.run_dry()
        assert isinstance(result, ChaosResult)

    def test_dry_run_scenario_name_in_result(self) -> None:
        exp = ChaosExperiment(scenario=make_scenario(name="my-scenario"))
        result = exp.run_dry()
        assert result.scenario_name == "my-scenario"

    def test_validate_hypothesis_all_within_limits(self) -> None:
        exp = ChaosExperiment(
            scenario=make_scenario(),
            steady_state={"error_rate_pct": 1.0, "p99_ms": 500.0},
        )
        assert exp.validate_hypothesis({"error_rate_pct": 0.5, "p99_ms": 400.0}) is True

    def test_validate_hypothesis_breach(self) -> None:
        exp = ChaosExperiment(
            scenario=make_scenario(),
            steady_state={"error_rate_pct": 1.0},
        )
        assert exp.validate_hypothesis({"error_rate_pct": 2.0}) is False

    def test_validate_hypothesis_missing_metric_fails(self) -> None:
        exp = ChaosExperiment(
            scenario=make_scenario(),
            steady_state={"error_rate_pct": 1.0},
        )
        assert exp.validate_hypothesis({}) is False

    def test_validate_hypothesis_no_steady_state_passes(self) -> None:
        exp = ChaosExperiment(scenario=make_scenario())
        assert exp.validate_hypothesis({}) is True

    def test_dry_run_slo_not_breached(self) -> None:
        exp = ChaosExperiment(scenario=make_scenario())
        result = exp.run_dry()
        assert result.slo_breached is False


# ── Pre-built scenarios ───────────────────────────────────────────────────────

class TestPrebuiltScenarios:
    def test_mlflow_down(self) -> None:
        s = mlflow_down_scenario()
        assert s.name == "mlflow-down"
        assert s.failure_type == FailureType.PROCESS_KILL
        assert s.blast_radius == "low"

    def test_minio_down(self) -> None:
        s = minio_down_scenario()
        assert s.blast_radius == "medium"
        assert len(s.inject_cmd) >= 1

    def test_kserve_crashloop(self) -> None:
        s = kserve_crashloop_scenario()
        assert s.failure_type == FailureType.BAD_ARTIFACT
        assert "helm" in s.inject_cmd[0]

    def test_gpu_node_gone(self) -> None:
        s = gpu_node_gone_scenario()
        assert s.failure_type == FailureType.NODE_DRAIN
        assert s.blast_radius == "high"
        assert len(s.inject_cmd) == 2

    def test_queue_backlog(self) -> None:
        s = queue_backlog_scenario()
        assert s.failure_type == FailureType.RESOURCE_EXHAUST

    def test_all_prebuilt_pass_dry_run(self) -> None:
        scenarios = [
            mlflow_down_scenario(),
            minio_down_scenario(),
            kserve_crashloop_scenario(),
            gpu_node_gone_scenario(),
            queue_backlog_scenario(),
        ]
        for s in scenarios:
            result = ChaosExperiment(scenario=s).run_dry()
            assert result.passed, f"{s.name} dry-run failed: {result.notes}"
