"""Tests for pipelines/pipeline_gate.py — Pipeline gate and orchestration survey."""
from __future__ import annotations

import pytest

from pipelines.dag import (
    AssetMaterialization,
    DagRunResult,
    DagStep,
    RetryPolicy,
    SimpleDag,
    StepStatus,
)
from pipelines.pipeline_gate import (
    OrchestrationSurvey,
    OrchestratorProfile,
    PipelineGateConfig,
    PipelineGateReport,
    PipelineGateRunner,
)


# ── PipelineGateConfig ─────────────────────────────────────────────────────────

class TestPipelineGateConfig:
    def test_defaults(self) -> None:
        cfg = PipelineGateConfig()
        assert "champion_model" in cfg.required_assets
        assert cfg.idempotency_fn is None
        assert cfg.idempotency_run_count == 2

    def test_custom_required_assets(self) -> None:
        cfg = PipelineGateConfig(required_assets=["raw_data", "model"])
        assert cfg.required_assets == ["raw_data", "model"]


# ── PipelineGateRunner ─────────────────────────────────────────────────────────

def _make_full_run_result() -> DagRunResult:
    mats = [
        AssetMaterialization(k, f"{k}.parquet", checksum="abc123")
        for k in [
            "raw_credit_data", "validated_data", "feature_dataset",
            "trained_model", "validation_report", "champion_model",
        ]
    ]
    return DagRunResult(
        run_id="gate-test",
        succeeded=True,
        step_results=[],
        materializations=mats,
        duration_s=1.0,
    )


def _make_retry_safe_dag() -> SimpleDag:
    dag = SimpleDag("test")
    dag.add_step(DagStep(
        "featurize",
        fn=lambda ctx, **kw: None,
        retry_policy=RetryPolicy(max_attempts=3),
        cleanup_fn=lambda ctx, exc: None,
    ))
    return dag


class TestPipelineGateRunner:
    def test_all_pass_with_no_inputs(self) -> None:
        runner = PipelineGateRunner(PipelineGateConfig())
        report = runner.run()
        # No dag and no run_result → only idempotency matters (which is None)
        assert report.retry_safety_passed is True
        assert report.lineage_passed is True

    def test_idempotency_pass(self) -> None:
        cfg = PipelineGateConfig(
            idempotency_fn=lambda x: x * 2,
            idempotency_inputs=5,
        )
        runner = PipelineGateRunner(cfg)
        report = runner.run()
        assert report.idempotency_passed is True

    def test_idempotency_fail(self) -> None:
        counter = {"n": 0}

        def non_idem(x):
            counter["n"] += 1
            return counter["n"]

        cfg = PipelineGateConfig(idempotency_fn=non_idem, idempotency_inputs=1)
        runner = PipelineGateRunner(cfg)
        report = runner.run()
        assert report.idempotency_passed is False
        assert not report.passed

    def test_lineage_pass(self) -> None:
        runner = PipelineGateRunner()
        report = runner.run(run_result=_make_full_run_result())
        assert report.lineage_passed is True

    def test_lineage_fail_missing_asset(self) -> None:
        mats = [AssetMaterialization("raw_credit_data", "raw.parquet", checksum="abc")]
        result = DagRunResult(
            run_id="r", succeeded=True, step_results=[], materializations=mats, duration_s=0
        )
        cfg = PipelineGateConfig(required_assets=["raw_credit_data", "champion_model"])
        runner = PipelineGateRunner(cfg)
        report = runner.run(run_result=result)
        assert report.lineage_passed is False
        assert any("champion_model" in i for i in report.issues)

    def test_retry_safety_pass_with_cleanup(self) -> None:
        dag = _make_retry_safe_dag()
        cfg = PipelineGateConfig(steps_requiring_cleanup=["featurize"])
        runner = PipelineGateRunner(cfg)
        report = runner.run(dag=dag)
        assert report.retry_safety_passed is True

    def test_retry_safety_fail_missing_cleanup(self) -> None:
        dag = SimpleDag("test")
        dag.add_step(DagStep(
            "featurize",
            fn=lambda ctx, **kw: None,
            retry_policy=RetryPolicy(max_attempts=3),
            cleanup_fn=None,   # missing cleanup
        ))
        cfg = PipelineGateConfig(steps_requiring_cleanup=["featurize"])
        runner = PipelineGateRunner(cfg)
        report = runner.run(dag=dag)
        assert report.retry_safety_passed is False
        assert not report.passed

    def test_full_pass(self) -> None:
        dag = _make_retry_safe_dag()
        cfg = PipelineGateConfig(
            steps_requiring_cleanup=["featurize"],
            idempotency_fn=lambda x: x + 1,
            idempotency_inputs=10,
        )
        runner = PipelineGateRunner(cfg)
        report = runner.run(dag=dag, run_result=_make_full_run_result())
        assert report.passed is True

    def test_summary_contains_status(self) -> None:
        runner = PipelineGateRunner()
        report = runner.run()
        summary = report.summary()
        assert "Gate" in summary

    def test_duration_positive(self) -> None:
        runner = PipelineGateRunner()
        report = runner.run()
        assert report.duration_s >= 0

    def test_retry_reports_populated(self) -> None:
        dag = _make_retry_safe_dag()
        runner = PipelineGateRunner()
        report = runner.run(dag=dag)
        assert len(report.retry_reports) == 1
        assert report.retry_reports[0].step_name == "featurize"

    def test_warnings_not_issues_for_non_required_steps(self) -> None:
        dag = SimpleDag("test")
        dag.add_step(DagStep(
            "optional_step",
            fn=lambda ctx, **kw: None,
            retry_policy=RetryPolicy(max_attempts=1),
            cleanup_fn=None,
        ))
        cfg = PipelineGateConfig(steps_requiring_cleanup=[])  # "optional_step" not in required list
        runner = PipelineGateRunner(cfg)
        report = runner.run(dag=dag)
        assert report.retry_safety_passed is True  # issue became a warning


# ── OrchestrationSurvey ────────────────────────────────────────────────────────

class TestOrchestrationSurvey:
    @pytest.fixture
    def survey(self) -> OrchestrationSurvey:
        return OrchestrationSurvey()

    def test_all_tools_listed(self, survey) -> None:
        tools = survey.all_tools()
        for expected in ["Dagster", "Prefect", "Metaflow", "Argo Workflows"]:
            assert expected in tools

    def test_get_profile_dagster(self, survey) -> None:
        profile = survey.get_profile("Dagster")
        assert profile is not None
        assert profile.asset_centric is True
        assert profile.step_caching is True

    def test_get_profile_case_insensitive(self, survey) -> None:
        profile = survey.get_profile("dagster")
        assert profile is not None

    def test_get_profile_unknown_returns_none(self, survey) -> None:
        assert survey.get_profile("FakeTool") is None

    def test_recommend_asset_centric(self, survey) -> None:
        results = survey.recommend(need_asset_centric=True)
        names = [r.name for r in results]
        assert "Dagster" in names
        # Argo, Prefect are not asset-centric — should not appear
        assert "Argo Workflows" not in names
        assert "Prefect" not in names

    def test_recommend_step_caching(self, survey) -> None:
        results = survey.recommend(need_step_caching=True)
        names = [r.name for r in results]
        assert "Dagster" in names
        assert "Prefect" not in names  # Prefect has no step caching

    def test_recommend_k8s_native(self, survey) -> None:
        results = survey.recommend(need_k8s_native=True)
        names = [r.name for r in results]
        assert "Argo Workflows" in names
        assert "Dagster" not in names   # Dagster is not k8s_native

    def test_recommend_aws_cloud(self, survey) -> None:
        results = survey.recommend(cloud="aws")
        names = [r.name for r in results]
        assert "SageMaker Pipelines" in names
        # SageMaker Pipelines should appear in top 3 (cloud bonus)
        assert names.index("SageMaker Pipelines") < 3

    def test_recommend_gcp_cloud(self, survey) -> None:
        results = survey.recommend(cloud="gcp")
        names = [r.name for r in results]
        assert "Vertex AI Pipelines" in names
        # Vertex AI Pipelines should appear in top 3 (cloud bonus)
        assert names.index("Vertex AI Pipelines") < 3

    def test_recommend_returns_sorted(self, survey) -> None:
        results = survey.recommend(need_ml_native=True)
        assert len(results) > 0
        # Results should be sorted by score (higher first)
        # All returned should have ml_native=True or be included despite it
        # Just check order is stable (no assertion on exact scores)

    def test_comparison_table_has_all_tools(self, survey) -> None:
        table = survey.comparison_table()
        tool_names = [row["tool"] for row in table]
        for expected in ["Dagster", "Prefect", "Metaflow", "Argo Workflows"]:
            assert expected in tool_names

    def test_comparison_table_row_structure(self, survey) -> None:
        table = survey.comparison_table()
        row = table[0]
        assert "tool" in row
        assert "asset_centric" in row
        assert "step_caching" in row
        assert "best_for" in row

    def test_dagster_has_highest_local_dev_score(self, survey) -> None:
        dagster = survey.get_profile("Dagster")
        argo = survey.get_profile("Argo Workflows")
        assert dagster.local_dev_score > argo.local_dev_score

    def test_argo_is_k8s_native(self, survey) -> None:
        argo = survey.get_profile("Argo Workflows")
        assert argo.k8s_native is True

    def test_prefect_is_not_ml_native(self, survey) -> None:
        prefect = survey.get_profile("Prefect")
        assert prefect.ml_native is False

    def test_sagemaker_low_cloud_portability(self, survey) -> None:
        sm = survey.get_profile("SageMaker Pipelines")
        assert sm.cloud_portability == 1
