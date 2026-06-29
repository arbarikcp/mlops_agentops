"""Tests for pipelines/dagster_pipeline.py — Dagster-style pipeline."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pipelines.dag import RunContext
from pipelines.dagster_pipeline import (
    PipelineConfig,
    PromotionResult,
    ResourceRegistry,
    SplitResult,
    TrainResult,
    TrainingAssets,
    TrainingPipeline,
    ValidationResult,
)


# ── PipelineConfig ─────────────────────────────────────────────────────────────

class TestPipelineConfig:
    def test_defaults(self) -> None:
        cfg = PipelineConfig()
        assert cfg.n_estimators == 200
        assert cfg.learning_rate == 0.05
        assert cfg.test_size == 0.2
        assert cfg.auc_threshold == 0.75

    def test_invalid_test_size_raises(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            PipelineConfig(test_size=0.0)

    def test_invalid_test_size_above_1(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            PipelineConfig(test_size=1.0)

    def test_invalid_auc_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="auc_threshold"):
            PipelineConfig(auc_threshold=0.0)

    def test_invalid_n_estimators_raises(self) -> None:
        with pytest.raises(ValueError, match="n_estimators"):
            PipelineConfig(n_estimators=0)

    def test_invalid_learning_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            PipelineConfig(learning_rate=0.0)

    def test_from_env_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv("PIPELINE_N_ESTIMATORS", raising=False)
        monkeypatch.delenv("PIPELINE_LEARNING_RATE", raising=False)
        cfg = PipelineConfig.from_env()
        assert cfg.n_estimators == 200
        assert cfg.learning_rate == 0.05

    def test_from_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("PIPELINE_N_ESTIMATORS", "50")
        monkeypatch.setenv("PIPELINE_AUC_THRESHOLD", "0.80")
        cfg = PipelineConfig.from_env()
        assert cfg.n_estimators == 50
        assert cfg.auc_threshold == 0.80


# ── ResourceRegistry ───────────────────────────────────────────────────────────

class TestResourceRegistry:
    def test_register_and_get(self) -> None:
        reg = ResourceRegistry()
        reg.register("counter", lambda: [0])
        result = reg.get("counter")
        assert result == [0]

    def test_get_unknown_raises(self) -> None:
        reg = ResourceRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("missing")

    def test_factory_called_once(self) -> None:
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            return object()

        reg = ResourceRegistry()
        reg.register("obj", factory)
        reg.get("obj")
        reg.get("obj")
        assert calls["n"] == 1

    def test_has_returns_true_for_registered(self) -> None:
        reg = ResourceRegistry()
        reg.register("x", lambda: None)
        assert reg.has("x") is True
        assert reg.has("y") is False

    def test_registered_names(self) -> None:
        reg = ResourceRegistry()
        reg.register("a", lambda: None)
        reg.register("b", lambda: None)
        assert set(reg.registered_names()) == {"a", "b"}


# ── Synthetic data helper ──────────────────────────────────────────────────────

def make_synthetic_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "LIMIT_BAL": rng.uniform(10_000, 500_000, n),
        "AGE": rng.integers(20, 70, n),
        "BILL_AMT1": rng.uniform(0, 200_000, n),
        "PAY_AMT1": rng.uniform(0, 50_000, n),
        "default.payment.next.month": rng.integers(0, 2, n),
        "EDUCATION": rng.integers(1, 5, n),
        "SEX": rng.integers(1, 3, n),
        "MARRIAGE": rng.integers(0, 4, n),
    })


# ── TrainingAssets ─────────────────────────────────────────────────────────────

class TestTrainingAssets:
    @pytest.fixture
    def config(self) -> PipelineConfig:
        return PipelineConfig(n_estimators=10, test_size=0.2, auc_threshold=0.5)

    @pytest.fixture
    def registry(self) -> ResourceRegistry:
        df = make_synthetic_df()
        reg = ResourceRegistry()
        reg.register("data_loader", lambda: lambda: df)
        return reg

    @pytest.fixture
    def assets(self, config, registry) -> TrainingAssets:
        return TrainingAssets(config, registry)

    @pytest.fixture
    def ctx(self) -> RunContext:
        return RunContext(run_id="test-run")

    def test_raw_credit_data_returns_dataframe(self, assets, ctx) -> None:
        df = assets.raw_credit_data(ctx)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_raw_credit_data_records_materialization(self, assets, ctx) -> None:
        assets.raw_credit_data(ctx)
        mat = ctx.get_materialization("raw_credit_data")
        assert mat is not None
        assert mat.row_count > 0

    def test_validated_data_passes_good_df(self, assets, ctx) -> None:
        df = make_synthetic_df()
        result = assets.validated_data(ctx, raw_df=df)
        assert isinstance(result, pd.DataFrame)

    def test_validated_data_rejects_empty(self, assets, ctx) -> None:
        empty = pd.DataFrame({"default.payment.next.month": []})
        with pytest.raises(ValueError, match="empty"):
            assets.validated_data(ctx, raw_df=empty)

    def test_feature_dataset_returns_split(self, assets, ctx) -> None:
        df = make_synthetic_df()
        split = assets.feature_dataset(ctx, validated_df=df)
        assert isinstance(split, SplitResult)
        assert split.n_train > 0
        assert split.n_test > 0

    def test_feature_dataset_excludes_label(self, assets, ctx) -> None:
        df = make_synthetic_df()
        split = assets.feature_dataset(ctx, validated_df=df)
        assert "default.payment.next.month" not in split.feature_names

    def test_feature_dataset_records_materialization(self, assets, ctx) -> None:
        df = make_synthetic_df()
        assets.feature_dataset(ctx, validated_df=df)
        mat = ctx.get_materialization("feature_dataset")
        assert mat is not None
        assert mat.extra["n_features"] > 0

    def test_trained_model_returns_train_result(self, assets, ctx) -> None:
        df = make_synthetic_df()
        split = assets.feature_dataset(ctx, validated_df=df)
        result = assets.trained_model(ctx, split=split)
        assert isinstance(result, TrainResult)
        assert 0.0 <= result.auc <= 1.0
        assert result.model is not None

    def test_trained_model_records_materialization(self, assets, ctx) -> None:
        df = make_synthetic_df()
        split = assets.feature_dataset(ctx, validated_df=df)
        assets.trained_model(ctx, split=split)
        mat = ctx.get_materialization("trained_model")
        assert mat is not None
        assert "auc" in mat.extra

    def test_validation_report_passes_above_threshold(self, assets, ctx) -> None:
        train_result = TrainResult(
            model=None, auc=0.80, n_train=800, n_test=200, feature_names=[]
        )
        report = assets.validation_report(ctx, train_result=train_result)
        assert report.passed is True
        assert report.rejection_reason is None

    def test_validation_report_fails_below_threshold(self, assets, ctx) -> None:
        train_result = TrainResult(
            model=None, auc=0.40, n_train=800, n_test=200, feature_names=[]
        )
        report = assets.validation_report(ctx, train_result=train_result)
        assert report.passed is False
        assert report.rejection_reason is not None

    def test_champion_model_promotes_on_pass(self, assets, ctx, tmp_path) -> None:
        assets.config.model_output_dir = str(tmp_path)
        train_result = TrainResult(
            model=object(), auc=0.80, n_train=800, n_test=200, feature_names=[]
        )
        validation = ValidationResult(auc=0.80, auc_threshold=0.50, passed=True)
        result = assets.champion_model(ctx, validation=validation, train_result=train_result)
        assert result.promoted is True
        assert "v-" in result.model_version

    def test_champion_model_raises_on_fail(self, assets, ctx) -> None:
        validation = ValidationResult(
            auc=0.40, auc_threshold=0.75, passed=False,
            rejection_reason="AUC too low",
        )
        train_result = TrainResult(
            model=None, auc=0.40, n_train=100, n_test=50, feature_names=[]
        )
        with pytest.raises(ValueError, match="promotion blocked"):
            assets.champion_model(ctx, validation=validation, train_result=train_result)


# ── TrainingPipeline ───────────────────────────────────────────────────────────

class TestTrainingPipeline:
    @pytest.fixture
    def pipeline(self, tmp_path) -> TrainingPipeline:
        config = PipelineConfig(
            n_estimators=10,
            test_size=0.2,
            auc_threshold=0.01,   # very low to ensure promotion in tests
            model_output_dir=str(tmp_path),
        )
        df = make_synthetic_df()
        registry = ResourceRegistry()
        registry.register("data_loader", lambda: lambda: df)
        return TrainingPipeline.build(config=config, registry=registry)

    def test_run_succeeds(self, pipeline) -> None:
        result = pipeline.run()
        assert result.succeeded

    def test_run_produces_materializations(self, pipeline) -> None:
        result = pipeline.run()
        asset_keys = {m.asset_key for m in result.materializations}
        assert "raw_credit_data" in asset_keys
        assert "validated_data" in asset_keys
        assert "feature_dataset" in asset_keys
        assert "trained_model" in asset_keys

    def test_run_with_partition(self, pipeline) -> None:
        result = pipeline.run(partition="2024-01")
        assert result.succeeded
        # All materializations should carry the run_id
        assert all(m.run_id for m in result.materializations)

    def test_run_with_explicit_run_id(self, pipeline) -> None:
        result = pipeline.run(run_id="fixed-test-run")
        assert result.run_id == "fixed-test-run"

    def test_run_fails_on_high_auc_threshold(self, tmp_path) -> None:
        config = PipelineConfig(
            n_estimators=5,
            auc_threshold=0.9999,   # impossible threshold
            model_output_dir=str(tmp_path),
        )
        df = make_synthetic_df()
        registry = ResourceRegistry()
        registry.register("data_loader", lambda: lambda: df)
        pipeline = TrainingPipeline.build(config=config, registry=registry)
        result = pipeline.run()
        # champion_model step should fail; overall DAG fails
        assert not result.succeeded
        assert "champion_model" in result.failed_steps

    def test_build_factory(self) -> None:
        pipeline = TrainingPipeline.build()
        assert pipeline.name == "credit_risk_training"

    def test_custom_name(self) -> None:
        pipeline = TrainingPipeline.build(name="custom_pipeline")
        assert pipeline.name == "custom_pipeline"
