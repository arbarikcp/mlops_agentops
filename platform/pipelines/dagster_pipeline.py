"""Dagster-style training pipeline for credit-risk model.

Implements the core Dagster patterns without requiring `dagster` to be
installed — same concepts, native Python execution:
  - PipelineConfig: Pydantic-like config object with from_env()
  - ResourceRegistry: injectable dependency container (like Dagster resources)
  - AssetFn: typed wrapper for an asset-producing function
  - TrainingPipeline: wires assets into a SimpleDag and runs them

Run without Dagster:
    from pipelines.dagster_pipeline import TrainingPipeline, PipelineConfig
    pipeline = TrainingPipeline.build(PipelineConfig())
    result = pipeline.run()

If dagster is installed, use the definitions in dagster_defs.py (not in scope
for this curriculum — this file teaches the patterns).

See: docs/phase5/day32_dagster_pipeline.md
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from pipelines.dag import (
    AssetMaterialization,
    DagStep,
    DagRunResult,
    RetryPolicy,
    RunContext,
    SimpleDag,
)

log = logging.getLogger(__name__)


# ── Pipeline Config ───────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Validated configuration for the training pipeline.

    All parameters have sensible defaults; override via from_env() or kwargs.

    Attributes:
        n_estimators:          LightGBM tree count.
        learning_rate:         LightGBM learning rate.
        max_depth:             LightGBM max tree depth.
        test_size:             Fraction held out for test split.
        random_seed:           Random seed for reproducibility.
        auc_threshold:         Minimum AUC to promote model to champion.
        slice_gap_threshold:   Maximum AUC gap across demographic slices.
        early_stopping_rounds: LightGBM early stopping patience.
        mlflow_uri:            MLflow tracking server URI.
        data_path:             Path to processed features Parquet.
        model_output_dir:      Directory to write model artifacts.
    """

    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 6
    test_size: float = 0.2
    random_seed: int = 42
    auc_threshold: float = 0.75
    slice_gap_threshold: float = 0.10
    early_stopping_rounds: int = 20
    mlflow_uri: str = "http://localhost:5000"
    data_path: str = "data/processed/features.parquet"
    model_output_dir: str = "models"

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")
        if not 0 < self.auc_threshold <= 1:
            raise ValueError(f"auc_threshold must be in (0, 1], got {self.auc_threshold}")
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Build config from environment variables with defaults."""
        return cls(
            n_estimators=int(os.environ.get("PIPELINE_N_ESTIMATORS", "200")),
            learning_rate=float(os.environ.get("PIPELINE_LEARNING_RATE", "0.05")),
            max_depth=int(os.environ.get("PIPELINE_MAX_DEPTH", "6")),
            test_size=float(os.environ.get("PIPELINE_TEST_SIZE", "0.2")),
            random_seed=int(os.environ.get("PIPELINE_RANDOM_SEED", "42")),
            auc_threshold=float(os.environ.get("PIPELINE_AUC_THRESHOLD", "0.75")),
            mlflow_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            data_path=os.environ.get("PIPELINE_DATA_PATH", "data/processed/features.parquet"),
            model_output_dir=os.environ.get("PIPELINE_MODEL_OUTPUT_DIR", "models"),
        )


# ── Resource Registry ─────────────────────────────────────────────────────────

class ResourceRegistry:
    """Injectable dependency container — analogous to Dagster resources.

    Resources are lazily instantiated on first `get()` call.

    Usage:
        registry = ResourceRegistry()
        registry.register("mlflow", lambda: mlflow)
        registry.register("data_loader", lambda: DataLoader(config.data_path))
    """

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {}
        self._cache: dict[str, Any] = {}

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a lazy factory for a named resource."""
        self._factories[name] = factory

    def get(self, name: str) -> Any:
        """Get a resource by name; instantiate once and cache."""
        if name not in self._cache:
            if name not in self._factories:
                raise KeyError(f"Resource {name!r} not registered")
            self._cache[name] = self._factories[name]()
        return self._cache[name]

    def has(self, name: str) -> bool:
        return name in self._factories

    def registered_names(self) -> list[str]:
        return list(self._factories.keys())


# ── Asset Results ─────────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    """Typed output from the train asset step."""
    model: Any                         # trained LightGBM / sklearn estimator
    auc: float
    n_train: int
    n_test: int
    feature_names: list[str]
    run_id: str = ""                   # MLflow run ID

    @property
    def passed_auc_gate(self) -> bool:
        return self.auc >= 0.0         # gate applied in ValidationResult


@dataclass
class SplitResult:
    """Train/test split output."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)


@dataclass
class ValidationResult:
    """Output from the model validation asset."""
    auc: float
    auc_threshold: float
    passed: bool
    rejection_reason: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PromotionResult:
    """Output from the champion promotion step."""
    promoted: bool
    model_version: str
    champion_path: str
    reason: str


# ── Asset Functions ───────────────────────────────────────────────────────────

class TrainingAssets:
    """Collection of asset-producing functions for the training pipeline.

    Each method corresponds to one asset node in the DAG.
    Methods receive (ctx: RunContext, **kwargs) and return a typed value.
    They also record AssetMaterialization on ctx.
    """

    def __init__(self, config: PipelineConfig, registry: ResourceRegistry) -> None:
        self.config = config
        self.registry = registry

    # ── Asset 1: raw_credit_data ──────────────────────────────────────────────

    def raw_credit_data(self, ctx: RunContext, **kwargs: Any) -> pd.DataFrame:
        """Load raw credit data from configured path.

        Returns a DataFrame with all raw columns.
        Records AssetMaterialization with row count.
        """
        # Use data_loader resource if registered, else load directly
        if self.registry.has("data_loader"):
            df = self.registry.get("data_loader")()
        else:
            df = self._load_raw(self.config.data_path)

        ctx.record_materialization(AssetMaterialization(
            asset_key="raw_credit_data",
            path=self.config.data_path,
            row_count=len(df),
            extra={"columns": list(df.columns)},
        ))
        log.info("raw_credit_data: loaded %d rows", len(df))
        return df

    def _load_raw(self, path: str) -> pd.DataFrame:
        """Load Parquet; fallback to synthetic data for tests."""
        try:
            return pd.read_parquet(path)
        except (FileNotFoundError, OSError):
            log.warning("Data file not found at %s — using synthetic data", path)
            rng = np.random.default_rng(self.config.random_seed)
            n = 1000
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

    # ── Asset 2: validated_data ───────────────────────────────────────────────

    def validated_data(self, ctx: RunContext, raw_df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Apply data contract validation. Raises if contract fails."""
        errors = self._validate(raw_df)
        if errors:
            raise ValueError(f"Data validation failed: {errors}")

        ctx.record_materialization(AssetMaterialization(
            asset_key="validated_data",
            path=self.config.data_path,
            row_count=len(raw_df),
            extra={"validation": "passed", "errors": []},
        ))
        return raw_df

    def _validate(self, df: pd.DataFrame) -> list[str]:
        errors: list[str] = []
        if len(df) == 0:
            errors.append("Dataset is empty")
        label_col = "default.payment.next.month"
        if label_col in df.columns:
            null_rate = df[label_col].isna().mean()
            if null_rate > 0.05:
                errors.append(f"Label null rate {null_rate:.1%} > 5%")
        return errors

    # ── Asset 3: feature_dataset ──────────────────────────────────────────────

    def feature_dataset(self, ctx: RunContext, validated_df: pd.DataFrame, **kwargs: Any) -> SplitResult:
        """Featurize and split into train/test."""
        label_col = "default.payment.next.month"
        protected_cols = ["EDUCATION", "SEX", "MARRIAGE"]
        feature_cols = [
            c for c in validated_df.columns
            if c != label_col and c not in protected_cols
        ]

        X = validated_df[feature_cols].fillna(0).astype(float)
        y = validated_df[label_col].astype(int)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_seed,
            stratify=y,
        )

        split = SplitResult(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            feature_names=feature_cols,
        )

        ctx.record_materialization(AssetMaterialization(
            asset_key="feature_dataset",
            path=self.config.data_path,
            row_count=len(validated_df),
            extra={"n_train": split.n_train, "n_test": split.n_test,
                   "n_features": len(feature_cols)},
        ))
        return split

    # ── Asset 4: trained_model ────────────────────────────────────────────────

    def trained_model(self, ctx: RunContext, split: SplitResult, **kwargs: Any) -> TrainResult:
        """Train LightGBM model and return TrainResult."""
        from sklearn.metrics import roc_auc_score

        try:
            import lightgbm as lgb  # type: ignore[import]
            model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=self.config.random_seed,
                verbosity=-1,
            )
            model.fit(
                split.X_train, split.y_train,
                eval_set=[(split.X_test, split.y_test)],
                callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
            )
        except ImportError:
            # Fallback for environments without LightGBM (tests)
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=min(self.config.n_estimators, 50),
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=self.config.random_seed,
            )
            model.fit(split.X_train, split.y_train)

        proba = model.predict_proba(split.X_test)[:, 1]
        auc = float(roc_auc_score(split.y_test, proba))

        result = TrainResult(
            model=model,
            auc=auc,
            n_train=split.n_train,
            n_test=split.n_test,
            feature_names=split.feature_names,
        )

        import os
        os.makedirs(self.config.model_output_dir, exist_ok=True)
        model_path = f"{self.config.model_output_dir}/credit_risk_lgbm.pkl"

        ctx.record_materialization(AssetMaterialization(
            asset_key="trained_model",
            path=model_path,
            extra={"auc": auc, "n_train": split.n_train},
        ))
        log.info("trained_model: AUC=%.4f", auc)
        return result

    # ── Asset 5: validation_report ────────────────────────────────────────────

    def validation_report(
        self,
        ctx: RunContext,
        train_result: TrainResult,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate trained model against promotion thresholds."""
        passed = train_result.auc >= self.config.auc_threshold
        rejection = None if passed else (
            f"AUC {train_result.auc:.4f} < threshold {self.config.auc_threshold}"
        )

        result = ValidationResult(
            auc=train_result.auc,
            auc_threshold=self.config.auc_threshold,
            passed=passed,
            rejection_reason=rejection,
            metrics={"auc": train_result.auc},
        )

        report_path = "metrics/validation_report.json"
        ctx.record_materialization(AssetMaterialization(
            asset_key="validation_report",
            path=report_path,
            extra={"passed": passed, "auc": train_result.auc},
        ))
        log.info(
            "validation_report: AUC=%.4f threshold=%.2f passed=%s",
            train_result.auc, self.config.auc_threshold, passed,
        )
        return result

    # ── Asset 6: champion_model ───────────────────────────────────────────────

    def champion_model(
        self,
        ctx: RunContext,
        validation: ValidationResult,
        train_result: TrainResult,
        **kwargs: Any,
    ) -> PromotionResult:
        """Promote model to champion if validation passed.

        Raises ValueError if validation failed — DAG marks this step as failed,
        which surfaces as a clear gate failure in the pipeline UI.
        """
        if not validation.passed:
            raise ValueError(
                f"Champion promotion blocked: {validation.rejection_reason}"
            )

        import os
        import pickle
        champion_path = f"{self.config.model_output_dir}/champion_model.pkl"
        os.makedirs(self.config.model_output_dir, exist_ok=True)

        try:
            with open(champion_path, "wb") as f:
                pickle.dump(train_result.model, f)
        except (OSError, TypeError) as exc:
            log.warning("Could not write champion model file: %s", exc)

        version = f"v-{ctx.run_id[:8]}"
        result = PromotionResult(
            promoted=True,
            model_version=version,
            champion_path=champion_path,
            reason=f"AUC {validation.auc:.4f} >= {self.config.auc_threshold}",
        )

        ctx.record_materialization(AssetMaterialization(
            asset_key="champion_model",
            path=champion_path,
            extra={"version": version, "auc": validation.auc},
        ))
        log.info("champion_model: promoted as %s", version)
        return result


# ── Training Pipeline ─────────────────────────────────────────────────────────

class TrainingPipeline:
    """Wires TrainingAssets into a SimpleDag for execution.

    Each asset function becomes a DagStep. The DAG resolves outputs via
    a shared outputs dict — simulating Dagster's IO Manager pattern.

    Args:
        name:     Human-readable pipeline name.
        config:   PipelineConfig instance.
        registry: ResourceRegistry with configured resources.
    """

    def __init__(
        self,
        name: str,
        config: PipelineConfig,
        registry: ResourceRegistry,
    ) -> None:
        self.name = name
        self.config = config
        self.registry = registry
        self._assets = TrainingAssets(config, registry)

    @classmethod
    def build(
        cls,
        config: PipelineConfig | None = None,
        registry: ResourceRegistry | None = None,
        name: str = "credit_risk_training",
    ) -> "TrainingPipeline":
        """Build a default pipeline with sensible resource defaults."""
        return cls(
            name=name,
            config=config or PipelineConfig(),
            registry=registry or ResourceRegistry(),
        )

    def _build_dag(self, outputs: dict[str, Any]) -> SimpleDag:
        """Construct the asset DAG, threading outputs between steps."""
        dag = SimpleDag(self.name)

        def step_raw(ctx: RunContext, **kw: Any) -> pd.DataFrame:
            result = self._assets.raw_credit_data(ctx)
            outputs["raw_df"] = result
            return result

        def step_validated(ctx: RunContext, **kw: Any) -> pd.DataFrame:
            result = self._assets.validated_data(ctx, raw_df=outputs["raw_df"])
            outputs["validated_df"] = result
            return result

        def step_features(ctx: RunContext, **kw: Any) -> SplitResult:
            result = self._assets.feature_dataset(ctx, validated_df=outputs["validated_df"])
            outputs["split"] = result
            return result

        def step_train(ctx: RunContext, **kw: Any) -> TrainResult:
            result = self._assets.trained_model(ctx, split=outputs["split"])
            outputs["train_result"] = result
            return result

        def step_validate(ctx: RunContext, **kw: Any) -> ValidationResult:
            result = self._assets.validation_report(ctx, train_result=outputs["train_result"])
            outputs["validation"] = result
            return result

        def step_promote(ctx: RunContext, **kw: Any) -> PromotionResult:
            return self._assets.champion_model(
                ctx,
                validation=outputs["validation"],
                train_result=outputs["train_result"],
            )

        (
            dag
            .add_step(DagStep("raw_credit_data", step_raw, RetryPolicy(max_attempts=2, delay_seconds=0)))
            .add_step(DagStep("validated_data", step_validated, RetryPolicy(max_attempts=1), depends_on=["raw_credit_data"]))
            .add_step(DagStep("feature_dataset", step_features, RetryPolicy(max_attempts=1), depends_on=["validated_data"]))
            .add_step(DagStep("trained_model", step_train, RetryPolicy(max_attempts=2, delay_seconds=0), depends_on=["feature_dataset"]))
            .add_step(DagStep("validation_report", step_validate, RetryPolicy(max_attempts=1), depends_on=["trained_model"]))
            .add_step(DagStep("champion_model", step_promote, RetryPolicy(max_attempts=1), depends_on=["validation_report"]))
        )
        return dag

    def run(
        self,
        partition: str | None = None,
        run_id: str | None = None,
    ) -> DagRunResult:
        """Execute the full training pipeline.

        Args:
            partition: Time partition key (e.g. "2024-01").
            run_id:    Explicit run ID (auto-generated if None).

        Returns:
            DagRunResult with per-step outcomes, materializations, and timing.
        """
        outputs: dict[str, Any] = {}
        dag = self._build_dag(outputs)
        return dag.run(partition=partition, run_id=run_id)
