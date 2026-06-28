"""Pydantic configuration models for training pipeline.

Everything that changes between experiment runs lives in params.yaml.
These models parse, validate, and type-annotate the YAML content.

Why Pydantic here instead of dataclasses:
  - Type coercion (e.g. YAML ints stay ints, not strings)
  - Validation (e.g. test_size must be in (0,1))
  - Dot-access to nested fields
  - Integrates with MLflow param logging (model_dump())

Usage:
    cfg = TrainingParams.from_yaml("params.yaml")
    print(cfg.model.learning_rate)  # 0.05
    mlflow.log_params(cfg.model.model_dump())
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    raw_path: Path = Path("data/raw/credit_card_default.csv")
    processed_path: Path = Path("data/processed/features.parquet")
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_seed: int = Field(42, ge=0)
    target_col: str = "DEFAULT_PAYMENT_NEXT_MONTH"


class FeaturesConfig(BaseModel):
    target_col: str = "DEFAULT_PAYMENT_NEXT_MONTH"
    id_col: str = "ID"
    categorical_cols: list[str] = Field(default_factory=list)
    payment_status_cols: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    algorithm: str = "lightgbm"
    n_estimators: int = Field(300, gt=0)
    learning_rate: float = Field(0.05, gt=0.0)
    max_depth: int = Field(6, ge=-1)
    num_leaves: int = Field(31, gt=1)
    min_child_samples: int = Field(20, gt=0)
    subsample: float = Field(0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(0.8, gt=0.0, le=1.0)
    random_state: int = Field(42, ge=0)
    early_stopping_rounds: int = Field(30, gt=0)

    def to_lgbm_params(self) -> dict[str, Any]:
        """Return kwargs suitable for LGBMClassifier constructor."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
            "verbosity": -1,
        }


class EvaluationConfig(BaseModel):
    threshold: float = Field(0.5, gt=0.0, lt=1.0)


class MLflowConfig(BaseModel):
    experiment_name: str = "m1-credit-risk-training"
    run_name_prefix: str = "lgbm"
    model_name: str = "credit-risk-model"
    artifact_path: str = "model"


class TrainingParams(BaseModel):
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TrainingParams":
        """Load and validate training params from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def flat_params(self) -> dict[str, Any]:
        """Flatten all params into a single dict for MLflow logging."""
        result: dict[str, Any] = {}
        for section, model in self.model_dump().items():
            if isinstance(model, dict):
                for k, v in model.items():
                    result[f"{section}.{k}"] = str(v)
        return result
