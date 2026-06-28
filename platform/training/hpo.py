"""Optuna hyperparameter optimization with MLflow nested runs — Day 12 deliverable.

Structure created in MLflow:
  Experiment: m1-credit-risk-training
    └── Run: lgbm-sweep (parent)            ← params: n_trials, best params
          ├── Run: trial-0 (child/nested)   ← params: trial hyperparams, metric: cv_auc
          ├── Run: trial-1 (child/nested)
          ...
          └── Run: trial-N → best model logged here

Why nested runs:
  - Parent run = the sweep as a whole (queryable as a unit)
  - Child runs = individual trials (compare in MLflow UI leaderboard)
  - The "best trial" is visible both in Optuna's study and MLflow's run table

Optuna concepts:
  - Study:    the optimization session (persists across calls)
  - Trial:    one evaluation of the objective function
  - Sampler:  strategy for proposing parameters (TPE = tree-structured Parzen estimator)
  - Pruner:   early stopping for bad trials (MedianPruner kills bottom 50%)
  - Direction: "maximize" for AUC

Usage:
    # Standard sweep (20 trials):
    PYTHONHASHSEED=42 python -m training.hpo --n-trials 20

    # Quick check (3 trials):
    PYTHONHASHSEED=42 python -m training.hpo --n-trials 3

    # Resume a study (same study name = Optuna reuses history):
    PYTHONHASHSEED=42 python -m training.hpo --n-trials 20 --study-name credit-risk-v2

Debugging:
    - View trial history: python -m training.hpo --report <study_name>
    - "No trials completed": pruner may be too aggressive → increase n_warmup_steps
    - AUC not improving: widen search space in _define_search_space()
    - Memory OOM: reduce n_splits in CV or add --sample N to subsample training data
    - View in MLflow UI:
        http://localhost:5000  → experiment → filter by run_name contains "sweep"
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.models
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from training.config import TrainingParams
from training.evaluate import compute_metrics
from training.features import split_data
from training.train import load_processed, load_raw_and_featurize, set_all_seeds

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Search space ─────────────────────────────────────────────────────────────

def _define_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """Define the hyperparameter search space for LightGBM.

    Ranges chosen based on:
    - n_estimators: wider than default; early stopping limits over-fitting
    - learning_rate: log-uniform (small changes at low end matter more)
    - max_depth: -1 means unlimited; constrain for regularisation
    - num_leaves: must be < 2^max_depth for meaningful constraint
    - subsample / colsample_bytree: standard regularisation range
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }


# ── Objective ────────────────────────────────────────────────────────────────

def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 3,
) -> float:
    """Cross-validated AUC for a single Optuna trial.

    StratifiedKFold preserves class balance in each fold (important for 22% positive rate).
    Reports intermediate values so MedianPruner can kill bad trials early.
    """
    params = _define_search_space(trial)
    params.update({"random_state": 42, "verbosity": -1, "n_jobs": -1})

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs: list[float] = []

    for fold, (idx_tr, idx_val) in enumerate(cv.split(X_train, y_train)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train[idx_tr], y_train[idx_tr])
        y_prob = model.predict_proba(X_train[idx_val])[:, 1]
        auc = roc_auc_score(y_train[idx_val], y_prob)
        fold_aucs.append(auc)

        # Report for pruner: if this fold is already below median, prune
        trial.report(float(np.mean(fold_aucs)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_aucs))


# ── Sweep orchestration ───────────────────────────────────────────────────────

def run_sweep(
    params_path: str = "params.yaml",
    n_trials: int = 20,
    study_name: str = "credit-risk-lgbm",
    input_path: str | None = None,
) -> optuna.Study:
    """Run Optuna sweep wrapped in MLflow parent run with nested trial runs.

    Steps:
      1. Load data.
      2. Create Optuna study.
      3. Open MLflow parent run.
      4. Each trial logs to a nested child run (via MLflowCallback).
      5. After all trials: log best params, retrain final model on full train set.
    """
    cfg = TrainingParams.from_yaml(params_path)
    set_all_seeds(cfg.data.random_seed)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Load data
    processed = Path(input_path) if input_path else cfg.data.processed_path
    df = load_processed(processed) if processed.exists() else load_raw_and_featurize(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg.data)

    X_train_arr = X_train.values
    y_train_arr = y_train.values

    # Optuna study (TPE sampler + MedianPruner)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    # MLflow callback: each trial → nested child run
    from optuna.integration.mlflow import MLflowCallback
    mlflow_cb = MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name="cv_auc",
        create_experiment=False,
        mlflow_kwargs={"nested": True},
    )

    # Parent MLflow run wraps the entire sweep
    with mlflow.start_run(run_name=f"{cfg.mlflow.run_name_prefix}-sweep") as parent:
        parent_run_id = parent.info.run_id
        mlflow.log_params({
            "sweep.n_trials": n_trials,
            "sweep.study_name": study_name,
            "sweep.sampler": "TPE",
            "sweep.pruner": "MedianPruner",
            "sweep.cv_folds": 3,
        })

        log.info("Starting Optuna sweep: %d trials", n_trials)
        study.optimize(
            lambda trial: objective(trial, X_train_arr, y_train_arr),
            n_trials=n_trials,
            callbacks=[mlflow_cb],
            show_progress_bar=False,
        )

        best = study.best_trial
        log.info("Best trial #%d: cv_auc=%.4f", best.number, best.value)
        mlflow.log_metric("best_cv_auc", best.value)
        mlflow.log_metric("n_completed_trials", len(study.trials))
        mlflow.log_params({f"best.{k}": v for k, v in best.params.items()})

        # Retrain final model with best params on full training set
        best_params = {**best.params, "random_state": 42, "verbosity": -1, "n_jobs": -1}
        final_model = lgb.LGBMClassifier(**best_params)
        final_model.fit(X_train_arr, y_train_arr)

        y_prob = final_model.predict_proba(X_test.values)[:, 1]
        final_metrics = compute_metrics(y_test.to_numpy(), y_prob, cfg.evaluation.threshold)
        mlflow.log_metrics({f"final.{k}": v for k, v in final_metrics.items()})

        # Log best model
        signature = mlflow.models.infer_signature(X_train, y_prob)
        mlflow.lightgbm.log_model(
            final_model,
            artifact_path=cfg.mlflow.artifact_path,
            signature=signature,
            input_example=X_train.iloc[:3],
        )
        mlflow.set_tag("sweep.parent_run_id", parent_run_id)
        log.info("Sweep complete. Final AUC=%.4f", final_metrics["roc_auc"])

    return study


def print_study_report(study: optuna.Study) -> None:
    print(f"\n{'='*60}")
    print(f"Study: {study.study_name}")
    print(f"Trials: {len(study.trials)} total, {len(study.best_trials)} best")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best cv_auc: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*60)


if __name__ == "__main__":
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run Optuna HPO sweep")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", default="credit-risk-lgbm")
    parser.add_argument("--input", default=None)
    args = parser.parse_args()

    study = run_sweep(
        params_path=args.params,
        n_trials=args.n_trials,
        study_name=args.study_name,
        input_path=args.input,
    )
    print_study_report(study)
