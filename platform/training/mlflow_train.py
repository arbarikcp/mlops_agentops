"""MLflow-integrated training — Day 10 deliverable.

Wraps the deterministic train.py pipeline with full MLflow tracking:
  - Logs all params from params.yaml (flat dict)
  - Logs metrics (AUC, AUPR, Brier, ECE, confusion details)
  - Logs the trained model with a signature and input example
  - Tags the run with feature list, data version, git commit

MLflow concepts used here:
  - mlflow.start_run():        Creates a run, returns a context manager.
  - mlflow.log_params():       Key-value pairs (strings). Max 500 per run.
  - mlflow.log_metrics():      Float values at optional step. Queryable in UI.
  - mlflow.lightgbm.log_model: Logs model + conda env + signature.
  - mlflow.models.infer_signature: Captures input schema from a sample.
  - mlflow.set_tag():          Free-form metadata.

MLflow Architecture (our setup):
  ┌─────────────────────────────────────┐
  │  MLflow Tracking Server (:5000)     │
  │    backend:  Postgres (metadata)    │
  │    artifacts: MinIO (model files)   │
  └─────────────────────────────────────┘

Usage:
    # Start MLflow first:
    make up

    # Run training:
    PYTHONHASHSEED=42 python -m training.mlflow_train --params params.yaml

    # View results:
    open http://localhost:5000

Debugging:
    - "Connection refused :5000": MLflow server not running → make up
    - "Access denied MinIO": Check MINIO_ROOT_USER/PASSWORD in .env
    - "Experiment not found": First run creates it automatically.
    - To override tracking URI: export MLFLOW_TRACKING_URI=http://localhost:5000
    - To see all runs in Python:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        runs = mlflow.search_runs(experiment_names=["m1-credit-risk-training"])
        print(runs[["run_id", "metrics.roc_auc"]].sort_values("metrics.roc_auc", ascending=False))
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.lightgbm
import mlflow.models
import pandas as pd

from training.config import TrainingParams
from training.evaluate import compute_metrics, compute_confusion_details
from training.features import clean_raw_data, engineer_features, split_data
from training.train import build_lgbm, load_processed, load_raw_and_featurize, set_all_seeds

log = logging.getLogger(__name__)


def _git_commit() -> str:
    """Get current git commit SHA for traceability."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def configure_mlflow(cfg: TrainingParams) -> None:
    """Point MLflow at our local tracking server and create/set the experiment."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    log.info("MLflow tracking URI: %s | experiment: %s", tracking_uri, cfg.mlflow.experiment_name)


def run_training_with_mlflow(
    params_path: str = "params.yaml",
    input_path: str | None = None,
    tags: dict[str, str] | None = None,
) -> str:
    """Run deterministic training inside an MLflow run. Returns run_id."""
    cfg = TrainingParams.from_yaml(params_path)
    set_all_seeds(cfg.data.random_seed)
    configure_mlflow(cfg)

    run_name = cfg.mlflow.run_name_prefix
    extra_tags = {
        "git.commit": _git_commit(),
        "params_file": params_path,
        "phase": "m1-phase1",
        **(tags or {}),
    }

    with mlflow.start_run(run_name=run_name, tags=extra_tags) as run:
        run_id = run.info.run_id
        log.info("Started MLflow run: %s", run_id)

        # ── 1. Log all params ─────────────────────────────────────────────
        mlflow.log_params(cfg.flat_params())
        mlflow.log_artifact(params_path, artifact_path="config")

        # ── 2. Load data ──────────────────────────────────────────────────
        processed = Path(input_path) if input_path else cfg.data.processed_path
        if processed.exists():
            df = load_processed(processed)
        else:
            df = load_raw_and_featurize(cfg)

        X_train, X_test, y_train, y_test = split_data(df, cfg.data)
        mlflow.log_params({
            "split.n_train": len(X_train),
            "split.n_test": len(X_test),
            "split.n_features": X_train.shape[1],
            "split.positive_rate_train": f"{y_train.mean():.4f}",
        })

        # ── 3. Train ──────────────────────────────────────────────────────
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        model = build_lgbm(cfg, n_pos, n_neg)
        model.fit(X_train, y_train)

        # ── 4. Evaluate ───────────────────────────────────────────────────
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test.to_numpy(), y_prob, cfg.evaluation.threshold)
        confusion = compute_confusion_details(y_test.to_numpy(), y_prob, cfg.evaluation.threshold)

        mlflow.log_metrics(metrics)
        mlflow.log_metrics(confusion)

        # ── 5. Log model with signature ───────────────────────────────────
        # Signature captures the exact input schema — enforced at serve time
        input_sample = X_train.iloc[:5]
        output_sample = model.predict_proba(input_sample)[:, 1]
        signature = mlflow.models.infer_signature(
            model_input=X_train,
            model_output=pd.Series(
                model.predict_proba(X_train)[:, 1], name="default_probability"
            ),
        )
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path=cfg.mlflow.artifact_path,
            signature=signature,
            input_example=input_sample,
        )

        # ── 6. Tags for searchability ─────────────────────────────────────
        mlflow.set_tag("features", ",".join(X_train.columns.tolist()))
        mlflow.set_tag("feature_count", str(X_train.shape[1]))

        log.info(
            "Run complete: run_id=%s | AUC=%.4f | AP=%.4f | ECE=%.4f",
            run_id, metrics["roc_auc"], metrics["average_precision"], metrics["calibration_error"],
        )

        # Save run_id to file for DVC to track
        Path("metrics/mlflow_run_id.txt").parent.mkdir(exist_ok=True)
        Path("metrics/mlflow_run_id.txt").write_text(run_id)

        return run_id


if __name__ == "__main__":
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train with MLflow tracking")
    parser.add_argument("--params", default="params.yaml")
    parser.add_argument("--input", default=None, help="Processed parquet path (optional)")
    args = parser.parse_args()

    run_id = run_training_with_mlflow(params_path=args.params, input_path=args.input)
    print(f"\nMLflow run_id: {run_id}")
    print("View at: http://localhost:5000")
