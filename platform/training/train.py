"""Deterministic training script — Day 7 deliverable.

WHAT this script does:
  1. Sets every source of non-determinism (Python, NumPy, LightGBM) to a fixed seed.
  2. Loads processed features (from featurize stage) or raw data as fallback.
  3. Performs time-based train/test split.
  4. Trains a LightGBM classifier with scale_pos_weight for class imbalance.
  5. Evaluates and writes metrics/train_metrics.json and models/credit_risk_model.pkl.

WHY determinism matters:
  Reproducibility Gate (Day 14): given a run_id, you must be able to reproduce the
  exact same model. This requires the same seed, same data, same code. This script
  is the "same code + same seed" part. DVC provides "same data".

NON-DETERMINISM SOURCES controlled here:
  - Python built-in random: random.seed(seed)
  - NumPy: np.random.seed(seed)
  - LightGBM: random_state=seed in constructor
  - Hash randomisation: PYTHONHASHSEED env var (set by Makefile / DVC stage cmd)
  - NOT controlled: GPU ops (we run CPU only for tabular models)

Usage:
    # Via DVC (recommended — also sets PYTHONHASHSEED):
    dvc repro train

    # Standalone (must set PYTHONHASHSEED externally):
    PYTHONHASHSEED=42 python -m training.train --params params.yaml

    # Override input path:
    PYTHONHASHSEED=42 python -m training.train \\
        --input data/processed/features.parquet \\
        --params params.yaml

Debugging:
    - "File not found" for input: run dvc repro featurize first.
    - Metric values vary run-to-run: PYTHONHASHSEED not set — use make train.
    - Memory error: reduce n_estimators or add --sample N flag.
    - To inspect model feature importances after training:
        python -c "import joblib; m=joblib.load('models/credit_risk_model.pkl'); print(dict(zip(m.feature_name_, m.feature_importances_)))"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from training.config import TrainingParams
from training.evaluate import compute_metrics, compute_confusion_details
from training.features import clean_raw_data, engineer_features, split_data

log = logging.getLogger(__name__)


# ── Seed control ─────────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    """Set every controllable source of randomness to the same seed.

    Must be called before any library that uses random numbers.
    Note: PYTHONHASHSEED must be set BEFORE the Python process starts —
    setting os.environ here has no effect on the current process's hash seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    # LightGBM uses its own RNG; random_state in constructor is the right control.
    # Scikit-learn uses numpy's global RNG; covered by np.random.seed above.
    log.debug("Set random seeds: python=%d, numpy=%d", seed, seed)

    current_hashseed = os.environ.get("PYTHONHASHSEED", "NOT SET")
    if current_hashseed != str(seed) and current_hashseed != "random":
        log.warning(
            "PYTHONHASHSEED=%s (expected %d). "
            "Use 'PYTHONHASHSEED=%d python -m training.train' for full determinism.",
            current_hashseed, seed, seed,
        )


# ── Data loading ─────────────────────────────────────────────────────────────

def load_processed(path: Path) -> pd.DataFrame:
    """Load already-featurized parquet file (output of DVC featurize stage)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed features not found at {path}. "
            "Run: dvc repro featurize  OR  python -m training.features"
        )
    df = pd.read_parquet(path)
    log.info("Loaded processed features: %d rows × %d cols from %s", len(df), len(df.columns), path)
    return df


def load_raw_and_featurize(cfg: TrainingParams) -> pd.DataFrame:
    """Fallback: load raw CSV and run featurization inline."""
    log.info("Loading raw data from %s (fallback path)", cfg.data.raw_path)
    df = pd.read_csv(cfg.data.raw_path)
    df.columns = [str(c).strip().upper() for c in df.columns]
    df = clean_raw_data(df)
    df = engineer_features(df, cfg.features)
    return df


# ── Model construction ───────────────────────────────────────────────────────

def build_lgbm(cfg: TrainingParams, n_pos: int, n_neg: int) -> lgb.LGBMClassifier:
    """Construct LightGBM with class-imbalance correction.

    scale_pos_weight = n_negative / n_positive adjusts the loss function
    so the model pays more attention to the minority class (defaulters).
    This is equivalent to oversampling positives, but faster and exact.
    """
    scale_pos_weight = n_neg / max(n_pos, 1)
    log.info(
        "Class balance: %d neg / %d pos → scale_pos_weight=%.2f",
        n_neg, n_pos, scale_pos_weight,
    )
    params = cfg.model.to_lgbm_params()
    params["scale_pos_weight"] = scale_pos_weight
    return lgb.LGBMClassifier(**params)


# ── Main training function ───────────────────────────────────────────────────

def run_training(
    params_path: str = "params.yaml",
    input_path: str | None = None,
) -> dict[str, Any]:
    """Full training pipeline. Returns metrics dict."""
    cfg = TrainingParams.from_yaml(params_path)
    set_all_seeds(cfg.data.random_seed)

    # Load data: prefer processed parquet (DVC output), fall back to raw
    processed = Path(input_path) if input_path else cfg.data.processed_path
    if processed.exists():
        df = load_processed(processed)
    else:
        log.warning("Processed parquet not found at %s — running raw pipeline", processed)
        df = load_raw_and_featurize(cfg)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, cfg.data)

    # Train
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    model = build_lgbm(cfg, n_pos, n_neg)

    log.info("Training LightGBM: %d features, n_estimators=%d", X_train.shape[1], cfg.model.n_estimators)
    model.fit(X_train, y_train)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_prob, cfg.evaluation.threshold)
    confusion = compute_confusion_details(y_test.to_numpy(), y_prob, cfg.evaluation.threshold)
    metrics.update(confusion)

    # Persist artifacts
    _save_model(model, Path("models/credit_risk_model.pkl"))
    _save_metrics(metrics, Path("metrics/train_metrics.json"))

    return metrics


def _save_model(model: lgb.LGBMClassifier, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Model saved → %s (%.1f KB)", path, path.stat().st_size / 1024)


def _save_metrics(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", path)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Train credit-risk LightGBM model")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--input", default=None, help="Path to processed parquet (overrides params)")
    args = parser.parse_args()

    metrics = run_training(params_path=args.params, input_path=args.input)
    print("\n=== Training complete ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
