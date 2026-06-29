"""Reference statistics for train/serve skew detection.

The reference snapshot is computed from the training feature set and saved
alongside the model artifact. At serving time, incoming batches are compared
against this snapshot to detect distribution shift.

Why save reference stats at training time?
  You can't detect drift without a baseline. Drift = distance between
  current distribution and training distribution. The reference IS the
  training distribution summary.

Key types:
    ReferenceStats     — training-time snapshot tied to a model version
    compute_reference_stats()  — build from training DataFrame
    save_reference_stats()     — persist to JSON (portable, model-agnostic)
    load_reference_stats()     — reload for serving-time comparison

See: docs/phase3/day21_train_serve_skew.md for theory and architecture.

Usage:
    from monitoring.reference_stats import (
        compute_reference_stats, save_reference_stats, load_reference_stats
    )

    stats = compute_reference_stats(X_train, model_version="v1.2", feature_names=X_train.columns.tolist())
    save_reference_stats(stats, Path("models/reference_stats.json"))

    # At serving time
    ref = load_reference_stats(Path("models/reference_stats.json"))
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data.contracts.statistical_checks import DatasetStats, compute_dataset_stats

log = logging.getLogger(__name__)


@dataclass
class ReferenceStats:
    """Training-time feature statistics bound to a specific model version.

    Attributes:
        model_version:    Identifies the model artifact these stats correspond to.
        training_date:    ISO-8601 datetime when training was run.
        n_training_rows:  Number of rows in the training set.
        feature_names:    Ordered list of feature column names the model expects.
        dataset_stats:    Per-column statistics (DatasetStats from statistical_checks).
    """

    model_version: str
    training_date: str
    n_training_rows: int
    feature_names: list[str]
    dataset_stats: DatasetStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "training_date": self.training_date,
            "n_training_rows": self.n_training_rows,
            "feature_names": self.feature_names,
            "dataset_stats": self.dataset_stats.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReferenceStats":
        return cls(
            model_version=d["model_version"],
            training_date=d["training_date"],
            n_training_rows=d["n_training_rows"],
            feature_names=d["feature_names"],
            dataset_stats=DatasetStats.from_dict(d["dataset_stats"]),
        )


def compute_reference_stats(
    X_train: pd.DataFrame,
    *,
    model_version: str = "unknown",
    feature_names: list[str] | None = None,
    target_col: str | None = None,
) -> ReferenceStats:
    """Compute reference statistics from the training feature DataFrame.

    Call this immediately after the train/val split, using the training split only.
    Save the result alongside the model artifact.

    Args:
        X_train:       Training features (DataFrame). Exclude the target column.
        model_version: Version tag for the model artifact (e.g. "1.0.0" or MLflow run ID).
        feature_names: Explicit list of feature names. Defaults to X_train.columns.tolist().
        target_col:    If present in X_train, compute positive_rate. Usually None here
                       since X_train should be features-only.

    Returns:
        ReferenceStats snapshot ready for JSON serialization.
    """
    from datetime import datetime, timezone
    training_date = datetime.now(timezone.utc).isoformat()

    if feature_names is None:
        feature_names = X_train.columns.tolist()

    dataset_stats = compute_dataset_stats(
        X_train,
        dataset_name=f"training_{model_version}",
        target_col=target_col,
    )

    ref = ReferenceStats(
        model_version=model_version,
        training_date=training_date,
        n_training_rows=len(X_train),
        feature_names=feature_names,
        dataset_stats=dataset_stats,
    )

    log.info(
        "Computed reference stats: model_version=%s, n_rows=%d, n_features=%d",
        model_version, len(X_train), len(feature_names),
    )
    return ref


def save_reference_stats(ref: ReferenceStats, path: Path) -> None:
    """Serialize ReferenceStats to a JSON file.

    The JSON format is human-readable and model-framework-agnostic.
    Store this file next to the model pickle/ONNX artifact.

    Args:
        ref:  ReferenceStats to save.
        path: Destination file path (e.g. Path("models/reference_stats.json")).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ref.to_dict(), f, indent=2)
    log.info("Saved reference stats to %s", path)


def load_reference_stats(path: Path) -> ReferenceStats:
    """Load ReferenceStats from a JSON file.

    Args:
        path: Path to reference_stats.json.

    Raises:
        FileNotFoundError: if the file does not exist.
        KeyError / ValueError: if the JSON is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Reference stats not found at {path}. "
            "Run training to generate reference_stats.json alongside the model artifact."
        )
    with open(path) as f:
        d = json.load(f)
    ref = ReferenceStats.from_dict(d)
    log.info(
        "Loaded reference stats: model_version=%s, training_date=%s, n_rows=%d",
        ref.model_version, ref.training_date, ref.n_training_rows,
    )
    return ref


def check_feature_alignment(
    ref: ReferenceStats,
    serving_df: pd.DataFrame,
) -> dict[str, Any]:
    """Verify that serving features match the reference feature list.

    Catches serving skew caused by pipeline bugs (missing features, renamed columns).

    Returns:
        Dict with keys: aligned (bool), missing_in_serving (list), extra_in_serving (list).
    """
    ref_set = set(ref.feature_names)
    serving_set = set(serving_df.columns)

    missing = sorted(ref_set - serving_set)
    extra = sorted(serving_set - ref_set)
    aligned = len(missing) == 0

    if missing:
        log.warning(
            "Feature alignment: %d features missing in serving data: %s",
            len(missing), missing,
        )
    if extra:
        log.info("Feature alignment: %d extra columns in serving data (ignored): %s", len(extra), extra)
    if aligned and not extra:
        log.info("Feature alignment: OK — all %d features present", len(ref.feature_names))

    return {"aligned": aligned, "missing_in_serving": missing, "extra_in_serving": extra}
