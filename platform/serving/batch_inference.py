"""Batch inference with idempotency, manifest tracking, and backfill support.

Key design:
    - Every batch job writes a manifest AFTER the output is complete.
    - The manifest acts as a completion marker: if it exists + checksum matches,
      the job is already done and should not be re-run.
    - This makes batch jobs safe to retry after failure (idempotent).

Idempotency key:
    (job_id, model_version, data_partition)

Backfill:
    A backfill re-scores historical data. It uses a new model_version +
    data_partition pairs. The ManifestStore tracks which partitions are done.

See: docs/phase4/day27_batch_inference.md for theory.

Usage:
    from serving.batch_inference import BatchInferenceJob, ManifestStore

    store = ManifestStore(root_dir=Path("manifests/"))
    job = BatchInferenceJob(job_id="nightly_2024-01", model_version="v1.2",
                            manifest_store=store)

    result = job.run(features_df, output_path, predict_fn=runner.predict_numpy)
    print(f"Scored {result.n_rows} rows, skipped={result.skipped}")
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_CHUNK_SIZE = 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checksum_df(df: pd.DataFrame) -> str:
    """SHA-256 of the DataFrame contents as bytes."""
    h = hashlib.sha256()
    h.update(df.to_csv(index=False).encode())
    return h.hexdigest()


def _checksum_file(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class BatchJobManifest:
    """Completion record written after a batch job finishes.

    Attributes:
        job_id:          Unique job identifier.
        model_version:   Model version used for scoring.
        data_partition:  Partition label (e.g. "2024-01").
        n_rows_scored:   Number of rows scored in this run.
        output_path:     Absolute path to the scored output file.
        output_checksum: SHA-256 hex digest of the output file.
        status:          "completed" or "failed".
        started_at:      ISO-8601 UTC start time.
        completed_at:    ISO-8601 UTC completion time.
    """

    job_id: str
    model_version: str
    data_partition: str
    n_rows_scored: int
    output_path: str
    output_checksum: str
    status: str
    started_at: str
    completed_at: str

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "model_version": self.model_version,
            "data_partition": self.data_partition,
            "n_rows_scored": self.n_rows_scored,
            "output_path": self.output_path,
            "output_checksum": self.output_checksum,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BatchJobManifest":
        return cls(**d)


@dataclass
class BatchJobResult:
    """Result returned by BatchInferenceJob.run().

    Attributes:
        job_id:      Job identifier.
        n_rows:      Number of rows in the output (0 if skipped).
        output_path: Path to the scored output file.
        manifest:    The manifest written (or read, if skipped).
        skipped:     True if the job was skipped (already complete).
    """

    job_id: str
    n_rows: int
    output_path: Path
    manifest: BatchJobManifest
    skipped: bool


class ManifestStore:
    """Reads and writes BatchJobManifest records as JSON files.

    One manifest per job_id, stored as {root_dir}/{job_id}.json.

    Args:
        root_dir: Directory to store manifest JSON files.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        return self.root_dir / f"{job_id}.json"

    def write(self, manifest: BatchJobManifest) -> None:
        """Persist a manifest to disk."""
        self._path(manifest.job_id).write_text(
            json.dumps(manifest.to_dict(), indent=2)
        )
        log.info("Manifest written: job_id=%s status=%s", manifest.job_id, manifest.status)

    def read(self, job_id: str) -> BatchJobManifest | None:
        """Return the manifest for job_id, or None if not found."""
        p = self._path(job_id)
        if not p.exists():
            return None
        return BatchJobManifest.from_dict(json.loads(p.read_text()))

    def is_complete(self, job_id: str) -> bool:
        """Return True if a completed manifest exists for job_id."""
        manifest = self.read(job_id)
        return manifest is not None and manifest.status == "completed"

    def list_all(self) -> list[BatchJobManifest]:
        """Return all manifests in the store."""
        manifests = []
        for p in sorted(self.root_dir.glob("*.json")):
            try:
                manifests.append(BatchJobManifest.from_dict(json.loads(p.read_text())))
            except (json.JSONDecodeError, KeyError):
                log.warning("Skipping corrupt manifest file: %s", p)
        return manifests


class BatchInferenceJob:
    """Scores a DataFrame in chunks, writes output, and tracks a manifest.

    Idempotency:
        Before running, checks if a completed manifest exists for this job_id.
        If yes, returns immediately (skipped=True) without re-scoring.

    Args:
        job_id:           Unique identifier for this job run.
        model_version:    Model version tag (stored in manifest).
        manifest_store:   ManifestStore to check/write manifests.
        chunk_size:       Number of rows per inference batch.
        data_partition:   Optional partition label (e.g. "2024-01").
    """

    def __init__(
        self,
        job_id: str,
        model_version: str,
        manifest_store: ManifestStore,
        *,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        data_partition: str = "default",
    ) -> None:
        self.job_id = job_id
        self.model_version = model_version
        self.manifest_store = manifest_store
        self.chunk_size = chunk_size
        self.data_partition = data_partition

    def run(
        self,
        features_df: pd.DataFrame,
        output_path: Path,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
        *,
        force: bool = False,
    ) -> BatchJobResult:
        """Score features_df and write results to output_path.

        Args:
            features_df:  Input feature DataFrame to score.
            output_path:  Destination path for scored output (CSV).
            predict_fn:   Callable(DataFrame) → np.ndarray of probabilities.
            force:        If True, skip the idempotency check and re-run.

        Returns:
            BatchJobResult (skipped=True if already complete and force=False).
        """
        output_path = Path(output_path)

        # Idempotency check
        if not force and self.manifest_store.is_complete(self.job_id):
            existing = self.manifest_store.read(self.job_id)
            log.info(
                "Skipping job %s — already completed (%d rows)",
                self.job_id, existing.n_rows_scored,
            )
            return BatchJobResult(
                job_id=self.job_id,
                n_rows=0,
                output_path=output_path,
                manifest=existing,
                skipped=True,
            )

        started_at = _utc_now()
        log.info(
            "Starting batch job: job_id=%s partition=%s model=%s n_rows=%d",
            self.job_id, self.data_partition, self.model_version, len(features_df),
        )

        scored_df = self._chunk_and_score(features_df, predict_fn)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored_df.to_csv(output_path, index=False)
        checksum = _checksum_file(output_path)

        completed_at = _utc_now()
        manifest = BatchJobManifest(
            job_id=self.job_id,
            model_version=self.model_version,
            data_partition=self.data_partition,
            n_rows_scored=len(scored_df),
            output_path=str(output_path.absolute()),
            output_checksum=checksum,
            status="completed",
            started_at=started_at,
            completed_at=completed_at,
        )
        self.manifest_store.write(manifest)

        log.info(
            "Batch job complete: job_id=%s n_rows=%d checksum=%s",
            self.job_id, len(scored_df), checksum[:12],
        )
        return BatchJobResult(
            job_id=self.job_id,
            n_rows=len(scored_df),
            output_path=output_path,
            manifest=manifest,
            skipped=False,
        )

    def _chunk_and_score(
        self,
        df: pd.DataFrame,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
    ) -> pd.DataFrame:
        """Score df in chunks; return DataFrame with original columns + score."""
        score_chunks: list[np.ndarray] = []

        for start in range(0, len(df), self.chunk_size):
            chunk = df.iloc[start: start + self.chunk_size]
            scores = predict_fn(chunk)
            score_chunks.append(np.asarray(scores))

        all_scores = np.concatenate(score_chunks) if score_chunks else np.array([])
        result = df.copy()
        result["score"] = all_scores
        result["label"] = (all_scores >= 0.5).astype(int)
        result["model_version"] = self.model_version
        return result


def plan_backfill(
    partitions: list[str],
    model_version: str,
    manifest_store: ManifestStore,
    *,
    job_prefix: str = "backfill",
) -> dict[str, str]:
    """Plan a backfill by identifying which partitions still need scoring.

    Args:
        partitions:      All partition labels to include in the backfill.
        model_version:   New model version to score with.
        manifest_store:  Store to check which partitions are already done.
        job_prefix:      Prefix for job_id construction.

    Returns:
        Dict mapping partition → "completed" | "pending".
    """
    status: dict[str, str] = {}
    for partition in partitions:
        job_id = f"{job_prefix}_{model_version}_{partition}"
        if manifest_store.is_complete(job_id):
            status[partition] = "completed"
        else:
            status[partition] = "pending"
    return status
