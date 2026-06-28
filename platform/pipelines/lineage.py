"""OpenLineage data lineage emission — Day 13 deliverable.

OpenLineage is an open standard for data lineage metadata.
It defines a common event format that lineage backends (Marquez, Atlan,
DataHub, OpenMetadata) can all consume.

Key concepts:
  - Job:         A transformation process (e.g. "featurize", "train")
  - Dataset:     A data asset (e.g. "credit_card_default.csv")
  - Run:         One execution of a Job
  - Event:       START or COMPLETE or FAIL event for a Run
  - Facet:       Structured metadata attached to a Job, Run, or Dataset
    - SchemaDatasetFacet:    column names + types
    - DataQualityDatasetFacet: row count, null counts
    - DocumentationJobFacet: human-readable description

Why lineage matters (from Day 4 system design):
  The M1 Reproducibility Gate requires: "given a prediction, trace the
  model version, data version, and code version."
  OpenLineage closes the data half of that chain:
    raw CSV ──[featurize]──▶ features.parquet ──[train]──▶ model v1

Usage:
    from pipelines.lineage import LineageEmitter
    emitter = LineageEmitter(producer="featurize-job", namespace="credit-risk")
    with emitter.run("featurize"):
        # ... do featurization ...
        emitter.emit_complete(
            inputs=[LineageDataset("credit_card_default.csv", row_count=30000)],
            outputs=[LineageDataset("features.parquet", row_count=30000, n_cols=32)],
        )

Debugging:
    - Set OPENLINEAGE_URL env var to point at Marquez or use the console backend.
    - Use LineageEmitter(backend="console") to print events to stdout (no server needed).
    - View events: marquez_url/api/v1/lineage (if Marquez is running)
    - Integration with MLflow: the mlflow_run_id is embedded as a run facet.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

log = logging.getLogger(__name__)

# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class LineageDataset:
    name: str
    namespace: str = "local"
    row_count: int | None = None
    n_cols: int | None = None
    columns: list[dict[str, str]] = field(default_factory=list)
    null_counts: dict[str, int] = field(default_factory=dict)

    def to_openlineage(self) -> dict[str, Any]:
        facets: dict[str, Any] = {}

        if self.columns:
            facets["schema"] = {
                "_producer": "mlops-platform",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
                "fields": self.columns,
            }

        if self.row_count is not None or self.null_counts:
            facets["dataQuality"] = {
                "_producer": "mlops-platform",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DataQualityDatasetFacet.json",
                "rowCount": self.row_count,
                "columnMetrics": {
                    col: {"nullCount": cnt}
                    for col, cnt in self.null_counts.items()
                },
            }

        return {
            "namespace": self.namespace,
            "name": self.name,
            "facets": facets,
        }


# ── Emitter ───────────────────────────────────────────────────────────────────

class LineageEmitter:
    """Emits OpenLineage events to a configured backend.

    Backends:
        "console"  — print JSON to stdout (development/testing)
        "http"     — POST to OPENLINEAGE_URL (Marquez, etc.)

    The emitter tracks the current run_id (UUID) across START/COMPLETE/FAIL.
    """

    def __init__(
        self,
        producer: str,
        namespace: str = "credit-risk",
        backend: str | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        self.producer = producer
        self.namespace = namespace
        self.backend = backend or os.getenv("OPENLINEAGE_BACKEND", "console")
        self.openlineage_url = os.getenv("OPENLINEAGE_URL", "http://localhost:5001/api/v1/lineage")
        self.mlflow_run_id = mlflow_run_id
        self._run_id: str | None = None
        self._job_name: str | None = None

    @contextmanager
    def run(self, job_name: str) -> Generator[None, None, None]:
        """Context manager: emits START on enter, COMPLETE/FAIL on exit."""
        self._run_id = str(uuid.uuid4())
        self._job_name = job_name
        self._emit("START", job_name, inputs=[], outputs=[])
        try:
            yield
        except Exception as exc:
            self._emit("FAIL", job_name, inputs=[], outputs=[], error=str(exc))
            raise
        # COMPLETE is called explicitly via emit_complete

    def emit_complete(
        self,
        inputs: list[LineageDataset],
        outputs: list[LineageDataset],
    ) -> None:
        assert self._job_name, "call inside .run() context"
        self._emit("COMPLETE", self._job_name, inputs=inputs, outputs=outputs)

    def _emit(
        self,
        event_type: str,
        job_name: str,
        inputs: list[LineageDataset],
        outputs: list[LineageDataset],
        error: str | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "eventType": event_type,
            "eventTime": datetime.datetime.utcnow().isoformat() + "Z",
            "producer": self.producer,
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json",
            "run": {
                "runId": self._run_id,
                "facets": self._run_facets(error),
            },
            "job": {
                "namespace": self.namespace,
                "name": job_name,
                "facets": {
                    "documentation": {
                        "_producer": "mlops-platform",
                        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                        "description": f"Credit risk platform — {job_name}",
                    }
                },
            },
            "inputs": [ds.to_openlineage() for ds in inputs],
            "outputs": [ds.to_openlineage() for ds in outputs],
        }

        if self.backend == "console":
            log.info("OpenLineage [%s] %s/%s", event_type, self.namespace, job_name)
            print(json.dumps(event, indent=2))
        elif self.backend == "http":
            self._post(event)
        else:
            log.debug("OpenLineage event (backend=%s): %s", self.backend, event_type)

    def _run_facets(self, error: str | None) -> dict[str, Any]:
        facets: dict[str, Any] = {}
        if self.mlflow_run_id:
            facets["mlflow"] = {
                "_producer": "mlops-platform",
                "run_id": self.mlflow_run_id,
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            }
        if error:
            facets["errorMessage"] = {
                "_producer": "mlops-platform",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json",
                "message": error,
                "programmingLanguage": "Python",
            }
        return facets

    def _post(self, event: dict[str, Any]) -> None:
        try:
            import requests
            resp = requests.post(
                self.openlineage_url,
                json=event,
                timeout=5,
            )
            resp.raise_for_status()
            log.debug("Lineage event posted: %s %d", event["eventType"], resp.status_code)
        except Exception as exc:
            log.warning("Failed to post lineage event (non-fatal): %s", exc)


# ── Convenience wrappers for our specific pipeline stages ─────────────────────

def emit_featurize_lineage(
    input_path: str,
    output_path: str,
    row_count: int,
    n_features: int,
    mlflow_run_id: str | None = None,
) -> None:
    """Emit lineage for the featurize stage."""
    emitter = LineageEmitter(
        producer="featurize-stage",
        namespace="credit-risk",
        mlflow_run_id=mlflow_run_id,
    )
    with emitter.run("featurize"):
        emitter.emit_complete(
            inputs=[LineageDataset(name=input_path, namespace="local")],
            outputs=[LineageDataset(
                name=output_path,
                namespace="local",
                row_count=row_count,
                n_cols=n_features,
            )],
        )


def emit_train_lineage(
    input_path: str,
    model_path: str,
    metrics_path: str,
    mlflow_run_id: str | None = None,
) -> None:
    """Emit lineage for the train stage."""
    emitter = LineageEmitter(
        producer="train-stage",
        namespace="credit-risk",
        mlflow_run_id=mlflow_run_id,
    )
    with emitter.run("train"):
        emitter.emit_complete(
            inputs=[LineageDataset(name=input_path, namespace="local")],
            outputs=[
                LineageDataset(name=model_path, namespace="local"),
                LineageDataset(name=metrics_path, namespace="local"),
            ],
        )
