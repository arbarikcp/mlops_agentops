"""ZenML-style ML pipeline with step caching and artifact versioning.

Implements ZenML patterns without requiring `zenml` to be installed:
  - StepDef:         Typed step with cache policy (@step equivalent)
  - CachePolicy:     Hash-based input caching to skip unchanged steps
  - NativeMaterializer: JSON-based artifact serialisation
  - ArtifactStore:   File-based versioned artifact storage
  - StackConfig:     Pluggable infra config (artifact URI, tracking URI)
  - ZenPipeline:     Ordered steps with output threading (@pipeline equivalent)

See: docs/phase5/day33_zenml_pipeline.md
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


# ── Stack Config ──────────────────────────────────────────────────────────────

@dataclass
class StackConfig:
    """Infrastructure configuration for a ZenML stack.

    Attributes:
        name:           Stack name (e.g. "local-dev", "production").
        artifact_uri:   Root URI for artifact storage (file path or s3://).
        tracking_uri:   MLflow / experiment tracker URI.
        orchestrator:   Orchestrator type ("local", "kubeflow", "airflow").
    """

    name: str = "local-dev"
    artifact_uri: str = ".zenml_artifacts"
    tracking_uri: str = "http://localhost:5000"
    orchestrator: str = "local"

    @classmethod
    def local(cls) -> "StackConfig":
        return cls(name="local-dev", artifact_uri=".zenml_artifacts", orchestrator="local")

    @classmethod
    def production(cls, artifact_uri: str = "s3://my-bucket/artifacts") -> "StackConfig":
        return cls(
            name="production",
            artifact_uri=artifact_uri,
            tracking_uri="http://mlflow.internal:5000",
            orchestrator="kubeflow",
        )


# ── Artifact Store ─────────────────────────────────────────────────────────────

@dataclass
class ArtifactMeta:
    """Metadata record for a stored artifact."""

    artifact_id: str
    pipeline_name: str
    step_name: str
    output_name: str
    data_type: str
    uri: str
    created_at: str
    cache_key: str | None = None


class ArtifactStore:
    """File-based versioned artifact store.

    Artifacts are stored as JSON (for simple types) under:
        <root_uri>/<pipeline>/<step>/<output>/<artifact_id>/data.json

    The index file maps cache_key → artifact_id for cache lookups.

    Args:
        root_uri: Root directory for artifact storage.
    """

    def __init__(self, root_uri: str) -> None:
        self.root = Path(root_uri)
        self._index_path = self.root / ".cache_index.json"
        self._index: dict[str, str] = self._load_index()

    def _load_index(self) -> dict[str, str]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _persist_index(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_path.write_text(json.dumps(self._index))

    def save(
        self,
        value: Any,
        pipeline_name: str,
        step_name: str,
        output_name: str,
        cache_key: str | None = None,
    ) -> ArtifactMeta:
        """Serialise and store an artifact.

        Args:
            value:         Python value to store.
            pipeline_name: Name of the pipeline.
            step_name:     Name of the step.
            output_name:   Name of the output variable.
            cache_key:     Hash key for cache lookup.

        Returns:
            ArtifactMeta with URI and ID.
        """
        artifact_id = str(uuid.uuid4())[:8]
        artifact_dir = self.root / pipeline_name / step_name / output_name / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Serialise: attempt JSON, fall back to repr
        try:
            payload = json.dumps({"value": value, "type": type(value).__name__})
        except (TypeError, ValueError):
            payload = json.dumps({"value": repr(value), "type": type(value).__name__})

        data_path = artifact_dir / "data.json"
        data_path.write_text(payload)

        meta = ArtifactMeta(
            artifact_id=artifact_id,
            pipeline_name=pipeline_name,
            step_name=step_name,
            output_name=output_name,
            data_type=type(value).__name__,
            uri=str(data_path),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            cache_key=cache_key,
        )

        # Write metadata
        meta_path = artifact_dir / "meta.json"
        meta_path.write_text(json.dumps(meta.__dict__))

        if cache_key:
            self._index[cache_key] = artifact_id
            self._persist_index()

        log.debug("Artifact saved: %s/%s/%s → %s", pipeline_name, step_name, output_name, artifact_id)
        return meta

    def load(self, uri: str) -> Any:
        """Load an artifact from its URI."""
        payload = json.loads(Path(uri).read_text())
        return payload["value"]

    def lookup_cache(self, cache_key: str) -> str | None:
        """Return artifact_id for a cache_key, or None if not cached."""
        return self._index.get(cache_key)


# ── Cache Policy ─────────────────────────────────────────────────────────────

@dataclass
class CachePolicy:
    """Controls whether a step result can be cached.

    Attributes:
        enabled:   If False, step always runs regardless of cache.
        ttl_hours: Time-to-live in hours; None = never expire.
    """

    enabled: bool = True
    ttl_hours: float | None = None

    def compute_key(
        self,
        step_name: str,
        step_source: str,
        input_hashes: list[str],
        config_hash: str,
    ) -> str:
        """Compute cache key from step identity and inputs.

        Key = SHA-256(step_name + source_code + sorted(input_hashes) + config)
        Changing any of these busts the cache for this step.
        """
        material = "|".join([
            step_name,
            step_source,
            ",".join(sorted(input_hashes)),
            config_hash,
        ])
        return hashlib.sha256(material.encode()).hexdigest()[:16]


def _hash_value(value: Any) -> str:
    """Stable content hash for a Python value (JSON-serialisable)."""
    try:
        content = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        content = repr(value)
    return hashlib.sha256(content.encode()).hexdigest()[:8]


# ── Step Definition ───────────────────────────────────────────────────────────

@dataclass
class StepOutput:
    """Typed output record from a ZenPipeline step execution."""

    step_name: str
    output_name: str
    value: Any
    artifact_meta: ArtifactMeta | None = None
    from_cache: bool = False
    duration_s: float = 0.0


@dataclass
class StepDef:
    """A typed pipeline step — analogous to a ZenML @step.

    Attributes:
        name:         Step name (unique in pipeline).
        fn:           Function (inputs: dict, stack: StackConfig) → dict[output_name, value].
        output_names: Names of the keys returned by fn.
        cache_policy: Cache policy for this step.
        depends_on:   Names of steps whose outputs this step requires.
    """

    name: str
    fn: Callable[[dict[str, Any], StackConfig], dict[str, Any]]
    output_names: list[str]
    cache_policy: CachePolicy = field(default_factory=CachePolicy)
    depends_on: list[str] = field(default_factory=list)

    def execute(
        self,
        inputs: dict[str, Any],
        stack: StackConfig,
        artifact_store: ArtifactStore,
        pipeline_name: str,
        config_hash: str = "",
    ) -> dict[str, StepOutput]:
        """Execute this step, using cache if possible.

        Args:
            inputs:         dict of named input values from upstream steps.
            stack:          Active stack configuration.
            artifact_store: Artifact store for caching.
            pipeline_name:  Pipeline name (for artifact paths).
            config_hash:    Hash of pipeline config for cache key.

        Returns:
            dict mapping output_name → StepOutput.
        """
        # Compute cache key
        input_hashes = [_hash_value(v) for v in inputs.values()]
        source = inspect.getsource(self.fn)
        cache_key = self.cache_policy.compute_key(
            self.name, source, input_hashes, config_hash
        )

        # Check cache
        if self.cache_policy.enabled:
            cached_id = artifact_store.lookup_cache(cache_key)
            if cached_id is not None:
                log.info("Step %r: cache HIT (key=%s)", self.name, cache_key)
                return {
                    name: StepOutput(
                        step_name=self.name,
                        output_name=name,
                        value=None,   # not loaded unless downstream needs it
                        from_cache=True,
                    )
                    for name in self.output_names
                }

        # Execute
        log.info("Step %r: running (inputs=%s)", self.name, list(inputs.keys()))
        start = time.monotonic()
        result = self.fn(inputs, stack)
        duration = time.monotonic() - start

        outputs: dict[str, StepOutput] = {}
        for out_name in self.output_names:
            value = result.get(out_name)
            meta = artifact_store.save(
                value=value,
                pipeline_name=pipeline_name,
                step_name=self.name,
                output_name=out_name,
                cache_key=cache_key if out_name == self.output_names[0] else None,
            )
            outputs[out_name] = StepOutput(
                step_name=self.name,
                output_name=out_name,
                value=value,
                artifact_meta=meta,
                from_cache=False,
                duration_s=duration,
            )
        return outputs


# ── ZenPipeline ───────────────────────────────────────────────────────────────

@dataclass
class PipelineRunResult:
    """Outcome of a ZenPipeline run."""

    run_id: str
    pipeline_name: str
    succeeded: bool
    step_outputs: dict[str, dict[str, StepOutput]]
    failed_step: str | None
    error: str | None
    duration_s: float
    cached_steps: list[str]


class ZenPipeline:
    """Ordered list of StepDefs with dependency-based caching.

    Analogous to a ZenML @pipeline. Steps are executed in order; each
    receives the accumulated outputs of all prior steps as its inputs.

    Args:
        name:  Pipeline name.
        stack: StackConfig for this run.
    """

    def __init__(self, name: str, stack: StackConfig | None = None) -> None:
        self.name = name
        self.stack = stack or StackConfig.local()
        self._steps: list[StepDef] = []

    def add_step(self, step: StepDef) -> "ZenPipeline":
        """Register a step. Returns self for chaining."""
        self._steps.append(step)
        return self

    def run(
        self,
        initial_inputs: dict[str, Any] | None = None,
        config_hash: str = "",
        run_id: str | None = None,
    ) -> PipelineRunResult:
        """Execute all steps in order.

        Args:
            initial_inputs: Seed values (e.g. data_path config).
            config_hash:    Hash of run config for cache key computation.
            run_id:         Explicit run ID (auto-generated if None).

        Returns:
            PipelineRunResult with per-step outputs and status.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        artifact_store = ArtifactStore(self.stack.artifact_uri)

        # accumulated outputs from all steps
        accumulated: dict[str, Any] = dict(initial_inputs or {})
        step_outputs: dict[str, dict[str, StepOutput]] = {}
        cached_steps: list[str] = []

        start = time.monotonic()
        log.info("ZenPipeline %r starting (run_id=%s)", self.name, run_id)

        for step in self._steps:
            # Build inputs from accumulated outputs that match expected deps
            step_inputs = {k: v for k, v in accumulated.items()}

            try:
                outputs = step.execute(
                    inputs=step_inputs,
                    stack=self.stack,
                    artifact_store=artifact_store,
                    pipeline_name=self.name,
                    config_hash=config_hash,
                )
                step_outputs[step.name] = outputs
                if any(o.from_cache for o in outputs.values()):
                    cached_steps.append(step.name)

                # Merge outputs into accumulated
                for out_name, step_out in outputs.items():
                    accumulated[out_name] = step_out.value

            except Exception as exc:  # noqa: BLE001
                log.error("Step %r failed: %s", step.name, exc)
                return PipelineRunResult(
                    run_id=run_id,
                    pipeline_name=self.name,
                    succeeded=False,
                    step_outputs=step_outputs,
                    failed_step=step.name,
                    error=str(exc),
                    duration_s=time.monotonic() - start,
                    cached_steps=cached_steps,
                )

        duration = time.monotonic() - start
        log.info("ZenPipeline %r finished in %.2fs (cached: %s)", self.name, duration, cached_steps)

        return PipelineRunResult(
            run_id=run_id,
            pipeline_name=self.name,
            succeeded=True,
            step_outputs=step_outputs,
            failed_step=None,
            error=None,
            duration_s=duration,
            cached_steps=cached_steps,
        )


# ── Credit Risk Pipeline Builder ───────────────────────────────────────────────

def build_credit_risk_pipeline(stack: StackConfig | None = None) -> ZenPipeline:
    """Build the credit-risk training pipeline as a ZenPipeline.

    Steps:
        1. load_config   — emit config values to downstream steps
        2. validate_data — check row count, label column
        3. featurize     — select feature columns, stratified split info
        4. train         — fit model (synthetic for tests), emit AUC
        5. validate_model — gate: AUC >= threshold?
        6. promote       — emit promotion record if passed
    """
    import numpy as np

    def load_config(inputs: dict, stack: StackConfig) -> dict:
        return {
            "n_rows": inputs.get("n_rows", 1000),
            "auc_threshold": inputs.get("auc_threshold", 0.60),
            "n_features": inputs.get("n_features", 5),
        }

    def validate_data(inputs: dict, stack: StackConfig) -> dict:
        n_rows = inputs["n_rows"]
        if n_rows <= 0:
            raise ValueError("Dataset is empty")
        return {"validated_n_rows": n_rows}

    def featurize(inputs: dict, stack: StackConfig) -> dict:
        rng = np.random.default_rng(42)
        n = inputs["validated_n_rows"]
        n_features = inputs["n_features"]
        X = rng.standard_normal((n, n_features))
        y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
        return {"X": X.tolist(), "y": y.tolist()}

    def train(inputs: dict, stack: StackConfig) -> dict:
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        X = np.array(inputs["X"])
        y = np.array(inputs["y"])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
        clf.fit(X_train, y_train)
        auc = float(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
        return {"auc": auc, "model_type": type(clf).__name__}

    def validate_model(inputs: dict, stack: StackConfig) -> dict:
        auc = inputs["auc"]
        threshold = inputs["auc_threshold"]
        passed = auc >= threshold
        return {"gate_passed": passed, "gate_auc": auc, "gate_threshold": threshold}

    def promote(inputs: dict, stack: StackConfig) -> dict:
        if not inputs["gate_passed"]:
            raise ValueError(
                f"Promotion blocked: AUC {inputs['gate_auc']:.4f} < {inputs['gate_threshold']}"
            )
        return {"champion_version": f"v-{time.strftime('%Y%m%d')}", "promoted": True}

    pipeline = ZenPipeline(name="credit_risk_zenml", stack=stack or StackConfig.local())
    (
        pipeline
        .add_step(StepDef("load_config", load_config, ["n_rows", "auc_threshold", "n_features"]))
        .add_step(StepDef("validate_data", validate_data, ["validated_n_rows"]))
        .add_step(StepDef("featurize", featurize, ["X", "y"]))
        .add_step(StepDef("train", train, ["auc", "model_type"]))
        .add_step(StepDef("validate_model", validate_model, ["gate_passed", "gate_auc", "gate_threshold"]))
        .add_step(StepDef("promote", promote, ["champion_version", "promoted"]))
    )
    return pipeline
