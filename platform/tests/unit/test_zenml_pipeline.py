"""Tests for pipelines/zenml_pipeline.py — ZenML-style pipeline."""
from __future__ import annotations

import pytest

from pipelines.zenml_pipeline import (
    ArtifactMeta,
    ArtifactStore,
    CachePolicy,
    PipelineRunResult,
    StackConfig,
    StepDef,
    StepOutput,
    ZenPipeline,
    _hash_value,
    build_credit_risk_pipeline,
)


# ── StackConfig ────────────────────────────────────────────────────────────────

class TestStackConfig:
    def test_local_defaults(self) -> None:
        cfg = StackConfig.local()
        assert cfg.name == "local-dev"
        assert cfg.orchestrator == "local"

    def test_production_s3(self) -> None:
        cfg = StackConfig.production("s3://my-bucket/artifacts")
        assert "s3://" in cfg.artifact_uri
        assert cfg.orchestrator == "kubeflow"

    def test_defaults(self) -> None:
        cfg = StackConfig()
        assert cfg.name == "local-dev"
        assert "artifacts" in cfg.artifact_uri


# ── _hash_value ────────────────────────────────────────────────────────────────

class TestHashValue:
    def test_same_input_same_hash(self) -> None:
        assert _hash_value(42) == _hash_value(42)

    def test_different_inputs_different_hash(self) -> None:
        assert _hash_value(1) != _hash_value(2)

    def test_dict_hashable(self) -> None:
        h = _hash_value({"a": 1, "b": 2})
        assert len(h) == 8

    def test_non_json_hashable(self) -> None:
        h = _hash_value(object())
        assert len(h) == 8


# ── CachePolicy ────────────────────────────────────────────────────────────────

class TestCachePolicy:
    def test_compute_key_deterministic(self) -> None:
        policy = CachePolicy()
        k1 = policy.compute_key("train", "def fn(): pass", ["hash1"], "cfg1")
        k2 = policy.compute_key("train", "def fn(): pass", ["hash1"], "cfg1")
        assert k1 == k2

    def test_different_source_different_key(self) -> None:
        policy = CachePolicy()
        k1 = policy.compute_key("train", "def fn(): pass", ["h1"], "cfg")
        k2 = policy.compute_key("train", "def fn(): return 1", ["h1"], "cfg")
        assert k1 != k2

    def test_different_inputs_different_key(self) -> None:
        policy = CachePolicy()
        k1 = policy.compute_key("train", "src", ["hash1"], "cfg")
        k2 = policy.compute_key("train", "src", ["hash2"], "cfg")
        assert k1 != k2

    def test_key_is_16_chars(self) -> None:
        policy = CachePolicy()
        k = policy.compute_key("train", "src", [], "")
        assert len(k) == 16


# ── ArtifactStore ──────────────────────────────────────────────────────────────

class TestArtifactStore:
    def test_save_and_load(self, tmp_path) -> None:
        store = ArtifactStore(str(tmp_path))
        meta = store.save({"auc": 0.78}, "pipe", "train", "metrics")
        loaded = store.load(meta.uri)
        assert loaded["auc"] == 0.78

    def test_save_creates_meta(self, tmp_path) -> None:
        store = ArtifactStore(str(tmp_path))
        meta = store.save(42, "pipe", "step", "value")
        assert meta.artifact_id is not None
        assert meta.pipeline_name == "pipe"
        assert meta.step_name == "step"

    def test_cache_key_stored_in_index(self, tmp_path) -> None:
        store = ArtifactStore(str(tmp_path))
        store.save(99, "pipe", "step", "out", cache_key="key123")
        assert store.lookup_cache("key123") is not None

    def test_lookup_cache_miss(self, tmp_path) -> None:
        store = ArtifactStore(str(tmp_path))
        assert store.lookup_cache("nonexistent") is None

    def test_save_non_json_value(self, tmp_path) -> None:
        store = ArtifactStore(str(tmp_path))
        meta = store.save(lambda: None, "pipe", "step", "fn")
        assert meta is not None  # should not raise


# ── StepDef ───────────────────────────────────────────────────────────────────

class TestStepDef:
    def _make_store(self, tmp_path) -> ArtifactStore:
        return ArtifactStore(str(tmp_path))

    def test_execute_returns_outputs(self, tmp_path) -> None:
        def fn(inputs, stack):
            return {"result": 42}

        step = StepDef("my_step", fn, ["result"])
        store = self._make_store(tmp_path)
        outputs = step.execute({}, StackConfig.local(), store, "test_pipe")
        assert "result" in outputs
        assert outputs["result"].value == 42

    def test_cache_hit_skips_fn(self, tmp_path) -> None:
        call_count = {"n": 0}

        def fn(inputs, stack):
            call_count["n"] += 1
            return {"val": 1}

        step = StepDef("cached_step", fn, ["val"], CachePolicy(enabled=True))
        store = self._make_store(tmp_path)
        # First run — populates cache
        step.execute({"x": 1}, StackConfig.local(), store, "pipe")
        assert call_count["n"] == 1
        # Second run — same inputs → cache hit
        step.execute({"x": 1}, StackConfig.local(), store, "pipe")
        assert call_count["n"] == 1  # not called again

    def test_cache_disabled_always_runs(self, tmp_path) -> None:
        call_count = {"n": 0}

        def fn(inputs, stack):
            call_count["n"] += 1
            return {"val": 1}

        step = StepDef("step", fn, ["val"], CachePolicy(enabled=False))
        store = self._make_store(tmp_path)
        step.execute({}, StackConfig.local(), store, "pipe")
        step.execute({}, StackConfig.local(), store, "pipe")
        assert call_count["n"] == 2

    def test_from_cache_flag_set(self, tmp_path) -> None:
        def fn(inputs, stack):
            return {"v": 1}

        step = StepDef("s", fn, ["v"])
        store = self._make_store(tmp_path)
        step.execute({}, StackConfig.local(), store, "pipe")
        outputs = step.execute({}, StackConfig.local(), store, "pipe")
        assert outputs["v"].from_cache is True

    def test_different_inputs_invalidate_cache(self, tmp_path) -> None:
        call_count = {"n": 0}

        def fn(inputs, stack):
            call_count["n"] += 1
            return {"val": inputs.get("x", 0)}

        step = StepDef("s", fn, ["val"])
        store = self._make_store(tmp_path)
        step.execute({"x": 1}, StackConfig.local(), store, "pipe")
        step.execute({"x": 2}, StackConfig.local(), store, "pipe")
        assert call_count["n"] == 2


# ── ZenPipeline ───────────────────────────────────────────────────────────────

class TestZenPipeline:
    def _make_pipeline(self, tmp_path) -> ZenPipeline:
        stack = StackConfig(artifact_uri=str(tmp_path))
        return ZenPipeline("test_pipe", stack=stack)

    def test_run_all_steps_succeed(self, tmp_path) -> None:
        pipe = self._make_pipeline(tmp_path)
        pipe.add_step(StepDef("a", lambda i, s: {"x": 1}, ["x"]))
        pipe.add_step(StepDef("b", lambda i, s: {"y": i["x"] + 1}, ["y"]))
        result = pipe.run()
        assert result.succeeded
        assert result.failed_step is None

    def test_step_receives_upstream_outputs(self, tmp_path) -> None:
        pipe = self._make_pipeline(tmp_path)
        pipe.add_step(StepDef("a", lambda i, s: {"msg": "hello"}, ["msg"]))
        pipe.add_step(StepDef("b", lambda i, s: {"echo": i["msg"]}, ["echo"]))
        result = pipe.run()
        assert result.succeeded
        assert result.step_outputs["b"]["echo"].value == "hello"

    def test_run_fails_on_step_error(self, tmp_path) -> None:
        pipe = self._make_pipeline(tmp_path)
        pipe.add_step(StepDef("bad", lambda i, s: (_ for _ in ()).throw(ValueError("oops")), ["x"]))
        result = pipe.run()
        assert not result.succeeded
        assert result.failed_step == "bad"
        assert "oops" in result.error

    def test_run_with_initial_inputs(self, tmp_path) -> None:
        pipe = self._make_pipeline(tmp_path)
        pipe.add_step(StepDef("use_input", lambda i, s: {"doubled": i["value"] * 2}, ["doubled"]))
        result = pipe.run(initial_inputs={"value": 5})
        assert result.step_outputs["use_input"]["doubled"].value == 10

    def test_cached_steps_list(self, tmp_path) -> None:
        pipe = self._make_pipeline(tmp_path)
        pipe.add_step(StepDef("a", lambda i, s: {"x": 1}, ["x"]))
        # Run twice
        pipe.run()
        result = pipe.run()
        assert "a" in result.cached_steps

    def test_run_id_in_result(self, tmp_path) -> None:
        pipe = self._make_pipeline(tmp_path)
        result = pipe.run(run_id="fixed")
        assert result.run_id == "fixed"

    def test_chaining_add_step(self, tmp_path) -> None:
        pipe = self._make_pipeline(tmp_path)
        returned = pipe.add_step(StepDef("a", lambda i, s: {}, []))
        assert returned is pipe


# ── build_credit_risk_pipeline ─────────────────────────────────────────────────

class TestCreditRiskZenPipeline:
    def test_pipeline_runs_successfully(self, tmp_path) -> None:
        stack = StackConfig(artifact_uri=str(tmp_path))
        pipeline = build_credit_risk_pipeline(stack=stack)
        result = pipeline.run(initial_inputs={"n_rows": 200, "auc_threshold": 0.01})
        assert result.succeeded

    def test_pipeline_promotes_on_low_threshold(self, tmp_path) -> None:
        stack = StackConfig(artifact_uri=str(tmp_path))
        pipeline = build_credit_risk_pipeline(stack=stack)
        result = pipeline.run(initial_inputs={"n_rows": 200, "auc_threshold": 0.01})
        assert result.succeeded
        promote_out = result.step_outputs.get("promote", {})
        assert promote_out.get("promoted") is not None

    def test_pipeline_fails_on_high_threshold(self, tmp_path) -> None:
        stack = StackConfig(artifact_uri=str(tmp_path))
        pipeline = build_credit_risk_pipeline(stack=stack)
        result = pipeline.run(initial_inputs={"n_rows": 200, "auc_threshold": 0.9999})
        assert not result.succeeded
        assert result.failed_step == "promote"

    def test_pipeline_name(self, tmp_path) -> None:
        stack = StackConfig(artifact_uri=str(tmp_path))
        pipeline = build_credit_risk_pipeline(stack=stack)
        assert pipeline.name == "credit_risk_zenml"

    def test_pipeline_step_count(self, tmp_path) -> None:
        stack = StackConfig(artifact_uri=str(tmp_path))
        pipeline = build_credit_risk_pipeline(stack=stack)
        assert len(pipeline._steps) == 6

    def test_second_run_uses_cache(self, tmp_path) -> None:
        stack = StackConfig(artifact_uri=str(tmp_path))
        pipeline = build_credit_risk_pipeline(stack=stack)
        inputs = {"n_rows": 200, "auc_threshold": 0.01}
        pipeline.run(initial_inputs=inputs)
        result = pipeline.run(initial_inputs=inputs)
        assert len(result.cached_steps) > 0
