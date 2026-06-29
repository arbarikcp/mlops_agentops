"""Tests for ci/gitlab_pipeline.py — CacheConfig, ArtifactConfig, GitLabJob, GitLabPipeline."""
from __future__ import annotations

import pytest

from ci.gitlab_pipeline import (
    ArtifactConfig,
    CacheConfig,
    GitLabJob,
    GitLabPipeline,
)


# ── CacheConfig ────────────────────────────────────────────────────────────────

class TestCacheConfig:
    def test_defaults(self) -> None:
        c = CacheConfig(key="test")
        assert c.policy == "pull-push"
        assert c.paths == []

    def test_invalid_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="policy"):
            CacheConfig(key="x", policy="invalid")

    def test_to_dict(self) -> None:
        c = CacheConfig(key="branch-py", paths=[".venv/"], policy="pull")
        d = c.to_dict()
        assert d["key"] == "branch-py"
        assert ".venv/" in d["paths"]
        assert d["policy"] == "pull"

    def test_valid_policies(self) -> None:
        for policy in ("pull-push", "pull", "push"):
            c = CacheConfig(key="k", policy=policy)
            assert c.policy == policy


# ── ArtifactConfig ─────────────────────────────────────────────────────────────

class TestArtifactConfig:
    def test_defaults(self) -> None:
        a = ArtifactConfig()
        assert a.expire_in == "7 days"
        assert a.paths == []
        assert a.junit_path == ""

    def test_to_dict_no_junit(self) -> None:
        a = ArtifactConfig(paths=["reports/"], expire_in="30 days")
        d = a.to_dict()
        assert d["paths"] == ["reports/"]
        assert "reports" not in d

    def test_to_dict_with_junit(self) -> None:
        a = ArtifactConfig(paths=["reports/"], junit_path="reports/junit.xml")
        d = a.to_dict()
        assert d["reports"]["junit"] == "reports/junit.xml"


# ── GitLabJob ──────────────────────────────────────────────────────────────────

class TestGitLabJob:
    def _job(self, **kw) -> GitLabJob:
        defaults = dict(name="lint", stage="validate", script=["ruff check ."])
        return GitLabJob(**{**defaults, **kw})

    def test_basic_to_dict(self) -> None:
        d = self._job().to_dict()
        assert d["stage"] == "validate"
        assert "ruff check ." in d["script"]

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            GitLabJob(name="", stage="validate", script=["x"])

    def test_empty_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="stage"):
            GitLabJob(name="lint", stage="", script=["x"])

    def test_empty_script_raises(self) -> None:
        with pytest.raises(ValueError, match="script"):
            GitLabJob(name="lint", stage="validate", script=[])

    def test_image_included_when_set(self) -> None:
        d = self._job(image="docker:24").to_dict()
        assert d["image"] == "docker:24"

    def test_image_omitted_when_empty(self) -> None:
        d = self._job().to_dict()
        assert "image" not in d

    def test_cache_in_dict(self) -> None:
        c = CacheConfig(key="k", paths=[".venv/"])
        d = self._job(cache=c).to_dict()
        assert "cache" in d
        assert d["cache"]["key"] == "k"

    def test_artifacts_in_dict(self) -> None:
        a = ArtifactConfig(paths=["reports/"])
        d = self._job(artifacts=a).to_dict()
        assert "artifacts" in d

    def test_rules_in_dict(self) -> None:
        rules = [{"when": "always"}]
        d = self._job(rules=rules).to_dict()
        assert d["rules"] == rules

    def test_needs_in_dict(self) -> None:
        d = self._job(needs=["lint"]).to_dict()
        assert d["needs"] == ["lint"]

    def test_parallel_in_dict(self) -> None:
        d = self._job(parallel=3).to_dict()
        assert d["parallel"] == 3

    def test_parallel_omitted_when_zero(self) -> None:
        d = self._job(parallel=0).to_dict()
        assert "parallel" not in d


# ── GitLabPipeline ─────────────────────────────────────────────────────────────

class TestGitLabPipeline:
    def _pipeline(self) -> GitLabPipeline:
        return GitLabPipeline(stages=["validate", "test", "deploy"])

    def test_empty_stages_raises(self) -> None:
        with pytest.raises(ValueError, match="stages"):
            GitLabPipeline(stages=[])

    def test_add_job(self) -> None:
        p = self._pipeline()
        p.add_job(GitLabJob("lint", "validate", ["ruff ."]))
        assert len(p) == 1

    def test_unknown_stage_raises(self) -> None:
        p = self._pipeline()
        with pytest.raises(ValueError, match="unknown stage"):
            p.add_job(GitLabJob("job", "nonexistent", ["x"]))

    def test_jobs_for_stage(self) -> None:
        p = self._pipeline()
        p.add_job(GitLabJob("lint", "validate", ["x"]))
        p.add_job(GitLabJob("unit", "test", ["x"]))
        p.add_job(GitLabJob("type", "validate", ["x"]))
        assert len(p.jobs_for_stage("validate")) == 2
        assert len(p.jobs_for_stage("test")) == 1

    def test_to_dict_structure(self) -> None:
        p = self._pipeline()
        p.add_job(GitLabJob("lint", "validate", ["x"]))
        d = p.to_dict()
        assert "stages" in d
        assert "default" in d
        assert "lint" in d

    def test_to_dict_variables(self) -> None:
        p = GitLabPipeline(stages=["test"], variables={"FOO": "bar"})
        d = p.to_dict()
        assert d["variables"]["FOO"] == "bar"

    def test_job_names(self) -> None:
        p = self._pipeline()
        p.add_job(GitLabJob("a", "validate", ["x"]))
        p.add_job(GitLabJob("b", "test", ["x"]))
        assert p.job_names() == ["a", "b"]

    def test_ml_pipeline_factory(self) -> None:
        p = GitLabPipeline.ml_pipeline()
        assert len(p) == 7  # lint, unit-tests, data-ci, model-ci, build-image, deploy-staging, promote-prod
        names = p.job_names()
        assert "lint" in names
        assert "model-ci" in names
        assert "promote-prod" in names

    def test_ml_pipeline_stages(self) -> None:
        p = GitLabPipeline.ml_pipeline()
        d = p.to_dict()
        assert d["stages"] == ["validate", "test", "model", "build", "deploy"]

    def test_ml_pipeline_model_ci_has_artifacts(self) -> None:
        p = GitLabPipeline.ml_pipeline()
        d = p.to_dict()
        assert "artifacts" in d["model-ci"]
        assert "30 days" in d["model-ci"]["artifacts"]["expire_in"]

    def test_ml_pipeline_promote_prod_manual(self) -> None:
        p = GitLabPipeline.ml_pipeline()
        d = p.to_dict()
        rules = d["promote-prod"]["rules"]
        assert any(r.get("when") == "manual" for r in rules)
