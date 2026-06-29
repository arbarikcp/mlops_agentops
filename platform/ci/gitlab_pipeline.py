"""GitLab CI pipeline definition builder.

Day 56 — generates .gitlab-ci.yml-compatible pipeline configs as Python
dicts so they can be validated, merged, and serialized to YAML without
requiring the PyYAML library in tests.

Classes:
  CacheConfig     — per-job cache specification (key, paths, policy)
  ArtifactConfig  — artifact paths, expiry, and JUnit report path
  GitLabJob       — one job definition (stage, image, script, rules, cache, artifacts)
  GitLabPipeline  — ordered list of jobs; builds the full pipeline dict

See: docs/phase8/day56_gitlab_ci.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── CacheConfig ───────────────────────────────────────────────────────────────

@dataclass
class CacheConfig:
    """GitLab CI cache block.

    Attributes:
        key:    Cache key string (may include CI_COMMIT_REF_SLUG etc.).
        paths:  Directories to cache.
        policy: "pull-push" (default), "pull" (read-only), or "push".
    """

    key: str
    paths: list[str] = field(default_factory=list)
    policy: str = "pull-push"

    def __post_init__(self) -> None:
        valid = {"pull-push", "pull", "push"}
        if self.policy not in valid:
            raise ValueError(f"policy must be one of {valid}; got {self.policy!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "paths": self.paths, "policy": self.policy}


# ── ArtifactConfig ────────────────────────────────────────────────────────────

@dataclass
class ArtifactConfig:
    """GitLab CI artifacts block.

    Attributes:
        paths:      Files/dirs to persist as downloadable artifacts.
        expire_in:  Expiry string, e.g. "7 days", "30 days" (default "7 days").
        junit_path: Optional JUnit XML path for GitLab test report integration.
    """

    paths: list[str] = field(default_factory=list)
    expire_in: str = "7 days"
    junit_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "paths": self.paths,
            "expire_in": self.expire_in,
        }
        if self.junit_path:
            d["reports"] = {"junit": self.junit_path}
        return d


# ── GitLabJob ─────────────────────────────────────────────────────────────────

@dataclass
class GitLabJob:
    """One GitLab CI job definition.

    Attributes:
        name:      Job name (key in .gitlab-ci.yml).
        stage:     Pipeline stage this job belongs to.
        script:    Shell commands to run (each item = one line).
        image:     Docker image override (empty = inherit default).
        cache:     Optional CacheConfig.
        artifacts: Optional ArtifactConfig.
        rules:     List of rule dicts ({"if": ..., "when": ..., "changes": ...}).
        needs:     Names of jobs this job explicitly depends on (DAG mode).
        parallel:  Number of parallel job instances (0 = not set).
    """

    name: str
    stage: str
    script: list[str]
    image: str = ""
    cache: CacheConfig | None = None
    artifacts: ArtifactConfig | None = None
    rules: list[dict[str, Any]] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    parallel: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GitLabJob.name cannot be empty")
        if not self.stage:
            raise ValueError("GitLabJob.stage cannot be empty")
        if not self.script:
            raise ValueError("GitLabJob.script cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "stage": self.stage,
            "script": self.script,
        }
        if self.image:
            d["image"] = self.image
        if self.cache:
            d["cache"] = self.cache.to_dict()
        if self.artifacts:
            d["artifacts"] = self.artifacts.to_dict()
        if self.rules:
            d["rules"] = self.rules
        if self.needs:
            d["needs"] = self.needs
        if self.parallel > 1:
            d["parallel"] = self.parallel
        return d


# ── GitLabPipeline ────────────────────────────────────────────────────────────

class GitLabPipeline:
    """Builds a GitLab CI pipeline configuration dict.

    A pipeline is an ordered list of stages plus a set of jobs.
    Use `to_dict()` to produce the full .gitlab-ci.yml structure.

    Usage::

        pipeline = GitLabPipeline(
            stages=["validate", "test", "build", "deploy"],
            default_image="python:3.11-slim",
        )
        pipeline.add_job(GitLabJob("lint", "validate", ["ruff check ."]))
        config = pipeline.to_dict()
    """

    def __init__(
        self,
        stages: list[str],
        default_image: str = "python:3.11-slim",
        variables: dict[str, str] | None = None,
    ) -> None:
        if not stages:
            raise ValueError("stages cannot be empty")
        self.stages = stages
        self.default_image = default_image
        self.variables: dict[str, str] = variables or {}
        self._jobs: list[GitLabJob] = []

    def add_job(self, job: GitLabJob) -> None:
        """Register a job. Raises if the job's stage is not in pipeline stages."""
        if job.stage not in self.stages:
            raise ValueError(f"job {job.name!r} uses unknown stage {job.stage!r}")
        self._jobs.append(job)

    def job_names(self) -> list[str]:
        return [j.name for j in self._jobs]

    def jobs_for_stage(self, stage: str) -> list[GitLabJob]:
        return [j for j in self._jobs if j.stage == stage]

    def __len__(self) -> int:
        return len(self._jobs)

    def to_dict(self) -> dict[str, Any]:
        """Build the full .gitlab-ci.yml dict."""
        config: dict[str, Any] = {
            "stages": self.stages,
            "default": {"image": self.default_image},
        }
        if self.variables:
            config["variables"] = self.variables
        for job in self._jobs:
            config[job.name] = job.to_dict()
        return config

    @staticmethod
    def ml_pipeline(
        registry_image: str = "${CI_REGISTRY_IMAGE}",
        mlflow_uri: str = "http://mlflow:5000",
    ) -> "GitLabPipeline":
        """Return a canonical ML platform GitLab CI pipeline config.

        Stages: validate → test → model → build → deploy
        Jobs:
          validate: lint
          test:     unit-tests, data-ci
          model:    model-ci
          build:    build-image
          deploy:   deploy-staging, promote-prod
        """
        pipeline = GitLabPipeline(
            stages=["validate", "test", "model", "build", "deploy"],
            default_image="python:3.11-slim",
            variables={
                "UV_CACHE_DIR": ".uv-cache",
                "MLFLOW_TRACKING_URI": mlflow_uri,
            },
        )

        py_cache = CacheConfig(
            key="${CI_COMMIT_REF_SLUG}-py",
            paths=[".uv-cache/", ".venv/"],
            policy="pull-push",
        )

        pipeline.add_job(GitLabJob(
            name="lint",
            stage="validate",
            script=[
                "pip install uv",
                "uv run ruff check platform/",
                "uv run mypy platform/ --ignore-missing-imports",
            ],
            cache=py_cache,
            rules=[{"when": "always"}],
        ))

        pipeline.add_job(GitLabJob(
            name="unit-tests",
            stage="test",
            script=["uv run pytest platform/tests/unit/ -v --junit-xml=reports/junit.xml"],
            cache=py_cache,
            artifacts=ArtifactConfig(paths=["reports/"], junit_path="reports/junit.xml"),
            needs=["lint"],
        ))

        pipeline.add_job(GitLabJob(
            name="data-ci",
            stage="test",
            script=["uv run python -m ci.data_contract_check"],
            cache=py_cache,
            artifacts=ArtifactConfig(paths=["reports/data_ci_report.json"]),
            rules=[
                {"if": '$CI_PIPELINE_SOURCE == "schedule"'},
                {"changes": ["data/**/*"]},
            ],
            needs=["lint"],
        ))

        pipeline.add_job(GitLabJob(
            name="model-ci",
            stage="model",
            script=[
                "uv run python -m ci.smoke_train_check",
                "uv run python -m ci.auc_guard_check",
            ],
            cache=py_cache,
            artifacts=ArtifactConfig(
                paths=["reports/model_ci_report.json", "artifacts/baseline_auc.json"],
                expire_in="30 days",
            ),
            rules=[{"changes": ["training/**/*", "ci/**/*"]}],
            needs=["unit-tests"],
        ))

        pipeline.add_job(GitLabJob(
            name="build-image",
            stage="build",
            image="docker:24",
            script=[
                f"docker build -t {registry_image}:$CI_COMMIT_SHA .",
                f"docker push {registry_image}:$CI_COMMIT_SHA",
            ],
            rules=[{"if": '$CI_COMMIT_BRANCH == "main"'}],
            needs=["unit-tests", "model-ci"],
        ))

        pipeline.add_job(GitLabJob(
            name="deploy-staging",
            stage="deploy",
            script=[
                "helm upgrade --install ml-api ./infra/helm/ml-api"
                f" --set image.tag=$CI_COMMIT_SHA --set environment=staging",
            ],
            rules=[{"if": '$CI_COMMIT_BRANCH == "main"'}],
            needs=["build-image"],
        ))

        pipeline.add_job(GitLabJob(
            name="promote-prod",
            stage="deploy",
            script=[
                "helm upgrade --install ml-api ./infra/helm/ml-api"
                f" --set image.tag=$CI_COMMIT_SHA --set environment=production",
            ],
            rules=[{"if": '$CI_COMMIT_BRANCH == "main"', "when": "manual"}],
            needs=["deploy-staging"],
        ))

        return pipeline
