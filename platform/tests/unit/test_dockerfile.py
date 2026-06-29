"""Tests for serving/Dockerfile — validate structural security properties.

These tests read the Dockerfile as text and assert key security patterns
without actually running Docker (no Docker daemon required in CI).
"""
from __future__ import annotations

from pathlib import Path

DOCKERFILE = Path(__file__).parent.parent.parent / "serving" / "Dockerfile"
DOCKERIGNORE = Path(__file__).parent.parent.parent / ".dockerignore"


class TestDockerfileStructure:
    def test_dockerfile_exists(self) -> None:
        assert DOCKERFILE.exists(), "Dockerfile not found in serving/"

    def test_multistage_build(self) -> None:
        content = DOCKERFILE.read_text()
        # Multi-stage = at least two FROM statements
        from_lines = [l for l in content.splitlines() if l.strip().upper().startswith("FROM")]
        assert len(from_lines) >= 2, "Expected multi-stage build (≥2 FROM statements)"

    def test_builder_stage_named(self) -> None:
        content = DOCKERFILE.read_text()
        assert "AS builder" in content or "as builder" in content.lower()

    def test_runtime_stage_named(self) -> None:
        content = DOCKERFILE.read_text()
        assert "AS runtime" in content or "as runtime" in content.lower()

    def test_non_root_user(self) -> None:
        content = DOCKERFILE.read_text()
        assert "USER appuser" in content or "useradd" in content

    def test_no_root_user(self) -> None:
        content = DOCKERFILE.read_text()
        # Ensure we don't CMD/ENTRYPOINT as root (USER appuser must appear)
        assert "USER" in content

    def test_exposes_port_8080(self) -> None:
        content = DOCKERFILE.read_text()
        assert "EXPOSE 8080" in content

    def test_no_secrets_in_env(self) -> None:
        content = DOCKERFILE.read_text()
        secret_keywords = ["PASSWORD", "SECRET_KEY", "API_KEY", "TOKEN"]
        for keyword in secret_keywords:
            assert keyword not in content, f"Potential secret found in Dockerfile: {keyword}"

    def test_uses_slim_base_image(self) -> None:
        content = DOCKERFILE.read_text()
        assert "slim" in content.lower(), "Base image should use 'slim' variant for smaller attack surface"

    def test_python_unbuffered_set(self) -> None:
        content = DOCKERFILE.read_text()
        assert "PYTHONUNBUFFERED" in content

    def test_copies_from_builder(self) -> None:
        content = DOCKERFILE.read_text()
        assert "COPY --from=builder" in content

    def test_healthcheck_defined(self) -> None:
        content = DOCKERFILE.read_text()
        assert "HEALTHCHECK" in content

    def test_uvicorn_in_cmd(self) -> None:
        content = DOCKERFILE.read_text()
        assert "uvicorn" in content.lower()


class TestDockerignore:
    def test_dockerignore_exists(self) -> None:
        assert DOCKERIGNORE.exists(), ".dockerignore not found"

    def test_git_excluded(self) -> None:
        content = DOCKERIGNORE.read_text()
        assert ".git/" in content or ".git" in content

    def test_env_files_excluded(self) -> None:
        content = DOCKERIGNORE.read_text()
        assert ".env" in content

    def test_tests_excluded(self) -> None:
        content = DOCKERIGNORE.read_text()
        assert "tests/" in content

    def test_pycache_excluded(self) -> None:
        content = DOCKERIGNORE.read_text()
        assert "__pycache__/" in content

    def test_raw_data_excluded(self) -> None:
        content = DOCKERIGNORE.read_text()
        assert "data/raw/" in content
