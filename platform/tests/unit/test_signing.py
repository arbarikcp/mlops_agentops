"""Tests for ci/signing.py — SBOMEntry, SBOMDocument, ArtifactProvenanceRecord, ArtifactSigner."""
from __future__ import annotations

import json

import pytest

from ci.signing import (
    ArtifactProvenanceRecord,
    ArtifactSigner,
    SBOMDocument,
    SBOMEntry,
    SigningResult,
)

ARTIFACT = b"model weights here"


# ── SBOMEntry ──────────────────────────────────────────────────────────────────

class TestSBOMEntry:
    def test_basic(self) -> None:
        e = SBOMEntry("numpy", "1.26.4", "BSD-3-Clause")
        assert e.name == "numpy"
        assert "numpy" in e.purl

    def test_auto_purl(self) -> None:
        e = SBOMEntry("scikit-learn", "1.4.0")
        assert "scikit-learn" in e.purl.lower()
        assert "1.4.0" in e.purl

    def test_to_dict(self) -> None:
        e = SBOMEntry("numpy", "1.26.4", "BSD-3-Clause")
        d = e.to_dict()
        assert d["name"] == "numpy"
        assert d["version"] == "1.26.4"
        assert d["license"] == "BSD-3-Clause"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            SBOMEntry("", "1.0")

    def test_empty_version_raises(self) -> None:
        with pytest.raises(ValueError, match="version"):
            SBOMEntry("numpy", "")


# ── SBOMDocument ───────────────────────────────────────────────────────────────

class TestSBOMDocument:
    def _sbom(self) -> SBOMDocument:
        s = SBOMDocument(component_name="credit-model", component_version="v1.2")
        s.add_entry(SBOMEntry("numpy", "1.26.4"))
        s.add_entry(SBOMEntry("pandas", "2.2.0"))
        return s

    def test_add_entry(self) -> None:
        s = self._sbom()
        assert "numpy" in s.component_names()
        assert "pandas" in s.component_names()

    def test_to_dict_structure(self) -> None:
        d = self._sbom().to_dict()
        assert d["bomFormat"] == "CycloneDX"
        assert d["specVersion"] == "1.5"
        assert "components" in d
        assert len(d["components"]) == 2

    def test_to_json(self) -> None:
        j = self._sbom().to_json()
        parsed = json.loads(j)
        assert parsed["bomFormat"] == "CycloneDX"

    def test_metadata_component(self) -> None:
        d = self._sbom().to_dict()
        assert d["metadata"]["component"]["name"] == "credit-model"


# ── ArtifactProvenanceRecord ───────────────────────────────────────────────────

class TestArtifactProvenanceRecord:
    def _record(self) -> ArtifactProvenanceRecord:
        return ArtifactProvenanceRecord(
            artifact_name="model_v1.pkl",
            artifact_sha256="abc123",
            commit_sha="6c6a398",
        )

    def test_built_at_auto_set(self) -> None:
        r = self._record()
        assert "2026" in r.built_at or "T" in r.built_at  # ISO format

    def test_empty_artifact_name_raises(self) -> None:
        with pytest.raises(ValueError, match="artifact_name"):
            ArtifactProvenanceRecord(artifact_name="", artifact_sha256="x")

    def test_to_dict(self) -> None:
        d = self._record().to_dict()
        assert d["artifact_name"] == "model_v1.pkl"
        assert d["commit_sha"] == "6c6a398"

    def test_to_json_roundtrip(self) -> None:
        r = self._record()
        parsed = json.loads(r.to_json())
        assert parsed["artifact_sha256"] == "abc123"

    def test_from_dict(self) -> None:
        r = self._record()
        r2 = ArtifactProvenanceRecord.from_dict(r.to_dict())
        assert r2.artifact_name == r.artifact_name
        assert r2.commit_sha == r.commit_sha


# ── ArtifactSigner ─────────────────────────────────────────────────────────────

class TestArtifactSigner:
    def _signer(self) -> ArtifactSigner:
        return ArtifactSigner(backend="hmac-sha256", signer_identity="ci@test.com")

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            ArtifactSigner(backend="bad")

    def test_sign_success(self) -> None:
        result = self._signer().sign(ARTIFACT, "model.pkl")
        assert result.success
        assert len(result.artifact_sha256) == 64  # sha256 hex

    def test_sign_signature_bundle_nonempty(self) -> None:
        result = self._signer().sign(ARTIFACT)
        assert result.signature_bundle

    def test_sign_rekor_uuid_nonempty(self) -> None:
        result = self._signer().sign(ARTIFACT)
        assert result.rekor_log_uuid

    def test_sign_different_artifacts_different_digest(self) -> None:
        r1 = self._signer().sign(b"artifact-a")
        r2 = self._signer().sign(b"artifact-b")
        assert r1.artifact_sha256 != r2.artifact_sha256

    def test_verify_valid(self) -> None:
        signer = self._signer()
        result = signer.sign(ARTIFACT)
        assert signer.verify(ARTIFACT, result.artifact_sha256)

    def test_verify_tampered(self) -> None:
        signer = self._signer()
        result = signer.sign(ARTIFACT)
        assert not signer.verify(b"tampered content", result.artifact_sha256)

    def test_cosign_backend_raises_not_implemented(self) -> None:
        signer = ArtifactSigner(backend="cosign-keyless")
        with pytest.raises(NotImplementedError, match="cosign"):
            signer.sign(ARTIFACT)

    def test_generate_provenance(self) -> None:
        signer = self._signer()
        result = signer.sign(ARTIFACT)
        prov = signer.generate_provenance(
            artifact_bytes=ARTIFACT,
            artifact_name="model.pkl",
            signing_result=result,
            commit_sha="abc123",
            branch="main",
        )
        assert prov.artifact_sha256 == result.artifact_sha256
        assert prov.commit_sha == "abc123"
        assert prov.signing_backend == "hmac-sha256"

    def test_build_sbom(self) -> None:
        sbom = ArtifactSigner.build_sbom(
            component_name="credit-model",
            component_version="v1",
            dependencies=[
                {"name": "numpy", "version": "1.26.4", "license": "BSD-3-Clause"},
                {"name": "pandas", "version": "2.2.0"},
            ],
        )
        assert isinstance(sbom, SBOMDocument)
        assert "numpy" in sbom.component_names()
        assert "pandas" in sbom.component_names()

    def test_signing_result_type(self) -> None:
        result = self._signer().sign(ARTIFACT)
        assert isinstance(result, SigningResult)
        assert result.backend == "hmac-sha256"
