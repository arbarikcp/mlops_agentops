"""Artifact signing, SBOM generation, and provenance records.

Day 57 — pure-Python (no cosign binary required) simulation of the signing
workflow. In production, `ArtifactSigner` would shell out to `cosign`. In CI
tests and this module, signing is simulated with SHA-256 HMAC so all logic
and data structures are exercised without external tools.

Classes:
  SBOMEntry               — one dependency in the Software Bill of Materials
  SBOMDocument            — full SBOM for one artifact (CycloneDX-inspired)
  ArtifactProvenanceRecord — full provenance JSON written alongside artifacts
  SigningResult           — outcome of one signing operation
  ArtifactSigner          — orchestrates sign → SBOM → provenance

See: docs/phase8/day57_signing_sbom.md
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ── SBOMEntry ─────────────────────────────────────────────────────────────────

@dataclass
class SBOMEntry:
    """One component in the Software Bill of Materials.

    Attributes:
        name:     Package name (e.g., "numpy").
        version:  Package version string.
        license:  SPDX license identifier (e.g., "BSD-3-Clause").
        purl:     Package URL — globally unique identifier.
    """

    name: str
    version: str
    license: str = "UNKNOWN"
    purl: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SBOMEntry.name cannot be empty")
        if not self.version:
            raise ValueError("SBOMEntry.version cannot be empty")
        if not self.purl:
            self.purl = f"pkg:pypi/{self.name.lower()}@{self.version}"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "license": self.license,
            "purl": self.purl,
        }


# ── SBOMDocument ──────────────────────────────────────────────────────────────

@dataclass
class SBOMDocument:
    """Full Software Bill of Materials for one artifact.

    Follows a CycloneDX-inspired structure without requiring the cdxgen tool.

    Attributes:
        format:            "CycloneDX" (default).
        spec_version:      CycloneDX spec version (default "1.5").
        component_name:    Artifact being described.
        component_version: Artifact version string.
        entries:           Dependency entries.
    """

    format: str = "CycloneDX"
    spec_version: str = "1.5"
    component_name: str = ""
    component_version: str = "unknown"
    entries: list[SBOMEntry] = field(default_factory=list)

    def add_entry(self, entry: SBOMEntry) -> None:
        self.entries.append(entry)

    def component_names(self) -> list[str]:
        return [e.name for e in self.entries]

    def to_dict(self) -> dict:
        return {
            "bomFormat": self.format,
            "specVersion": self.spec_version,
            "metadata": {
                "component": {
                    "name": self.component_name,
                    "version": self.component_version,
                }
            },
            "components": [e.to_dict() for e in self.entries],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── ArtifactProvenanceRecord ──────────────────────────────────────────────────

@dataclass
class ArtifactProvenanceRecord:
    """Full provenance metadata for a signed ML artifact.

    Attributes:
        artifact_name:    Filename of the artifact.
        artifact_sha256:  SHA-256 hex digest of the artifact bytes.
        built_at:         ISO-8601 UTC timestamp.
        commit_sha:       Git commit SHA that produced this artifact.
        branch:           Git branch name.
        pipeline_id:      CI pipeline identifier.
        builder_identity: Signer identity (CI OIDC email or job ID).
        data_version:     Data version tag used for training.
        signing_backend:  "cosign-keyless" in prod; "hmac-sha256" in tests.
        rekor_log_uuid:   Rekor transparency log entry UUID (empty if not signed).
        sbom_path:        Path to the accompanying SBOM document.
    """

    artifact_name: str
    artifact_sha256: str
    built_at: str = ""
    commit_sha: str = "unknown"
    branch: str = "unknown"
    pipeline_id: str = "unknown"
    builder_identity: str = "ci@example.com"
    data_version: str = "v1"
    signing_backend: str = "hmac-sha256"
    rekor_log_uuid: str = ""
    sbom_path: str = ""

    def __post_init__(self) -> None:
        if not self.built_at:
            self.built_at = datetime.now(timezone.utc).isoformat()
        if not self.artifact_name:
            raise ValueError("artifact_name cannot be empty")

    def to_dict(self) -> dict:
        return {
            "artifact_name": self.artifact_name,
            "artifact_sha256": self.artifact_sha256,
            "built_at": self.built_at,
            "commit_sha": self.commit_sha,
            "branch": self.branch,
            "pipeline_id": self.pipeline_id,
            "builder_identity": self.builder_identity,
            "data_version": self.data_version,
            "signing_backend": self.signing_backend,
            "rekor_log_uuid": self.rekor_log_uuid,
            "sbom_path": self.sbom_path,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ArtifactProvenanceRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── SigningResult ─────────────────────────────────────────────────────────────

@dataclass
class SigningResult:
    """Outcome of one artifact signing operation.

    Attributes:
        success:          True if signing completed without error.
        artifact_sha256:  SHA-256 hex digest of the artifact.
        signature_bundle: Base64-encoded signature bytes (or HMAC hex in test mode).
        rekor_log_uuid:   Transparency log entry UUID (simulated in test mode).
        backend:          Signing backend used.
        message:          Human-readable status.
    """

    success: bool
    artifact_sha256: str
    signature_bundle: str = ""
    rekor_log_uuid: str = ""
    backend: str = "hmac-sha256"
    message: str = ""


# ── ArtifactSigner ────────────────────────────────────────────────────────────

class ArtifactSigner:
    """Signs artifacts, generates SBOMs, and builds provenance records.

    In production, set `backend="cosign-keyless"` and ensure `cosign` is on
    PATH. The test/simulation backend ("hmac-sha256") signs with HMAC-SHA256
    keyed on the signer_identity string — no external binary required.

    Args:
        backend:          "hmac-sha256" (default/test) or "cosign-keyless" (prod).
        signer_identity:  OIDC identity of the signer (used as HMAC key in test mode).
    """

    _SIMULATED_REKOR_PREFIX = "24296fb24b3d"

    def __init__(
        self,
        backend: str = "hmac-sha256",
        signer_identity: str = "ci@example.com",
    ) -> None:
        valid_backends = {"hmac-sha256", "cosign-keyless"}
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}; got {backend!r}")
        self.backend = backend
        self.signer_identity = signer_identity

    def _compute_sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _simulate_sign(self, digest: str) -> tuple[str, str]:
        """Return (signature_bundle, rekor_log_uuid) for test mode."""
        key = self.signer_identity.encode()
        sig = hmac.new(key, digest.encode(), hashlib.sha256).hexdigest()
        uuid = self._SIMULATED_REKOR_PREFIX + digest[:16]
        return sig, uuid

    def compute_file_sha256(self, path: str) -> str:
        """Compute SHA-256 digest of a file on disk."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def sign(
        self,
        artifact_bytes: bytes,
        artifact_name: str = "artifact",
    ) -> SigningResult:
        """Sign an artifact. In test mode: HMAC-SHA256. In prod: cosign keyless.

        Args:
            artifact_bytes: Raw bytes of the artifact to sign.
            artifact_name:  Filename for logging.

        Returns:
            SigningResult with digest, signature bundle, and Rekor UUID.
        """
        digest = self._compute_sha256(artifact_bytes)

        if self.backend == "cosign-keyless":
            # In a real deployment cosign would be invoked here via subprocess.
            # We raise so callers know this requires the binary.
            raise NotImplementedError(
                "cosign-keyless requires the cosign binary — "
                "set backend='hmac-sha256' for tests"
            )

        sig, uuid = self._simulate_sign(digest)
        return SigningResult(
            success=True,
            artifact_sha256=digest,
            signature_bundle=sig,
            rekor_log_uuid=uuid,
            backend=self.backend,
            message=f"signed {artifact_name} ({len(artifact_bytes)} bytes)",
        )

    def verify(self, artifact_bytes: bytes, expected_sha256: str) -> bool:
        """Verify artifact integrity by comparing SHA-256 digests.

        In production this would invoke `cosign verify` with the full bundle.

        Args:
            artifact_bytes:  Raw bytes to check.
            expected_sha256: Hex digest from the stored provenance record.

        Returns:
            True if digests match.
        """
        actual = self._compute_sha256(artifact_bytes)
        return hmac.compare_digest(actual, expected_sha256)

    def generate_provenance(
        self,
        artifact_bytes: bytes,
        artifact_name: str,
        signing_result: SigningResult,
        commit_sha: str = "unknown",
        branch: str = "unknown",
        pipeline_id: str = "unknown",
        data_version: str = "v1",
        sbom_path: str = "",
    ) -> ArtifactProvenanceRecord:
        """Build a provenance record from a signing result."""
        return ArtifactProvenanceRecord(
            artifact_name=artifact_name,
            artifact_sha256=signing_result.artifact_sha256,
            commit_sha=commit_sha,
            branch=branch,
            pipeline_id=pipeline_id,
            builder_identity=self.signer_identity,
            data_version=data_version,
            signing_backend=signing_result.backend,
            rekor_log_uuid=signing_result.rekor_log_uuid,
            sbom_path=sbom_path,
        )

    @staticmethod
    def build_sbom(
        component_name: str,
        component_version: str,
        dependencies: list[dict],
    ) -> SBOMDocument:
        """Build a CycloneDX-style SBOM document.

        Args:
            component_name:    Name of the model/artifact.
            component_version: Version string.
            dependencies:      List of dicts with "name", "version", "license"? keys.

        Returns:
            SBOMDocument with all entries populated.
        """
        sbom = SBOMDocument(
            component_name=component_name,
            component_version=component_version,
        )
        for dep in dependencies:
            sbom.add_entry(SBOMEntry(
                name=dep["name"],
                version=dep["version"],
                license=dep.get("license", "UNKNOWN"),
            ))
        return sbom
