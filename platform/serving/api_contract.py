"""Model API contract: schema versioning, compatibility checking, rollback plan.

A Model API Contract is the formal agreement between the ML team (producer)
and downstream systems (consumers). It defines what requests look like, what
responses guarantee, and what changes are safe to make without breaking callers.

Key concepts:
    COMPATIBLE change — additive (new optional field); consumers still work
    BREAKING change   — removed/renamed required field, type change; consumers break
    WARNING           — removed optional field; consumers may break silently

Rollback plan:
    A versioned document describing how to revert to the previous model
    in < 7 minutes. Stored alongside the model in the registry.

See: docs/phase4/day28_api_contract.md for theory.

Usage:
    from serving.api_contract import ApiContractChecker, ApiContractVersion, FieldSchema

    v1 = ApiContractVersion(version="v1", request_fields=[...], response_fields=[...])
    v2 = ApiContractVersion(version="v2", request_fields=[...], response_fields=[...])

    checker = ApiContractChecker()
    report = checker.check_compatibility(v1, v2)
    if not report.is_backward_compatible:
        raise RuntimeError(f"Breaking changes: {report.breaking_changes}")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


# ── Field schema ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FieldSchema:
    """Schema definition for a single request or response field.

    Attributes:
        name:        Field name as it appears in the JSON body.
        type:        Python type name: "int", "float", "str", "bool", "dict".
        required:    True if the field must be present.
        nullable:    True if None is an allowed value.
        description: Human-readable description.
    """

    name: str
    type: str
    required: bool = True
    nullable: bool = False
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "nullable": self.nullable,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FieldSchema":
        return cls(**d)


# ── Contract version ──────────────────────────────────────────────────────────

@dataclass
class ApiContractVersion:
    """A specific version of the Model API contract.

    Attributes:
        version:         Semantic version string (e.g. "v1", "v2.1").
        request_fields:  List of FieldSchema for the request body.
        response_fields: List of FieldSchema for the response body.
        deprecated:      True if this version is scheduled for removal.
        sunset_date:     ISO-8601 date after which this version may be removed.
        description:     What changed in this version.
    """

    version: str
    request_fields: list[FieldSchema] = field(default_factory=list)
    response_fields: list[FieldSchema] = field(default_factory=list)
    deprecated: bool = False
    sunset_date: str | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "request_fields": [f.to_dict() for f in self.request_fields],
            "response_fields": [f.to_dict() for f in self.response_fields],
            "deprecated": self.deprecated,
            "sunset_date": self.sunset_date,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ApiContractVersion":
        return cls(
            version=d["version"],
            request_fields=[FieldSchema.from_dict(f) for f in d.get("request_fields", [])],
            response_fields=[FieldSchema.from_dict(f) for f in d.get("response_fields", [])],
            deprecated=d.get("deprecated", False),
            sunset_date=d.get("sunset_date"),
            description=d.get("description", ""),
        )

    def save(self, path: Path) -> None:
        """Persist this contract version to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "ApiContractVersion":
        """Load a contract version from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))


# ── Compatibility report ──────────────────────────────────────────────────────

@dataclass
class CompatibilityReport:
    """Result of comparing two contract versions for backward compatibility.

    Attributes:
        from_version:          Source (older) version string.
        to_version:            Target (newer) version string.
        breaking_changes:      Changes that break backward compatibility.
        warnings:              Changes that may break some consumers.
        compatible_changes:    Safe additive changes.
        is_backward_compatible: True if no breaking changes found.
    """

    from_version: str
    to_version: str
    breaking_changes: list[str]
    warnings: list[str]
    compatible_changes: list[str]

    @property
    def is_backward_compatible(self) -> bool:
        return len(self.breaking_changes) == 0

    def summary(self) -> str:
        lines = [
            f"Compatibility: {self.from_version} → {self.to_version}",
            f"  Backward compatible: {self.is_backward_compatible}",
        ]
        for msg in self.breaking_changes:
            lines.append(f"  BREAKING: {msg}")
        for msg in self.warnings:
            lines.append(f"  WARNING:  {msg}")
        for msg in self.compatible_changes:
            lines.append(f"  OK:       {msg}")
        return "\n".join(lines)


# ── Compatibility checker ─────────────────────────────────────────────────────

class ApiContractChecker:
    """Compares two ApiContractVersions and classifies each schema change.

    Breaking changes:
        - Removed required field
        - Changed field type
        - Changed field from optional to required (tightening)
        - Changed field from nullable to non-nullable

    Warnings:
        - Removed optional field (consumers may silently read None)

    Compatible changes:
        - Added optional field
        - Loosened constraint (required → optional)
        - Added description
    """

    def check_compatibility(
        self,
        v1: ApiContractVersion,
        v2: ApiContractVersion,
    ) -> CompatibilityReport:
        """Compare v1 → v2 for backward compatibility.

        Args:
            v1: The current/older contract version.
            v2: The proposed/newer contract version.

        Returns:
            CompatibilityReport classifying all changes.
        """
        breaking: list[str] = []
        warnings: list[str] = []
        compatible: list[str] = []

        # Check request fields
        req_b, req_w, req_c = self._compare_fields(
            v1.request_fields, v2.request_fields, context="request"
        )
        breaking.extend(req_b)
        warnings.extend(req_w)
        compatible.extend(req_c)

        # Check response fields
        resp_b, resp_w, resp_c = self._compare_fields(
            v1.response_fields, v2.response_fields, context="response"
        )
        breaking.extend(resp_b)
        warnings.extend(resp_w)
        compatible.extend(resp_c)

        report = CompatibilityReport(
            from_version=v1.version,
            to_version=v2.version,
            breaking_changes=breaking,
            warnings=warnings,
            compatible_changes=compatible,
        )

        if not report.is_backward_compatible:
            log.warning(
                "Breaking changes detected (%s → %s): %s",
                v1.version, v2.version, breaking,
            )

        return report

    def _compare_fields(
        self,
        old_fields: list[FieldSchema],
        new_fields: list[FieldSchema],
        context: str,
    ) -> tuple[list[str], list[str], list[str]]:
        """Compare two field lists; return (breaking, warnings, compatible) lists."""
        breaking: list[str] = []
        warnings: list[str] = []
        compatible: list[str] = []

        old_map = {f.name: f for f in old_fields}
        new_map = {f.name: f for f in new_fields}

        # Fields in v1 but not in v2 (removed)
        for name, old_f in old_map.items():
            if name not in new_map:
                if old_f.required:
                    breaking.append(
                        f"{context}.{name}: removed required field (BREAKING)"
                    )
                else:
                    warnings.append(
                        f"{context}.{name}: removed optional field (consumers may break)"
                    )

        # Fields in v2 but not in v1 (added)
        for name, new_f in new_map.items():
            if name not in old_map:
                if new_f.required:
                    breaking.append(
                        f"{context}.{name}: added required field (consumers won't send it)"
                    )
                else:
                    compatible.append(
                        f"{context}.{name}: added optional field"
                    )

        # Fields in both (modified)
        for name in old_map.keys() & new_map.keys():
            old_f = old_map[name]
            new_f = new_map[name]

            if old_f.type != new_f.type:
                breaking.append(
                    f"{context}.{name}: type changed {old_f.type!r} → {new_f.type!r}"
                )

            if not old_f.required and new_f.required:
                breaking.append(
                    f"{context}.{name}: changed from optional to required"
                )
            elif old_f.required and not new_f.required:
                compatible.append(
                    f"{context}.{name}: relaxed from required to optional"
                )

            if old_f.nullable and not new_f.nullable:
                breaking.append(
                    f"{context}.{name}: changed from nullable to non-nullable"
                )
            elif not old_f.nullable and new_f.nullable:
                compatible.append(
                    f"{context}.{name}: relaxed to nullable"
                )

        return breaking, warnings, compatible


# ── Rollback plan ─────────────────────────────────────────────────────────────

@dataclass
class RollbackPlan:
    """Versioned document describing how to revert to the previous model.

    Stored alongside the model artifact in the registry.
    Target rollback time: < 7 minutes from alert to stable.

    Attributes:
        model_version:         Version being deployed.
        previous_stable:       Last known-good version to roll back to.
        rollback_steps:        Ordered list of steps for the on-call engineer.
        rollback_ttl_minutes:  SLA: rollback must complete within this time.
        canary_traffic_pct:    Traffic percentage for the canary phase.
    """

    model_version: str
    previous_stable: str
    rollback_steps: list[str] = field(default_factory=list)
    rollback_ttl_minutes: int = 7
    canary_traffic_pct: int = 10

    def to_dict(self) -> dict:
        return {
            "model_version": self.model_version,
            "previous_stable": self.previous_stable,
            "rollback_steps": self.rollback_steps,
            "rollback_ttl_minutes": self.rollback_ttl_minutes,
            "canary_traffic_pct": self.canary_traffic_pct,
        }

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> "RollbackPlan":
        d = json.loads(Path(path).read_text())
        return cls(**d)

    def validate(self) -> list[str]:
        """Return a list of problems with this rollback plan (empty = valid)."""
        issues: list[str] = []
        if not self.rollback_steps:
            issues.append("rollback_steps is empty — on-call will have no guidance")
        if self.rollback_ttl_minutes > 15:
            issues.append(f"rollback_ttl_minutes={self.rollback_ttl_minutes} exceeds 15min SLA")
        if self.previous_stable == self.model_version:
            issues.append("previous_stable == model_version — cannot roll back to self")
        if not (0 < self.canary_traffic_pct <= 50):
            issues.append(f"canary_traffic_pct={self.canary_traffic_pct} should be in (0, 50]")
        return issues


# ── Default contract versions for credit risk model ───────────────────────────

def build_v1_contract() -> ApiContractVersion:
    """The v1 contract for the credit risk inference API."""
    return ApiContractVersion(
        version="v1",
        description="Initial credit risk scoring API",
        request_fields=[
            FieldSchema("applicant_id", "int", required=True, description="Applicant identifier"),
            FieldSchema("features", "dict", required=True, description="Feature name → float"),
        ],
        response_fields=[
            FieldSchema("applicant_id", "int", required=True),
            FieldSchema("score", "float", required=True, description="Default probability"),
            FieldSchema("label", "int", required=True, description="Binary decision"),
            FieldSchema("model_version", "str", required=True),
            FieldSchema("latency_ms", "float", required=True),
        ],
    )


def build_v2_contract() -> ApiContractVersion:
    """The v2 contract — adds optional explanation field."""
    return ApiContractVersion(
        version="v2",
        description="Added optional SHAP explanation in response",
        request_fields=[
            FieldSchema("applicant_id", "int", required=True, description="Applicant identifier"),
            FieldSchema("features", "dict", required=True, description="Feature name → float"),
            FieldSchema("explain", "bool", required=False, description="Request SHAP explanation"),
        ],
        response_fields=[
            FieldSchema("applicant_id", "int", required=True),
            FieldSchema("score", "float", required=True, description="Default probability"),
            FieldSchema("label", "int", required=True, description="Binary decision"),
            FieldSchema("model_version", "str", required=True),
            FieldSchema("latency_ms", "float", required=True),
            FieldSchema("explanation", "dict", required=False, nullable=True,
                        description="SHAP feature importances (optional)"),
        ],
    )
