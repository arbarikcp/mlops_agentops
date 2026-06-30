"""
prompt_registry.py — Prompt Management & Versioning (Day 102)

Covers prompts-as-code (git-reviewed), a centralized prompt registry with
version history and status lifecycle (draft -> staging -> production ->
deprecated), and deterministic hash-based A/B testing of prompt variants.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PromptStatus(str, Enum):
    """Lifecycle status of a prompt version."""

    DRAFT = "draft"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class PromptVersion:
    """A single versioned prompt template."""

    name: str
    version: str
    template: str
    status: PromptStatus = PromptStatus.DRAFT
    variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not self.version:
            raise ValueError("version must be non-empty")
        if not self.template:
            raise ValueError("template must be non-empty")

    def render(self, **kwargs) -> str:
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise KeyError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "status": self.status.value,
            "variables": self.variables,
        }


@dataclass
class PromptRegistry:
    """Centralized store for prompt versions keyed by name."""

    versions: dict[str, list[PromptVersion]] = field(default_factory=dict)

    def register(self, pv: PromptVersion) -> None:
        self.versions.setdefault(pv.name, []).append(pv)

    def get(self, name: str, version: str) -> PromptVersion:
        for pv in self.versions.get(name, []):
            if pv.version == version:
                return pv
        raise KeyError(f"No prompt {name!r} version {version!r} found")

    def get_production(self, name: str) -> PromptVersion | None:
        candidates = [
            pv for pv in self.versions.get(name, [])
            if pv.status == PromptStatus.PRODUCTION
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pv: pv.version)

    def promote(self, name: str, version: str, new_status: PromptStatus) -> None:
        pv = self.get(name, version)
        pv.status = new_status

    def history(self, name: str) -> list[PromptVersion]:
        return list(self.versions.get(name, []))


@dataclass
class PromptABTest:
    """Deterministic traffic-split A/B test between two prompt variants."""

    name: str
    variant_a: PromptVersion
    variant_b: PromptVersion
    traffic_split_b: float = 0.5

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")
        if not (0 < self.traffic_split_b < 1):
            raise ValueError("traffic_split_b must be in (0, 1)")

    def assign_variant(self, request_id: str) -> PromptVersion:
        bucket = hash(request_id) % 100
        if bucket < self.traffic_split_b * 100:
            return self.variant_b
        return self.variant_a

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "variant_a": self.variant_a.to_dict(),
            "variant_b": self.variant_b.to_dict(),
            "traffic_split_b": self.traffic_split_b,
        }
