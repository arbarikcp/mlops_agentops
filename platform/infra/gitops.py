"""GitOps manifest builders for Argo CD Application and sync policy.

Day 74 — defines Application CRD manifest builders and sync result types
for GitOps-driven ML model deployment with Argo CD.

Classes:
  AppHealthStatus — enumeration of Argo CD health states
  SyncPolicy      — automated sync configuration
  AppSyncResult   — result of an Argo CD sync operation
  ArgoCDApp       — Application CRD manifest builder

See: docs/phase11/day74_gitops.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AppHealthStatus(str, Enum):
    """Argo CD application health / sync status."""
    SYNCED = "Synced"
    OUT_OF_SYNC = "OutOfSync"
    PROGRESSING = "Progressing"
    DEGRADED = "Degraded"
    SUSPENDED = "Suspended"
    UNKNOWN = "Unknown"


# ── SyncPolicy ────────────────────────────────────────────────────────────────

@dataclass
class SyncPolicy:
    """Argo CD automated sync policy.

    Attributes:
        automated:       Enable automated sync.
        prune:           Delete resources removed from Git.
        self_heal:       Re-sync if cluster drifts from Git.
        retry_limit:     Max retry attempts after a failed sync.
        retry_backoff_s: Initial retry backoff in seconds.
    """

    automated: bool = True
    prune: bool = True
    self_heal: bool = True
    retry_limit: int = 3
    retry_backoff_s: int = 30

    def __post_init__(self) -> None:
        if self.retry_limit < 0:
            raise ValueError("retry_limit must be >= 0")
        if self.retry_backoff_s < 1:
            raise ValueError("retry_backoff_s must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "syncOptions": ["CreateNamespace=true", "ApplyOutOfSyncOnly=true"],
            "retry": {
                "limit": self.retry_limit,
                "backoff": {
                    "duration": f"{self.retry_backoff_s}s",
                    "factor": 2,
                    "maxDuration": "5m",
                },
            },
        }
        if self.automated:
            d["automated"] = {
                "prune": self.prune,
                "selfHeal": self.self_heal,
            }
        return d


# ── AppSyncResult ─────────────────────────────────────────────────────────────

@dataclass
class AppSyncResult:
    """Result of an Argo CD sync operation (or status poll).

    Attributes:
        app_name:  Application name.
        status:    AppHealthStatus value.
        revision:  Git commit SHA that was synced.
        message:   Human-readable status message.
    """

    app_name: str
    status: AppHealthStatus
    revision: str = ""
    message: str = ""

    def __post_init__(self) -> None:
        if not self.app_name:
            raise ValueError("AppSyncResult.app_name cannot be empty")

    def is_healthy(self) -> bool:
        return self.status == AppHealthStatus.SYNCED

    def is_degraded(self) -> bool:
        return self.status == AppHealthStatus.DEGRADED

    def is_in_progress(self) -> bool:
        return self.status == AppHealthStatus.PROGRESSING


# ── ArgoCDApp ─────────────────────────────────────────────────────────────────

@dataclass
class ArgoCDApp:
    """Argo CD Application CRD manifest builder.

    Attributes:
        name:                    Application name.
        repo_url:                Source Git repo URL.
        chart_path:              Path in the repo to the Helm chart.
        destination_namespace:   Target K8s namespace.
        target_revision:         Git branch / tag / SHA.
        argocd_namespace:        Namespace where Argo CD is installed.
        value_files:             Helm value files to layer (in order).
        sync_policy:             Automated sync configuration.
        sync_wave:               Argo CD sync-wave annotation (0 = first).
        project:                 Argo CD project name.
    """

    name: str
    repo_url: str
    chart_path: str
    destination_namespace: str = "ml-serving"
    target_revision: str = "main"
    argocd_namespace: str = "argocd"
    value_files: list[str] = field(default_factory=lambda: ["values.yaml"])
    sync_policy: SyncPolicy = field(default_factory=SyncPolicy)
    sync_wave: int = 0
    project: str = "default"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ArgoCDApp.name cannot be empty")
        if not self.repo_url:
            raise ValueError("ArgoCDApp.repo_url cannot be empty")
        if not self.chart_path:
            raise ValueError("ArgoCDApp.chart_path cannot be empty")
        if not self.value_files:
            raise ValueError("ArgoCDApp.value_files must have at least one entry")

    def to_manifest(self) -> dict[str, Any]:
        annotations: dict[str, str] = {}
        if self.sync_wave != 0:
            annotations["argocd.argoproj.io/sync-wave"] = str(self.sync_wave)

        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Application",
            "metadata": {
                "name": self.name,
                "namespace": self.argocd_namespace,
                **({"annotations": annotations} if annotations else {}),
            },
            "spec": {
                "project": self.project,
                "source": {
                    "repoURL": self.repo_url,
                    "targetRevision": self.target_revision,
                    "path": self.chart_path,
                    "helm": {"valueFiles": self.value_files},
                },
                "destination": {
                    "server": "https://kubernetes.default.svc",
                    "namespace": self.destination_namespace,
                },
                "syncPolicy": self.sync_policy.to_dict(),
            },
        }

    def with_model_version(self, model_version: str, storage_uri: str) -> "ArgoCDApp":
        """Return a new ArgoCDApp with an extra values file for the model version.

        This is the pattern for CI-triggered model promotion: CI writes a new
        values file, pushes to Git, and Argo CD reconciles to the new model.
        """
        extra_values = f"values-{model_version}.yaml"
        return ArgoCDApp(
            name=self.name,
            repo_url=self.repo_url,
            chart_path=self.chart_path,
            destination_namespace=self.destination_namespace,
            target_revision=self.target_revision,
            argocd_namespace=self.argocd_namespace,
            value_files=self.value_files + [extra_values],
            sync_policy=self.sync_policy,
            sync_wave=self.sync_wave,
            project=self.project,
        )
