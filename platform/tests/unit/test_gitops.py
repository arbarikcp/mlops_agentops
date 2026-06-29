"""Tests for infra/gitops.py — AppHealthStatus, SyncPolicy, AppSyncResult, ArgoCDApp."""
from __future__ import annotations

import pytest

from infra.gitops import (
    AppHealthStatus,
    AppSyncResult,
    ArgoCDApp,
    SyncPolicy,
)


def make_app(**kwargs) -> ArgoCDApp:
    defaults = dict(
        name="credit-risk-serving",
        repo_url="https://github.com/arbarikcp/mlops_agentops.git",
        chart_path="platform/infra/helm/credit-risk",
        destination_namespace="ml-serving",
    )
    defaults.update(kwargs)
    return ArgoCDApp(**defaults)


# ── AppHealthStatus ───────────────────────────────────────────────────────────

class TestAppHealthStatus:
    def test_all_values_exist(self) -> None:
        values = {s.value for s in AppHealthStatus}
        assert "Synced" in values
        assert "OutOfSync" in values
        assert "Degraded" in values

    def test_is_str_enum(self) -> None:
        assert AppHealthStatus.SYNCED == "Synced"


# ── SyncPolicy ────────────────────────────────────────────────────────────────

class TestSyncPolicy:
    def test_default_construction(self) -> None:
        sp = SyncPolicy()
        assert sp.automated is True
        assert sp.prune is True

    def test_negative_retry_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="retry_limit"):
            SyncPolicy(retry_limit=-1)

    def test_zero_backoff_raises(self) -> None:
        with pytest.raises(ValueError, match="retry_backoff_s"):
            SyncPolicy(retry_backoff_s=0)

    def test_to_dict_automated(self) -> None:
        d = SyncPolicy(automated=True, prune=True, self_heal=False).to_dict()
        assert "automated" in d
        assert d["automated"]["prune"] is True
        assert d["automated"]["selfHeal"] is False

    def test_to_dict_no_automated(self) -> None:
        d = SyncPolicy(automated=False).to_dict()
        assert "automated" not in d

    def test_retry_in_dict(self) -> None:
        d = SyncPolicy(retry_limit=5, retry_backoff_s=60).to_dict()
        assert d["retry"]["limit"] == 5
        assert "60s" in d["retry"]["backoff"]["duration"]

    def test_sync_options_in_dict(self) -> None:
        d = SyncPolicy().to_dict()
        assert "CreateNamespace=true" in d["syncOptions"]


# ── AppSyncResult ─────────────────────────────────────────────────────────────

class TestAppSyncResult:
    def test_empty_app_name_raises(self) -> None:
        with pytest.raises(ValueError, match="app_name"):
            AppSyncResult(app_name="", status=AppHealthStatus.SYNCED)

    def test_is_healthy_synced(self) -> None:
        r = AppSyncResult("app", AppHealthStatus.SYNCED)
        assert r.is_healthy() is True

    def test_is_healthy_degraded(self) -> None:
        r = AppSyncResult("app", AppHealthStatus.DEGRADED)
        assert r.is_healthy() is False

    def test_is_degraded(self) -> None:
        r = AppSyncResult("app", AppHealthStatus.DEGRADED)
        assert r.is_degraded() is True

    def test_is_in_progress(self) -> None:
        r = AppSyncResult("app", AppHealthStatus.PROGRESSING)
        assert r.is_in_progress() is True

    def test_not_in_progress_when_synced(self) -> None:
        r = AppSyncResult("app", AppHealthStatus.SYNCED)
        assert r.is_in_progress() is False


# ── ArgoCDApp ─────────────────────────────────────────────────────────────────

class TestArgoCDApp:
    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            make_app(name="")

    def test_empty_repo_url_raises(self) -> None:
        with pytest.raises(ValueError, match="repo_url"):
            make_app(repo_url="")

    def test_empty_chart_path_raises(self) -> None:
        with pytest.raises(ValueError, match="chart_path"):
            make_app(chart_path="")

    def test_empty_value_files_raises(self) -> None:
        with pytest.raises(ValueError, match="value_files"):
            make_app(value_files=[])

    def test_to_manifest_kind(self) -> None:
        m = make_app().to_manifest()
        assert m["kind"] == "Application"
        assert m["apiVersion"] == "argoproj.io/v1alpha1"

    def test_to_manifest_metadata_name(self) -> None:
        m = make_app().to_manifest()
        assert m["metadata"]["name"] == "credit-risk-serving"

    def test_to_manifest_source(self) -> None:
        m = make_app().to_manifest()
        src = m["spec"]["source"]
        assert src["path"] == "platform/infra/helm/credit-risk"
        assert src["targetRevision"] == "main"

    def test_to_manifest_destination(self) -> None:
        m = make_app().to_manifest()
        dest = m["spec"]["destination"]
        assert dest["namespace"] == "ml-serving"

    def test_to_manifest_value_files(self) -> None:
        app = make_app(value_files=["values.yaml", "values-prod.yaml"])
        m = app.to_manifest()
        assert m["spec"]["source"]["helm"]["valueFiles"] == ["values.yaml", "values-prod.yaml"]

    def test_sync_wave_annotation_added(self) -> None:
        app = make_app(sync_wave=3)
        m = app.to_manifest()
        assert m["metadata"]["annotations"]["argocd.argoproj.io/sync-wave"] == "3"

    def test_no_annotation_when_wave_zero(self) -> None:
        app = make_app(sync_wave=0)
        m = app.to_manifest()
        assert "annotations" not in m["metadata"]

    def test_sync_policy_in_manifest(self) -> None:
        m = make_app().to_manifest()
        assert "syncPolicy" in m["spec"]
        assert "automated" in m["spec"]["syncPolicy"]

    def test_with_model_version_adds_values_file(self) -> None:
        app = make_app()
        new_app = app.with_model_version("v1.3", "s3://ml-models/v1.3/")
        assert "values-v1.3.yaml" in new_app.value_files

    def test_with_model_version_preserves_original(self) -> None:
        app = make_app()
        _ = app.with_model_version("v1.3", "s3://ml-models/v1.3/")
        assert "values-v1.3.yaml" not in app.value_files

    def test_default_project(self) -> None:
        m = make_app().to_manifest()
        assert m["spec"]["project"] == "default"

    def test_custom_project(self) -> None:
        m = make_app(project="ml-platform").to_manifest()
        assert m["spec"]["project"] == "ml-platform"
