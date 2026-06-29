"""Tests for infra/helm_chart.py — HelmValues, HelmChart."""
from __future__ import annotations

import pytest

from infra.helm_chart import HelmChart, HelmValues


class TestHelmValues:
    def test_defaults(self) -> None:
        v = HelmValues()
        assert v.replica_count == 3
        assert v.autoscaling_enabled is False

    def test_to_dict_structure(self) -> None:
        d = HelmValues().to_dict()
        assert "replicaCount" in d
        assert "image" in d
        assert "resources" in d
        assert "autoscaling" in d

    def test_image_dict(self) -> None:
        d = HelmValues(image_tag="abc123").to_dict()
        assert d["image"]["tag"] == "abc123"

    def test_resources_dict(self) -> None:
        d = HelmValues(cpu_request="250m", memory_limit="4Gi").to_dict()
        assert d["resources"]["requests"]["cpu"] == "250m"
        assert d["resources"]["limits"]["memory"] == "4Gi"

    def test_autoscaling_disabled(self) -> None:
        d = HelmValues(autoscaling_enabled=False).to_dict()
        assert d["autoscaling"]["enabled"] is False

    def test_autoscaling_enabled(self) -> None:
        d = HelmValues(autoscaling_enabled=True, min_replicas=2, max_replicas=8).to_dict()
        assert d["autoscaling"]["enabled"] is True
        assert d["autoscaling"]["maxReplicas"] == 8

    def test_replica_count_lt_1_raises(self) -> None:
        with pytest.raises(ValueError, match="replica_count"):
            HelmValues(replica_count=0)

    def test_min_gt_max_replicas_raises(self) -> None:
        with pytest.raises(ValueError, match="min_replicas"):
            HelmValues(min_replicas=10, max_replicas=5)

    def test_config_dict(self) -> None:
        d = HelmValues(mlflow_uri="http://mlflow:5000").to_dict()
        assert d["config"]["mlflowUri"] == "http://mlflow:5000"


class TestHelmChart:
    def _chart(self) -> HelmChart:
        return HelmChart(name="credit-risk", chart_version="0.1.0", app_version="1.2.0")

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            HelmChart(name="")

    def test_chart_yaml(self) -> None:
        d = self._chart().to_chart_yaml()
        assert d["name"] == "credit-risk"
        assert d["version"] == "0.1.0"
        assert d["appVersion"] == "1.2.0"
        assert d["apiVersion"] == "v2"
        assert d["type"] == "application"

    def test_values_dict(self) -> None:
        chart = self._chart()
        d = chart.to_values_dict()
        assert "replicaCount" in d

    def test_render_install_cmd_upgrade(self) -> None:
        cmd = self._chart().render_install_cmd()
        assert "helm upgrade --install" in cmd
        assert "credit-risk" in cmd
        assert "--namespace ml-serving" in cmd

    def test_render_install_cmd_no_upgrade(self) -> None:
        cmd = self._chart().render_install_cmd(upgrade=False)
        assert "helm install" in cmd
        assert "upgrade" not in cmd

    def test_render_install_cmd_extra_sets(self) -> None:
        cmd = self._chart().render_install_cmd(extra_sets={"image.tag": "abc123"})
        assert "--set image.tag=abc123" in cmd

    def test_render_install_cmd_custom_namespace(self) -> None:
        cmd = self._chart().render_install_cmd(namespace="ml-prod")
        assert "--namespace ml-prod" in cmd

    def test_default_description(self) -> None:
        d = self._chart().to_chart_yaml()
        assert "credit-risk" in d["description"]
