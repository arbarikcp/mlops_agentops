"""Tests for monitoring/grafana_dashboard.py — PanelTarget, Panel, GrafanaDashboard."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoring.grafana_dashboard import GrafanaDashboard, Panel, PanelTarget


# ── PanelTarget ────────────────────────────────────────────────────────────────

class TestPanelTarget:
    def test_to_dict(self) -> None:
        t = PanelTarget("rate(requests_total[1m])", "RPS")
        d = t.to_dict()
        assert d["expr"] == "rate(requests_total[1m])"
        assert d["legendFormat"] == "RPS"
        assert d["refId"] == "A"


# ── Panel ──────────────────────────────────────────────────────────────────────

class TestPanel:
    def test_to_dict_basic(self) -> None:
        p = Panel("My Panel", targets=[PanelTarget("metric", "v")])
        d = p.to_dict()
        assert d["title"] == "My Panel"
        assert d["type"] == "timeseries"
        assert len(d["targets"]) == 1

    def test_empty_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            Panel("")

    def test_thresholds_in_field_config(self) -> None:
        p = Panel("P", thresholds=[{"value": 0.5, "color": "red"}])
        d = p.to_dict()
        steps = d["fieldConfig"]["defaults"]["thresholds"]["steps"]
        assert any(s["color"] == "red" for s in steps)

    def test_unit_propagated(self) -> None:
        p = Panel("P", unit="ms")
        d = p.to_dict()
        assert d["fieldConfig"]["defaults"]["unit"] == "ms"


# ── GrafanaDashboard ──────────────────────────────────────────────────────────

class TestGrafanaDashboard:
    def test_basic_creation(self) -> None:
        d = GrafanaDashboard("ML Dashboard", "ml-dash-v1")
        assert d.title == "ML Dashboard"
        assert d.uid == "ml-dash-v1"

    def test_empty_title_raises(self) -> None:
        with pytest.raises(ValueError, match="title"):
            GrafanaDashboard("", "uid")

    def test_empty_uid_raises(self) -> None:
        with pytest.raises(ValueError, match="uid"):
            GrafanaDashboard("Title", "")

    def test_add_panel_increments_count(self) -> None:
        d = GrafanaDashboard("D", "d1")
        d.add_panel(Panel("P1"))
        d.add_panel(Panel("P2"))
        assert d.panel_count() == 2

    def test_add_ml_golden_signals_adds_eight_panels(self) -> None:
        d = GrafanaDashboard("D", "d1")
        d.add_ml_golden_signals()
        assert d.panel_count() == 8

    def test_panel_ids_unique(self) -> None:
        d = GrafanaDashboard("D", "d1")
        d.add_ml_golden_signals()
        ids = [p.to_dict()["id"] for p in d._panels]
        assert len(ids) == len(set(ids))

    def test_to_dict_has_required_keys(self) -> None:
        d = GrafanaDashboard("D", "d1")
        result = d.to_dict()
        assert "title" in result
        assert "uid" in result
        assert "panels" in result
        assert "schemaVersion" in result

    def test_to_json_valid_json(self) -> None:
        d = GrafanaDashboard("D", "d1")
        d.add_ml_golden_signals()
        j = d.to_json()
        parsed = json.loads(j)
        assert parsed["uid"] == "d1"

    def test_save_writes_file(self, tmp_path: Path) -> None:
        d = GrafanaDashboard("D", "d1")
        d.add_ml_golden_signals()
        out = tmp_path / "dashboard.json"
        d.save(out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["uid"] == "d1"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        d = GrafanaDashboard("D", "d1")
        nested = tmp_path / "infra" / "grafana" / "dash.json"
        d.save(nested)
        assert nested.exists()

    def test_prefix_in_promql_expressions(self) -> None:
        d = GrafanaDashboard("D", "d1")
        d.add_ml_golden_signals(prefix="credit_risk")
        j = d.to_json()
        assert "credit_risk_prediction_requests_total" in j

    def test_ml_golden_signals_contains_auc_panel(self) -> None:
        d = GrafanaDashboard("D", "d1")
        d.add_ml_golden_signals()
        titles = [p.title for p in d._panels]
        assert any("AUC" in t for t in titles)
