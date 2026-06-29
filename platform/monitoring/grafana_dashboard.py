"""Grafana dashboard builder: generates ML golden signals dashboard JSON.

Day 50 — produces valid Grafana JSON from structured Python objects so dashboards
can be committed to Git and provisioned via Grafana's filesystem provisioner or
GitOps (Argo CD / Flux).

Classes:
  PanelTarget      — one PromQL query target in a panel
  Panel            — one Grafana panel (timeseries, stat, gauge)
  GrafanaDashboard — top-level dashboard with panel builder helpers

See: docs/phase7/day50_grafana.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# ── PanelTarget ───────────────────────────────────────────────────────────────

@dataclass
class PanelTarget:
    """One PromQL query target within a panel.

    Attributes:
        expr:          PromQL expression.
        legend_format: Legend label template (e.g., "{{feature}}").
        ref_id:        Grafana internal reference (A, B, C, …).
    """

    expr: str
    legend_format: str = ""
    ref_id: str = "A"

    def to_dict(self) -> dict:
        return {
            "expr": self.expr,
            "legendFormat": self.legend_format,
            "refId": self.ref_id,
        }


# ── Panel ─────────────────────────────────────────────────────────────────────

@dataclass
class Panel:
    """One Grafana panel.

    Attributes:
        title:       Panel title shown in the dashboard.
        panel_type:  "timeseries" / "stat" / "gauge" / "table".
        targets:     List of PanelTarget objects (one PromQL query each).
        thresholds:  List of {"value": float, "color": str} threshold steps.
        unit:        Grafana unit string (e.g., "ms", "percentunit", "short").
        panel_id:    Grafana panel ID (must be unique within dashboard).
        grid_pos:    {"x", "y", "w", "h"} grid position.
    """

    title: str
    panel_type: str = "timeseries"
    targets: list[PanelTarget] = field(default_factory=list)
    thresholds: list[dict] = field(default_factory=list)
    unit: str = "short"
    panel_id: int = 1
    grid_pos: dict = field(default_factory=lambda: {"x": 0, "y": 0, "w": 12, "h": 8})

    def __post_init__(self) -> None:
        if not self.title:
            raise ValueError("Panel title cannot be empty")

    def to_dict(self) -> dict:
        d: dict = {
            "id": self.panel_id,
            "title": self.title,
            "type": self.panel_type,
            "targets": [t.to_dict() for t in self.targets],
            "gridPos": self.grid_pos,
            "fieldConfig": {
                "defaults": {
                    "unit": self.unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": self.thresholds or [{"value": 0, "color": "green"}],
                    },
                }
            },
        }
        return d


# ── GrafanaDashboard ─────────────────────────────────────────────────────────

class GrafanaDashboard:
    """Builder for a Grafana dashboard JSON.

    Args:
        title:       Dashboard title.
        uid:         Unique dashboard identifier (used in URLs).
        description: Optional description shown in dashboard list.
        refresh:     Auto-refresh interval (e.g., "30s", "1m").
    """

    def __init__(
        self,
        title: str,
        uid: str,
        description: str = "",
        refresh: str = "30s",
    ) -> None:
        if not title:
            raise ValueError("Dashboard title cannot be empty")
        if not uid:
            raise ValueError("Dashboard uid cannot be empty")
        self.title = title
        self.uid = uid
        self.description = description
        self.refresh = refresh
        self._panels: list[Panel] = []
        self._next_id: int = 1
        self._next_y: int = 0

    def add_panel(self, panel: Panel) -> None:
        """Add a panel to the dashboard."""
        panel.panel_id = self._next_id
        self._next_id += 1
        self._panels.append(panel)
        self._next_y += panel.grid_pos.get("h", 8)

    def add_ml_golden_signals(self, prefix: str = "mlops") -> None:
        """Add the standard ML golden signals panels to the dashboard.

        Adds 8 panels in two rows:
          Row 1: Service Health  (RPS, Error rate, p99 latency, p50 latency)
          Row 2: ML Quality      (AUC, drift score, feature freshness, approval rate)
        """
        y = self._next_y

        # Row 1 — Service Health
        self.add_panel(Panel(
            title="Prediction RPS",
            panel_type="timeseries",
            targets=[PanelTarget(f"rate({prefix}_prediction_requests_total[1m])", "RPS")],
            unit="reqps",
            grid_pos={"x": 0, "y": y, "w": 6, "h": 8},
        ))
        self.add_panel(Panel(
            title="Error Rate %",
            panel_type="timeseries",
            targets=[PanelTarget(
                f"rate({prefix}_prediction_errors_total[1m]) / rate({prefix}_prediction_requests_total[1m])",
                "Error %",
            )],
            thresholds=[{"value": 0, "color": "green"}, {"value": 0.01, "color": "red"}],
            unit="percentunit",
            grid_pos={"x": 6, "y": y, "w": 6, "h": 8},
        ))
        self.add_panel(Panel(
            title="p99 Latency (ms)",
            panel_type="timeseries",
            targets=[PanelTarget(
                f"histogram_quantile(0.99, rate({prefix}_prediction_latency_ms_bucket[5m]))",
                "p99",
            )],
            thresholds=[{"value": 0, "color": "green"}, {"value": 500, "color": "yellow"}, {"value": 1000, "color": "red"}],
            unit="ms",
            grid_pos={"x": 12, "y": y, "w": 6, "h": 8},
        ))
        self.add_panel(Panel(
            title="p50 Latency (ms)",
            panel_type="timeseries",
            targets=[PanelTarget(
                f"histogram_quantile(0.50, rate({prefix}_prediction_latency_ms_bucket[5m]))",
                "p50",
            )],
            unit="ms",
            grid_pos={"x": 18, "y": y, "w": 6, "h": 8},
        ))

        y += 8

        # Row 2 — ML Quality
        self.add_panel(Panel(
            title="Model AUC",
            panel_type="stat",
            targets=[PanelTarget(f"{prefix}_model_auc", "AUC")],
            thresholds=[
                {"value": 0, "color": "red"},
                {"value": 0.72, "color": "yellow"},
                {"value": 0.78, "color": "green"},
            ],
            unit="short",
            grid_pos={"x": 0, "y": y, "w": 6, "h": 8},
        ))
        self.add_panel(Panel(
            title="Drift Score by Feature",
            panel_type="timeseries",
            targets=[PanelTarget(f"{prefix}_drift_score", "{{{{feature}}}}")],
            thresholds=[{"value": 0, "color": "green"}, {"value": 0.10, "color": "yellow"}, {"value": 0.20, "color": "red"}],
            unit="short",
            grid_pos={"x": 6, "y": y, "w": 6, "h": 8},
        ))
        self.add_panel(Panel(
            title="Feature Freshness (hours)",
            panel_type="timeseries",
            targets=[PanelTarget(f"{prefix}_feature_freshness_hours", "{{{{view}}}}")],
            thresholds=[{"value": 0, "color": "green"}, {"value": 25, "color": "yellow"}, {"value": 50, "color": "red"}],
            unit="h",
            grid_pos={"x": 12, "y": y, "w": 6, "h": 8},
        ))
        self.add_panel(Panel(
            title="Approval Rate",
            panel_type="stat",
            targets=[PanelTarget(f"{prefix}_approval_rate", "Approval Rate")],
            thresholds=[
                {"value": 0, "color": "red"},
                {"value": 0.60, "color": "yellow"},
                {"value": 0.70, "color": "green"},
            ],
            unit="percentunit",
            grid_pos={"x": 18, "y": y, "w": 6, "h": 8},
        ))

    def panel_count(self) -> int:
        return len(self._panels)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "uid": self.uid,
            "description": self.description,
            "refresh": self.refresh,
            "schemaVersion": 36,
            "version": 1,
            "panels": [p.to_dict() for p in self._panels],
            "time": {"from": "now-1h", "to": "now"},
            "timepicker": {},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save dashboard JSON to a file path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json())
