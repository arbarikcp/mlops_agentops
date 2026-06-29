"""Monitoring module — taxonomy, drift, Evidently, Prometheus, Grafana, logging, SLOs.

Phase 3 (Day 21):
  monitoring.reference_stats      — training-time feature stats
  monitoring.skew_detector        — PSI, KS test, JS divergence per feature

Phase 7 (Days 46–53):
  monitoring.taxonomy             — MonitorType, Severity, Monitor, MonitorRegistry
  monitoring.drift                — DriftDetector: PSI, KS, MMD, classifier-based
  monitoring.evidently_reporter   — Evidently adapter with DriftDetector fallback
  monitoring.prometheus_metrics   — MLMetricsCollector, text exposition format
  monitoring.grafana_dashboard    — GrafanaDashboard builder (JSON for GitOps)
  monitoring.prediction_logger    — PredictionLogger: JSONL audit log, correlation IDs
  monitoring.closed_loop          — ClosedLoop: 8-step orchestration, LoopApprover
  monitoring.slo                  — SLOChecker, SLOReport, BudgetStatus
"""
