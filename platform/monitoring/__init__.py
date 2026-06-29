"""Monitoring module — skew detection, drift alerting, reference statistics.

Phase 3 (Day 21): Train/serve skew detection.
  monitoring.reference_stats  — compute and persist training-time feature stats
  monitoring.skew_detector    — PSI, KS test, JS divergence per feature

Phase 4+ will add:
  monitoring.drift_alerts     — alert routing and threshold management
  monitoring.model_quality    — outcome-based quality metrics (delayed labels)
  monitoring.infra_metrics    — latency, throughput, error rates
"""
