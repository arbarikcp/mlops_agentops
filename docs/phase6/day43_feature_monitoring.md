# Day 43 — Feature Monitoring

## Why Feature Monitoring Is Distinct From Model Monitoring

Model monitoring detects that **predictions have changed**. Feature monitoring detects **why**.

```
Alert: approval_rate dropped 8% this week
  → Model monitoring: prediction distribution shifted
  → Feature monitoring: util_rate mean jumped from 0.45 → 0.72
    → Root cause: upstream payment processor changed billing cycle
```

Feature monitoring is earlier in the signal chain — it catches issues before they reach the model.

---

## Three Monitoring Pillars

| Pillar | What it checks | Example failure |
|---|---|---|
| **Freshness** | Was the feature materialised recently enough? | Cron job failed; online store has 3-day-old data |
| **Data quality** | Are values within expected ranges? | `util_rate` suddenly 150% (>1.0) — impossible |
| **Feature drift** | Has the distribution shifted from training? | `avg_pay_ratio_6m` mean dropped due to COVID policy |

---

## Freshness Monitoring

Each feature view has a `ttl_days` and an expected materialization frequency. Freshness
monitoring compares `now()` with `last_materialized_at` from the registry.

```
FreshnessStatus:
  FRESH   → now - last_materialized_at < threshold_hours
  STALE   → threshold_hours ≤ now - last_materialized_at < 3 × threshold_hours
  MISSING → last_materialized_at is None or > 3 × threshold_hours
```

Alert on STALE; page on MISSING.

---

## Data Quality

Checks per-column constraints at inference time (or as a daily batch scan):

| Check | Example threshold | Severity |
|---|---|---|
| Null rate | < 1% for required features | CRITICAL if > 5% |
| Out-of-range rate | 0% for bounded features | WARNING if > 0.1% |
| Constant column | Variance > 0 | WARNING if stddev = 0 for > 1 day |
| Impossible values | `util_rate` ∈ [0, 1] | ERROR if any > 1.0 |

---

## Feature Drift

Uses the same PSI and KS statistics as the train/serve skew detector (Phase 3), but applied
**per feature view** on a rolling basis:

- **Reference distribution** → training snapshot (DVC-managed)
- **Current distribution** → last 7 days of inference feature log

| Metric | NONE | LOW | HIGH |
|---|---|---|---|
| PSI | < 0.10 | 0.10–0.20 | > 0.20 |
| KS stat | < 0.05 | 0.05–0.10 | > 0.10 |

HIGH drift on any feature → alert + trigger model revalidation.

---

## Class Diagram

```mermaid
classDiagram
    class FreshnessStatus {
        <<enumeration>>
        FRESH
        STALE
        MISSING
    }

    class FreshnessCheck {
        +feature_view_name: str
        +last_materialized_at: datetime
        +threshold_hours: float
        +status: FreshnessStatus
        +age_hours: float
    }

    class FeatureQualityResult {
        +feature_name: str
        +null_rate: float
        +out_of_range_rate: float
        +min_val: float
        +max_val: float
        +passed: bool
        +issues: list~str~
    }

    class FeatureDriftResult {
        +feature_name: str
        +psi: float
        +ks_stat: float
        +severity: str
    }

    class FeatureMonitorReport {
        +freshness: list~FreshnessCheck~
        +quality: list~FeatureQualityResult~
        +drift: list~FeatureDriftResult~
        +overall_passed: bool
        +summary() str
    }

    class FreshnessChecker {
        +check(name, last_mat, threshold_hours) FreshnessCheck
    }

    class FeatureQualityChecker {
        +check(df, bounds) list~FeatureQualityResult~
    }

    class FeatureDriftMonitor {
        +check(reference_df, current_df, features) list~FeatureDriftResult~
    }

    class FeatureMonitor {
        +freshness_checker: FreshnessChecker
        +quality_checker: FeatureQualityChecker
        +drift_monitor: FeatureDriftMonitor
        +run(reference_df, current_df, bounds) FeatureMonitorReport
    }

    FeatureMonitor --> FreshnessChecker
    FeatureMonitor --> FeatureQualityChecker
    FeatureMonitor --> FeatureDriftMonitor
    FeatureMonitorReport --> FreshnessCheck
    FeatureMonitorReport --> FeatureQualityResult
    FeatureMonitorReport --> FeatureDriftResult
```

---

## Monitoring Sequence

```mermaid
sequenceDiagram
    participant SCHED as Scheduler
    participant FM as FeatureMonitor
    participant REG as FeatureRegistry
    participant OFF as OfflineStore

    SCHED->>FM: run(reference_df, current_df, bounds)
    FM->>REG: last_materialized_at(each feature view)
    REG-->>FM: timestamps
    FM->>FM: FreshnessChecker.check(timestamps)
    FM->>FM: FeatureQualityChecker.check(current_df, bounds)
    FM->>FM: FeatureDriftMonitor.check(reference_df, current_df)
    FM-->>SCHED: FeatureMonitorReport
    SCHED->>SCHED: alert if any severity == HIGH or MISSING
```

---

## Alerting Strategy

| Condition | Action |
|---|---|
| Any feature MISSING | Page on-call immediately |
| Any feature STALE | Slack alert + auto-retry materialization |
| Any feature PSI HIGH | Slack alert + block next model promotion |
| Any null rate > 5% | Slack alert |
| Any out-of-range > 0.1% | Log + investigate |
