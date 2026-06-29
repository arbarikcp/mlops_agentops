# Day 46 — Monitoring Taxonomy: Operational vs ML vs Business

## Why Taxonomy Matters

A common anti-pattern is to treat all alerts the same. A p99 latency spike, an AUC decay,
and a drop in approval revenue are three completely different events — they have different
owners, different SLOs, different remediation paths, and different escalation routes.

Mixing them into a single alert channel means:
- SREs get paged for AUC drops they can't action
- Data scientists miss latency regressions that corrupt their features
- Finance never sees revenue drift until it's too late

The taxonomy enforces **alert routing by monitor type**.

---

## Three Monitor Types

```mermaid
graph TD
    A[All Monitors] --> B[Operational]
    A --> C[ML]
    A --> D[Business]

    B --> B1[Latency p99]
    B --> B2[Error Rate]
    B --> B3[CPU / Memory]
    B --> B4[Throughput RPS]

    C --> C1[Data Drift PSI/KS]
    C --> C2[Prediction Distribution]
    C --> C3[Model AUC Decay]
    C --> C4[Feature Freshness]

    D --> D1[Approval Rate]
    D --> D2[Default Rate]
    D --> D3[Revenue Impact]
    D --> D4[Cohort Outcomes]

    B --> E[Slack: #oncall-infra]
    C --> F[Slack: #ml-alerts]
    D --> G[Slack: #business-risk]
```

---

## Monitor Registry — Concepts

Each monitor has:
- **type** — `OPERATIONAL` / `ML` / `BUSINESS`
- **name** — unique identifier
- **check** — a callable that returns a `MonitorResult`
- **alert_channel** — routed by type automatically
- **severity** — `INFO` / `WARNING` / `CRITICAL`

```mermaid
classDiagram
    class MonitorType {
        <<enumeration>>
        OPERATIONAL
        ML
        BUSINESS
    }

    class Severity {
        <<enumeration>>
        INFO
        WARNING
        CRITICAL
    }

    class MonitorResult {
        +name: str
        +monitor_type: MonitorType
        +passed: bool
        +value: float
        +threshold: float
        +severity: Severity
        +message: str
        +alert_channel: str
    }

    class Monitor {
        +name: str
        +monitor_type: MonitorType
        +check_fn: Callable
        +threshold: float
        +severity: Severity
        +run() MonitorResult
    }

    class MonitorRegistry {
        +monitors: dict
        +register(monitor) void
        +run_all() list~MonitorResult~
        +run_by_type(MonitorType) list~MonitorResult~
        +failed_results() list~MonitorResult~
        +alert_channel(MonitorType) str
    }

    MonitorRegistry --> Monitor
    Monitor --> MonitorResult
    MonitorResult --> MonitorType
    MonitorResult --> Severity
```

---

## Alert Routing Table

| Monitor Type | Alert Channel | Owner | SLO | Remediation |
|---|---|---|---|---|
| OPERATIONAL | `#oncall-infra` | SRE | 99.9% uptime | Restart / scale out |
| ML | `#ml-alerts` | Data Scientist | AUC > 0.72 | Retrain / rollback |
| BUSINESS | `#business-risk` | Risk Officer | Approval rate 60–80% | Model review / policy update |

---

## Sequence: Monitor Run Cycle

```mermaid
sequenceDiagram
    participant S as Scheduler
    participant R as MonitorRegistry
    participant O as OperationalMonitor
    participant M as MLMonitor
    participant B as BusinessMonitor
    participant A as AlertRouter

    S->>R: run_all()
    R->>O: run()
    O-->>R: MonitorResult(OPERATIONAL, passed=True)
    R->>M: run()
    M-->>R: MonitorResult(ML, passed=False, severity=WARNING)
    R->>B: run()
    B-->>R: MonitorResult(BUSINESS, passed=True)
    R->>A: route(failed_results)
    A-->>S: alert sent to #ml-alerts
```

---

## Five Monitor Invariants

| # | Invariant |
|---|---|
| 1 | Every monitor has exactly one type — no "mixed" monitors |
| 2 | Alert channel is determined by type, not by individual monitor |
| 3 | `CRITICAL` monitors block the serving gate on failure |
| 4 | `WARNING` monitors emit alerts but do NOT block promotion |
| 5 | `INFO` monitors are logged but never alert |

---

## What This Enables in Phase 7

Day 46 builds the taxonomy skeleton. Each subsequent day adds monitors of specific types:
- Day 47 — ML drift monitors (type = `ML`)
- Day 49 — Prometheus operational monitors (type = `OPERATIONAL`)
- Day 51 — Prediction logger feeds business monitors (type = `BUSINESS`)
- Day 52 — Closed loop monitor (type = `ML`)
- Day 53 — SLOs wire all three types together
