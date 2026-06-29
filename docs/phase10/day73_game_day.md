# Day 73 — Game Day + Runbooks + Postmortems

## What is a Game Day?

A **game day** is a scheduled, controlled chaos exercise where the team:

1. Announces the exercise (or keeps it surprise)
2. Injects real or simulated failures into production or staging
3. Observes response: did alerts fire, did on-call follow the runbook?
4. Debrief: what worked, what didn't, what needs updating

Game days validate the **entire sociotechnical system**: monitoring, runbooks, on-call
rotation, communication channels, and escalation paths — not just the code.

---

## Game Day Flow

```mermaid
sequenceDiagram
    participant Lead as Game Day Lead
    participant On as On-Call Eng
    participant Sys as System
    participant Mon as Monitoring
    participant Run as Runbook

    Lead->>Lead: scope experiment (blast radius, rollback plan)
    Lead->>On: brief on-call (time window, rules of engagement)
    Lead->>Sys: inject failure (Phase 10 chaos scenarios)
    Sys-->>Mon: SLI metrics change
    Mon->>On: alert fires (or doesn't — that's a finding!)
    On->>Run: open runbook, follow steps
    Run->>Sys: recovery actions
    Sys-->>Mon: steady state restored
    Mon->>Lead: confirm recovery
    Lead->>Lead: record duration, errors, gaps
    Lead->>Lead: write postmortem
```

---

## Runbook Template

Every ML system incident needs a runbook at `docs/runbooks/<incident-name>.md`:

```
# Runbook: <incident-name>

## Alert
<Alert name> fires when <condition> for <duration>.

## Immediate Steps (first 5 minutes)
1. Acknowledge alert in PagerDuty / Slack
2. Check dashboard: <Grafana link>
3. Confirm symptom: <specific kubectl / curl command>

## Root Cause (investigate)
- [ ] Check <subsystem A>: <command>
- [ ] Check <subsystem B>: <command>

## Recovery
1. <Step 1 with exact command>
2. <Step 2>
3. Verify: <command to confirm recovery>

## Escalate if
- Recovery > 30 min
- Multiple SLOs breached simultaneously
- Data loss suspected

## Postmortem trigger
Any SLO breach, or if detection took > 15 min.
```

---

## Postmortem Template

A blameless postmortem focuses on **system improvements**, not people.

```
# Postmortem: <incident name> — <date>

## Summary
One paragraph: what happened, impact, duration.

## Timeline (UTC)
| Time | Event |
|---|---|
| 08:00 | Materialization job fails (no alert) |
| 14:00 | On-call notices predictions look odd |
| 14:10 | Root cause identified: stale features |
| 14:30 | Re-materialization complete; SLO restored |

## Impact
- Error budget consumed: X%
- Affected users / predictions: N
- Business impact: approval rate off by ±Y%

## Root Cause
<Technical description of what failed and why>

## Contributing Factors
- No alert on materialization failure
- Feature freshness not checked at inference time

## Action Items
| Action | Owner | Due |
|---|---|---|
| Add materialization failure alert | Data Eng | 2026-07-06 |
| FeatureMonitor.check_freshness() in serving path | ML Eng | 2026-07-06 |

## What went well
- Recovery runbook existed and was accurate
- P50 latency unaffected (silent degradation caught within 6h)
```

---

## Phase 10 Game Day Checklist

### Before the exercise

```
☐ Experiment scope documented (which scenarios, which environment)
☐ Rollback plan written and tested
☐ Monitoring dashboards accessible
☐ Runbooks exist for each scenario
☐ On-call briefed on time window
☐ Executive stakeholders informed (if prod)
```

### During the exercise

```
☐ Start timer at failure injection
☐ Note exact time alerts fire
☐ Note which runbook steps were unclear or wrong
☐ Do NOT help on-call — observe only (game day lead role)
☐ Abort if blast radius exceeds plan
```

### After the exercise

```
☐ Write postmortem within 48h
☐ File action items in Linear / Jira
☐ Update runbooks with corrections
☐ Re-schedule game day in 6 weeks to verify fixes
```

---

## Class Diagram

```mermaid
classDiagram
    class GameDay {
        +name: str
        +scenarios: list~ChaosScenario~
        +scheduled_date: str
        +environment: str
        +rollback_plan: str
        +run_dry() GameDayReport
    }

    class GameDayReport {
        +name: str
        +total_scenarios: int
        +scenarios_passed: int
        +slo_breaches: int
        +action_items: list~str~
        +pass_rate() float
    }

    class Runbook {
        +incident_name: str
        +alert_name: str
        +immediate_steps: list~str~
        +investigation_steps: list~str~
        +recovery_steps: list~str~
        +escalation_criteria: list~str~
        +is_complete() bool
    }

    class Postmortem {
        +incident_name: str
        +date: str
        +summary: str
        +timeline: list~dict~
        +root_cause: str
        +action_items: list~dict~
        +is_blameless() bool
    }

    GameDay --> GameDayReport
    GameDay --> Runbook
    GameDayReport --> Postmortem
```
