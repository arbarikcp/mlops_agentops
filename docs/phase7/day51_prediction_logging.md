# Day 51 — Prediction Logging for Audit/Replay

## Why Log Predictions?

Predictions are ephemeral — without logging, you cannot:
1. **Debug** — reproduce a prediction that produced a wrong outcome
2. **Audit** — prove to regulators what the model decided for customer X on date Y
3. **Replay** — re-score historical predictions with a new model to estimate impact before promotion
4. **Feedback loop** — join predictions to outcomes (Phase 6 — Day 44)

A production prediction log is a **first-class artifact**, not an afterthought.

---

## Structured Log Schema

Every prediction log entry must contain:

| Field | Type | Description |
|---|---|---|
| `prediction_id` | str | UUID — unique per request, used for outcome join |
| `correlation_id` | str | Request-level ID for tracing across microservices |
| `entity_key` | str | Customer / entity identifier |
| `model_version` | str | Model version / run_id |
| `score` | float | Raw probability output (not thresholded) |
| `decision` | str | approve / review / decline |
| `features` | dict | Feature snapshot at inference time |
| `prediction_ts` | ISO8601 | When the prediction was made |
| `latency_ms` | float | How long inference took |
| `environment` | str | prod / staging / shadow |

---

## Correlation IDs

Every request generates two IDs:
- **correlation_id** — shared across all microservices for one request (e.g., same value in gateway, feature store, model API)
- **prediction_id** — unique to this model prediction (enables outcome join)

```mermaid
sequenceDiagram
    participant GW as API Gateway
    participant FS as Feature Store
    participant M as Model API
    participant L as PredictionLogger

    GW->>GW: generate correlation_id = "req-abc123"
    GW->>FS: GET /features (X-Correlation-ID: req-abc123)
    FS->>M: POST /predict (X-Correlation-ID: req-abc123)
    M->>M: generate prediction_id = "pred-xyz789"
    M->>L: log(prediction_id, correlation_id, score, features)
    L->>L: write JSON line to log store
    M-->>GW: {score, decision, prediction_id}
```

---

## PredictionLogger Class Diagram

```mermaid
classDiagram
    class PredictionLogEntry {
        +prediction_id: str
        +correlation_id: str
        +entity_key: str
        +model_version: str
        +score: float
        +decision: str
        +features: dict
        +prediction_ts: datetime
        +latency_ms: float
        +environment: str
        +to_dict() dict
        +to_json() str
    }

    class PredictionLogger {
        +log_path: str
        +environment: str
        +model_version: str
        +log(entity_key, score, decision, features, latency_ms, correlation_id) PredictionLogEntry
        +read_log(n_last) list~PredictionLogEntry~
        +flush() void
    }

    PredictionLogger --> PredictionLogEntry
```

---

## Log Storage Strategies

| Strategy | Format | Use case |
|---|---|---|
| **Local JSONL** | One JSON object per line | Dev, testing, small volume |
| **S3 / MinIO** | Parquet partitioned by date | Production — efficient batch read |
| **Kafka** | JSON stream | Real-time feedback loop integration |
| **Database** | PostgreSQL table | Audit trail — queryable, indexed |

Our implementation uses local JSONL (easy to test) with the same schema that works
for all other backends.

---

## Replay Use Case

Before promoting a new model, replay recent predictions through it:

```
1. Read last 30 days from prediction log
2. Re-score each entry with new model (same features captured in log)
3. Compare: new_score vs logged_score, new_decision vs logged_decision
4. Compute: AUC delta, approval rate delta, decision flip rate
5. If delta > threshold → human review required
```

The `features` field in `PredictionLogEntry` is the crucial replay enabler.
