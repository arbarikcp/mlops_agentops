# Day 19 — Data Contracts

## What is a Data Contract?

A **data contract** is a formal, versioned agreement between a data producer (upstream pipeline) and a data consumer (model training, serving, monitoring) that specifies:

- **Schema** — column names, types, nullability
- **Domain rules** — valid ranges, allowed values, uniqueness
- **Statistical invariants** — expected distributions, class balance, null rate bounds
- **Freshness** — how old data is allowed to be before it's considered stale
- **Ownership** — who is responsible for each field, who is accountable for violations
- **Enforcement mode** — whether violations raise an error (strict) or emit a warning (warn)

Without contracts, data quality issues silently degrade model performance. The contract is the specification; validation is the test.

---

## Why Contracts Break Down Without Formalisation

| Failure | Effect |
|---------|--------|
| Upstream adds a column, downstream assumes it exists | KeyError at serving time |
| Upstream changes EDUCATION encoding silently | Model sees out-of-distribution values |
| Null rate in LIMIT_BAL spikes from 0% to 8% | Feature distribution shift, AUC drops |
| Label arrives 3 months late, pipeline runs without waiting | Training on ~15% of available labels |
| Owner of PAY_0 leaves; nobody knows what the allowed values are | Undefined behaviour propagates for months |

---

## Three Layers of Enforcement

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 3 — Statistical (distribution-level)                          │
│     "The mean of LIMIT_BAL is within ±20% of the reference value"   │
│     "The null rate of AGE has not drifted above 0.5%"               │
│     "Class balance is still 15%–30%"                                │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 2 — Semantic / Domain (row-level, field-level)                │
│     "LIMIT_BAL is between 10,000 and 1,000,000"                     │
│     "SEX is in {1, 2}"                                              │
│     "utilization_rate is not infinite or NaN"                        │
├──────────────────────────────────────────────────────────────────────┤
│  Layer 1 — Syntactic / Schema (structural)                           │
│     "LIMIT_BAL is float, not string"                                │
│     "PAY_0 column exists"                                           │
│     "No duplicate IDs"                                              │
└──────────────────────────────────────────────────────────────────────┘
```

Layer 1 is cheapest to run and catches the most obvious failures. Layer 3 is most expensive but catches subtle drift. Run all three.

---

## Contract Versioning

A contract is versioned independently of the model. The version tuple is `(major, minor)`:

| Change | Version bump |
|--------|-------------|
| New column added (nullable) | `minor` |
| New column added (non-nullable) | `major` |
| Column removed | `major` |
| Valid range widened | `minor` |
| Valid range narrowed | `major` |
| Encoding changed (e.g. EDUCATION 0→4) | `major` |

When the contract major version changes, **all downstream consumers must be re-evaluated** before the new data enters production. This is enforced by the contract registry.

---

## Ownership Metadata

Each column has an owner (team or person) and a description. This is stored in the contract registry, not in the Pandera schema itself. The distinction:

- **Pandera** enforces *technical* rules (type, range, null)
- **Registry** records *organisational* rules (who cares, why it exists, what it means)

```
Column:   PAY_0
Owner:    data-engineering-team
SLA:      available by 08:00 UTC daily
Semantic: Repayment status for the most recent month.
          -2 = no consumption, -1 = paid on time, N = N months late.
          Values 0–8 indicate active delinquency.
          Source: core-banking transaction ledger, batch extract.
```

---

## Enforcement Modes

| Mode | Behaviour | Use case |
|------|-----------|----------|
| `strict` | Raise `SchemaError` — pipeline stops | Production ingestion |
| `warn` | Log WARNING — pipeline continues | Exploration, EDA |
| `log_only` | Log INFO — silent in CI | Development |

The enforcement mode is set per-contract-instance, not per-rule. You can override it at call time.

---

## Freshness

Freshness is not a Pandera check — it is a metadata check on file/table timestamps. The contract specifies `max_age_hours`. At validation time, the caller passes the data's `created_at` timestamp.

```python
# Credit card payment data arrives daily.
# If the file is > 26 hours old, something went wrong upstream.
max_age_hours = 26

if age_hours > max_age_hours:
    raise DataFreshnessError(f"Data is {age_hours:.1f}h old, max allowed: {max_age_hours}h")
```

---

## Pandera: Row-Level Validation

Pandera validates every row. For 30,000 rows and 32 columns, this is fast (<1 s). Key patterns:

### `Check.isin()` — categorical domain
```python
Column("SEX", int, checks=Check.isin([1, 2]), nullable=False)
```

### `Check.between()` — bounded numeric
```python
Column("LIMIT_BAL", float, checks=Check.between(10_000, 1_000_000))
```

### `Check.greater_than_or_equal_to(0)` — non-negative
```python
Column("PAY_AMT1", float, checks=Check.greater_than_or_equal_to(0))
```

### Custom check — derived feature invariant
```python
Column(
    "utilization_rate",
    float,
    checks=Check(lambda s: s.notna().all(), error="utilization_rate has NaN"),
    nullable=False,
)
```

### `lazy=True` — collect all failures before raising
```python
schema.validate(df, lazy=True)
# raises SchemaErrors with ALL failures, not just the first
```

Without `lazy=True`, Pandera stops at the first violation. In production you want the full list so you can triage all issues in one pass.

---

## Feature Schema: Semantic Checks on Derived Features

After `engineer_features()`, seven new columns are added. Their semantics:

| Feature | Expected range | Why |
|---------|---------------|-----|
| `utilization_rate` | `[-1, 20]` | BILL_AMT1 / (LIMIT_BAL+1); can be negative (credit balance) or large (heavily overdrawn) |
| `payment_ratio` | `[0, 100]` | PAY_AMT1 / (|BILL_AMT1|+1); >1 = overpaid; very large values are possible but extreme |
| `max_delay` | `[-2, 9]` | max of PAY_* columns; same domain as PAY_* status values |
| `avg_delay` | `[-2, 9]` | mean of PAY_* columns; float interpolation of the same range |
| `consecutive_delays` | `[0, 6]` | count of PAY_* > 0; 6 months max |
| `bill_trend` | `[-100, 100]` | (BILL_AMT1 - BILL_AMT6) / (|BILL_AMT6|+1); bounded by large bill swings |
| `total_payment_ratio` | `[0, 100]` | sum(PAY_AMT) / (sum(|BILL_AMT|)+1); bounded |

---

## Statistical Layer: Beyond Row Validation

Row-level checks don't catch distribution shift. The statistical layer computes dataset-level statistics and compares against a **reference snapshot** taken at a known-good time (typically training data split).

### Null drift
```
null_rate(column, current) - null_rate(column, reference) > threshold → ALERT
```

### Mean drift
```
|mean(column, current) - mean(column, reference)| / std(reference) > z_threshold → ALERT
```
A z-score of 3 corresponds to a 3-sigma shift — significant enough to be non-random.

### Population Stability Index (PSI) — preview for Day 21
The PSI is introduced briefly here because it bridges Layer 2 and Layer 3. Full detail in Day 21.

```
PSI = Σ (actual_pct_i - expected_pct_i) × ln(actual_pct_i / expected_pct_i)

PSI < 0.10   — stable, no action needed
PSI 0.10–0.20 — slight shift, monitor
PSI > 0.20   — major shift, investigate before training
```

---

## Code Walkthrough

### `feature_schema.py`

Pandera `DataFrameSchema` for the post-featurization dataset. Covers all 32 base columns + 7 derived columns. Uses `coerce=True` so float/int coercions are handled automatically.

Key design choice: the feature schema is **strict=False** (extra columns are allowed) because slice columns (EDUCATION, SEX, MARRIAGE) may still be present alongside feature columns.

### `contract_registry.py`

`ContractMetadata` — frozen dataclass:
- `name`: contract identifier (e.g. `"credit_feature_v1"`)
- `version`: semantic version string `"1.0"`
- `owner`: team/person email
- `description`: human-readable contract purpose
- `enforcement_mode`: `"strict"` | `"warn"` | `"log_only"`
- `max_age_hours`: freshness threshold (None = no freshness check)
- `schema`: the Pandera `DataFrameSchema` object

`ContractRegistry` — dict-like container for contracts. `validate()` method:
1. Looks up contract by name
2. Runs freshness check if `max_age_hours` set
3. Calls `schema.validate(df, lazy=True)`
4. On `SchemaErrors`: raise if `strict`, warn + return if `warn`, return quietly if `log_only`

### `statistical_checks.py`

`DatasetStats` — dataclass storing per-column statistics:
- null_rate, mean, std, p5, p25, p50, p75, p95, n_unique

`compute_dataset_stats()` — builds a `DatasetStats` snapshot from a DataFrame.

`check_null_drift()` — compares current null rates against reference. Returns a DataFrame of `(column, current_null_rate, reference_null_rate, drift, flag)`.

`check_mean_drift()` — z-score comparison of means. Flags columns where `|z| > threshold` (default 3.0).

`check_class_balance()` — from `raw_schema`, expanded to accept a tolerance parameter.

---

## How to Run

```bash
# Validate features.parquet against the feature contract (strict mode)
make data-contract

# Run all Day 19 unit tests
cd platform && uv run pytest tests/unit/test_feature_schema.py tests/unit/test_contract_registry.py tests/unit/test_statistical_checks.py -v

# Validate raw CSV directly
uv run python -m data.contracts.raw_schema data/raw/credit_card_default.csv --verbose
```

---

## Debugging Contract Failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `SchemaErrors: column 'utilization_rate' not found` | `engineer_features()` not run before validation | Ensure featurize step ran |
| `Check isin([1,2,3,4]) failed for EDUCATION` | Raw data not cleaned before feature schema validation | Run `clean_raw_data()` first |
| `null drift > 0.05 on BILL_AMT3` | Upstream extract missed a join | Check upstream pipeline logs |
| `mean_drift z-score = 4.2 on LIMIT_BAL` | New customer segment added to production | Re-evaluate model on new segment |
| `DataFreshnessError: data is 30h old` | Upstream cron job failed | Check upstream job status, re-run or alert |

---

## Key Invariants to Remember

1. **Raw schema validates before cleaning** — raw values include EDUCATION∈{0,5,6} which are cleaned away.
2. **Feature schema validates after cleaning AND featurization** — all 39 columns must be present.
3. **Statistical checks require a reference snapshot** — there's no check without a baseline.
4. **Enforcement mode is per-contract, not per-check** — you can't make one check strict and another warn inside the same contract instance (use separate contracts for separate severity levels).
5. **`lazy=True` always** — collect all failures, not just the first.
