# Day 83 — SageMaker Model Monitor & Clarify

## WHY — Automated Drift Detection Without Custom Code

Most teams detect model degradation the hard way: a business metric drops, someone
notices, they trace it back to data drift weeks later. Building a custom monitoring
pipeline requires ingesting prediction logs, computing baseline statistics, writing
drift metrics, and alerting — hundreds of lines of plumbing before any ML work.

SageMaker Model Monitor eliminates this plumbing:

| Concern | Custom monitoring | SageMaker Model Monitor |
|---|---|---|
| Log collection | Configure data capture, ship logs | One flag: `DataCaptureConfig(enable=True)` |
| Baseline statistics | Write Spark/Pandas jobs | `suggest_baseline()` from reference dataset |
| Drift detection | Implement statistical tests | Built-in (KL divergence, PSI, chi-squared) |
| Scheduling | Cron job + Lambda | `MonitoringSchedule` — cron built-in |
| Reports | Custom S3 + dashboard | Violation report JSON written to S3 automatically |
| Bias detection | Implement SHAP / Aequitas | SageMaker Clarify — no custom code |

> **Model Monitor is worth the vendor lock-in** at small-to-medium scale. The
> operational cost of building equivalent infrastructure exceeds the flexibility
> gained until you have a dedicated MLOps team.

---

## HOW — Four Monitor Types

SageMaker Model Monitor has four specialised monitor types:

| Monitor type | What it checks | Data source |
|---|---|---|
| **Data Quality** | Feature distribution drift vs baseline | Endpoint input/output capture |
| **Model Quality** | Prediction accuracy vs ground truth | Capture + ground truth merge |
| **Bias (Clarify)** | Fairness metrics vs baseline | Capture + labels |
| **Explainability (Clarify)** | Feature attribution drift | Capture |

---

## HOW — Data Quality Monitor

### Step 1: Enable data capture on the endpoint

```python
DataCaptureConfig(
    enable_capture=True,
    sampling_percentage=100,        # capture every request (reduce for high traffic)
    destination_s3_uri="s3://ml-monitoring/captured/credit-risk/",
    capture_options=["REQUEST", "RESPONSE"]
)
```

Captured data is written to S3 as JSON lines:
```json
{"captureData": {"endpointInput": {...}, "endpointOutput": {...}}, "eventMetadata": {...}}
```

### Step 2: Create a baseline from training data

```mermaid
flowchart LR
    A["Training dataset\ns3://ml-data/train/"] --> B["suggest_baseline()\nbaseline job"]
    B --> C["statistics.json\nper-feature: mean, std, quantiles"]
    B --> D["constraints.json\nper-feature: allowed range, completeness"]
    C --> E["MonitoringSchedule\ncompares live data to baseline"]
    D --> E
```

### Step 3: Schedule monitoring

The monitor runs as a Processing Job on a cron schedule, comparing captured
data to the baseline:

```
MonitoringSchedule:
  schedule_expression: "cron(0 * ? * * *)"   <- hourly
  monitoring_job_definition:
    baseline_config:
      statistics_resource: s3://.../statistics.json
      constraints_resource: s3://.../constraints.json
    monitoring_output: s3://ml-monitoring/reports/
    monitoring_resources: ml.m5.xlarge
```

### Violation report output

```json
{
  "violations": [
    {
      "feature_name": "credit_score",
      "constraint_check_type": "distribution_anchor_test",
      "description": "Inferred data type (Integral) does not match baseline (Fractional)"
    },
    {
      "feature_name": "annual_income",
      "constraint_check_type": "baseline_drift_check",
      "description": "p_value is 0.0003 <= threshold 0.05"
    }
  ]
}
```

---

## HOW — Model Quality Monitor

Model Quality monitoring compares **predictions against ground truth labels**.
This requires a merge step because ground truth arrives with a delay.

```mermaid
sequenceDiagram
    participant EP as Endpoint
    participant Capture as Data Capture (S3)
    participant GT as Ground Truth System
    participant MQ as Model Quality Monitor
    participant CW as CloudWatch

    EP->>Capture: Log prediction (inference_id=abc123, score=0.73)
    Note over GT: Customer defaults 30 days later
    GT->>Capture: Upload ground truth (inference_id=abc123, label=1)
    MQ->>Capture: Join predictions + ground truth by inference_id
    MQ->>MQ: Compute AUC, precision, recall, F1
    MQ->>CW: Emit metric: ModelQuality/AUC = 0.71
    Note over CW: Alarm if AUC < 0.80 baseline
    CW->>SNS: Alert "Model quality degraded"
```

The `inference_id` is the critical link — it must be set consistently in both
the prediction capture and the ground truth upload.

---

## HOW — SageMaker Clarify

Clarify detects **bias** in training data and **model predictions**, and computes
**feature attributions** (SHAP values) for explainability.

### Bias metrics computed by Clarify

| Metric | What it measures |
|---|---|
| Class Imbalance (CI) | Over/under-representation of a group in training data |
| Difference in Positive Proportions (DPP) | Difference in positive prediction rate across groups |
| Disparate Impact (DI) | Ratio of positive prediction rates (DI < 0.8 = disparate impact) |
| Accuracy Difference | Accuracy gap between demographic groups |

### Clarify processing job flow

```mermaid
flowchart TD
    A["Training data\n+ trained model"] --> B["ClarifyProcessor\nrun analysis"]
    B --> C{"Pre-training bias"}
    B --> D{"Post-training bias"}
    B --> E{"Feature attribution\nSHAP values"}
    C --> F["bias_report.json\nCI, DPP per sensitive feature"]
    D --> G["bias_report.json\nDI, Accuracy Diff per group"]
    E --> H["explainability_report.json\nSHAP per feature per prediction"]
    F --> I["Studio Bias Dashboard"]
    G --> I
    H --> I
```

---

## Data Structures — Class Diagram

```mermaid
classDiagram
    class SMDataQualityMonitor {
        +String monitor_name
        +String endpoint_name
        +String baseline_dataset_uri
        +String output_s3_uri
        +String instance_type
        +String schedule_expression
        +Dict constraints_violations_config
        +suggest_baseline() Dict
        +create_monitoring_schedule() str
        +list_executions() List
        +get_latest_report() Dict
    }

    class BaselineConfig {
        +String statistics_resource
        +String constraints_resource
        +to_dict() dict
    }

    class MonitoringViolation {
        +String feature_name
        +String constraint_check_type
        +String description
        +Float metric_value
        +Float threshold
        +is_critical() bool
    }

    class SMModelQualityMonitor {
        +String monitor_name
        +String endpoint_name
        +String ground_truth_s3_uri
        +String problem_type
        +String inference_attribute
        +String probability_attribute
        +String probability_threshold_attribute
        +suggest_baseline(training_dataset, target_attribute) Dict
        +create_monitoring_schedule() str
        +get_metrics() Dict
    }

    class SMClarifyConfig {
        +String model_name
        +Int instance_count
        +String instance_type
        +DataConfig data_config
        +ModelConfig model_config
        +BiasConfig bias_config
        +ExplainabilityConfig explainability_config
        +run() Dict
    }

    class BiasConfig {
        +String label_name
        +String facet_name
        +String facet_values_or_threshold
        +String group_name
        +to_dict() dict
    }

    class ExplainabilityConfig {
        +Int shap_num_samples
        +Bool shap_use_logit
        +Bool shap_save_local_shap_values
        +to_dict() dict
    }

    class MonitoringSchedule {
        +String schedule_name
        +String endpoint_name
        +String schedule_expression
        +String monitor_type
        +String status
        +pause() None
        +resume() None
        +delete() None
    }

    SMDataQualityMonitor "1" --> "1" BaselineConfig
    SMDataQualityMonitor "1" --> "many" MonitoringViolation
    SMDataQualityMonitor "1" --> "1" MonitoringSchedule
    SMModelQualityMonitor "1" --> "1" MonitoringSchedule
    SMClarifyConfig "1" --> "1" BiasConfig
    SMClarifyConfig "1" --> "1" ExplainabilityConfig
```

---

## HOW — Alerting on Violations

Model Monitor writes violation reports to S3 and emits CloudWatch metrics.
Connect these to operational alerts:

```mermaid
flowchart LR
    MM["Model Monitor\nViolation Report"] --> S3["S3\nviolations.json"]
    MM --> CW["CloudWatch Metric\nModelMonitor/DataQuality"]
    CW --> Alarm["CloudWatch Alarm\nViolationCount > 0"]
    Alarm --> SNS["SNS Topic"]
    SNS --> Slack["Slack / PagerDuty"]
    SNS --> Lambda["Lambda\nAuto-retrain trigger"]
    Lambda --> Pipeline["SageMaker Pipeline\nStartPipelineExecution"]
```

This pattern creates a **closed loop**: drift detected -> alert -> retrain ->
new model registered -> deployed -> monitoring continues.

---

## Key Takeaways

1. **Data capture is a one-flag prerequisite** — `DataCaptureConfig(enable_capture=True)` on the endpoint config; everything else builds on captured data.
2. **Baseline = statistical contract** — `suggest_baseline()` computes per-feature statistics and constraints from the training distribution; drift is measured against this contract.
3. **Model Quality requires ground truth** — use `inference_id` consistently in predictions and ground truth uploads; the monitor joins them by ID.
4. **Clarify quantifies fairness, not just accuracy** — run it on training data (pre-training) and production predictions (post-training) to detect disparate impact.
5. **SHAP via Clarify is operationally free** — no custom SHAP code; Clarify runs as a Processing Job with the same container as training.
6. **Wire violations to CloudWatch alarms** — connect Monitor -> CloudWatch -> SNS -> Lambda -> Pipeline for a fully automated drift-to-retrain loop.
