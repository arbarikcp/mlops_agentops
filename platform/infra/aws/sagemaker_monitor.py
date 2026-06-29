"""SageMaker Model Monitor and Clarify — automated drift and bias detection.

Day 83: SageMaker Monitor provides out-of-the-box drift/bias detection
without custom code. Four monitor types cover the full quality spectrum:
data quality (schema + statistics), model quality (prediction accuracy),
bias (Clarify fairness), and feature attribution (explainability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MonitorSchedule(str, Enum):
    HOURLY = "cron(0 * ? * * *)"
    DAILY = "cron(0 0 ? * * *)"
    WEEKLY = "cron(0 0 ? * 1 *)"


# ── Data Quality Monitor ───────────────────────────────────────────────────────


@dataclass
class MonitoringConstraints:
    """S3 location of baseline constraints (statistics/schema for comparison)."""

    s3_uri: str
    file_name: str = "constraints.json"

    def __post_init__(self) -> None:
        if not self.s3_uri:
            raise ValueError("s3_uri must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {"S3Uri": self.s3_uri.rstrip("/") + "/" + self.file_name}


@dataclass
class SMDataQualityMonitor:
    """SageMaker Data Quality Monitor — detects schema drift and statistical anomalies.

    Compares live data capture against a baseline computed from training data.
    Violations are published to CloudWatch for alerting.
    """

    monitor_name: str
    endpoint_name: str
    role_arn: str
    output_s3_uri: str
    baseline_constraints_s3: str
    baseline_statistics_s3: str
    schedule: MonitorSchedule = MonitorSchedule.DAILY
    instance_type: str = "ml.m5.xlarge"
    volume_size_gb: int = 20
    data_capture_s3_uri: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.monitor_name:
            raise ValueError("monitor_name must not be empty")
        if not self.endpoint_name:
            raise ValueError("endpoint_name must not be empty")
        if not self.role_arn:
            raise ValueError("role_arn must not be empty")
        if not self.output_s3_uri:
            raise ValueError("output_s3_uri must not be empty")
        if not self.baseline_constraints_s3:
            raise ValueError("baseline_constraints_s3 must not be empty")
        if not self.baseline_statistics_s3:
            raise ValueError("baseline_statistics_s3 must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "MonitoringScheduleName": self.monitor_name,
            "MonitoringType": "DataQuality",
            "MonitoringScheduleConfig": {
                "ScheduleConfig": {"ScheduleExpression": self.schedule.value},
                "MonitoringJobDefinition": {
                    "BaselineConfig": {
                        "ConstraintsResource": {"S3Uri": self.baseline_constraints_s3},
                        "StatisticsResource": {"S3Uri": self.baseline_statistics_s3},
                    },
                    "MonitoringInputs": [{
                        "EndpointInput": {
                            "EndpointName": self.endpoint_name,
                            "LocalPath": "/opt/ml/processing/input/endpoint",
                        }
                    }],
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [{
                            "S3Output": {
                                "S3Uri": self.output_s3_uri,
                                "LocalPath": "/opt/ml/processing/output",
                                "S3UploadMode": "EndOfJob",
                            }
                        }]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceType": self.instance_type,
                            "InstanceCount": 1,
                            "VolumeSizeInGB": self.volume_size_gb,
                        }
                    },
                    "RoleArn": self.role_arn,
                },
            },
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }

    @classmethod
    def for_endpoint(
        cls,
        endpoint_name: str,
        role_arn: str,
        bucket: str,
        schedule: MonitorSchedule = MonitorSchedule.DAILY,
    ) -> "SMDataQualityMonitor":
        """Factory: data quality monitor wired to a specific endpoint."""
        return cls(
            monitor_name=f"{endpoint_name}-dq-monitor",
            endpoint_name=endpoint_name,
            role_arn=role_arn,
            output_s3_uri=f"s3://{bucket}/monitors/{endpoint_name}/data-quality/",
            baseline_constraints_s3=f"s3://{bucket}/baselines/{endpoint_name}/constraints.json",
            baseline_statistics_s3=f"s3://{bucket}/baselines/{endpoint_name}/statistics.json",
            schedule=schedule,
        )


# ── Model Quality Monitor ──────────────────────────────────────────────────────


@dataclass
class SMModelQualityMonitor:
    """SageMaker Model Quality Monitor — tracks prediction accuracy over time.

    Joins predictions (from data capture) with ground-truth labels (from S3)
    and computes classification/regression metrics on a schedule.
    """

    monitor_name: str
    endpoint_name: str
    role_arn: str
    output_s3_uri: str
    ground_truth_s3_uri: str
    problem_type: str  # "BinaryClassification" | "MulticlassClassification" | "Regression"
    inference_attribute: str = "prediction"
    probability_attribute: str = "probability"
    probability_threshold: float = 0.5
    schedule: MonitorSchedule = MonitorSchedule.DAILY
    instance_type: str = "ml.m5.xlarge"
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.monitor_name:
            raise ValueError("monitor_name must not be empty")
        if not self.endpoint_name:
            raise ValueError("endpoint_name must not be empty")
        if not self.role_arn:
            raise ValueError("role_arn must not be empty")
        if not self.ground_truth_s3_uri:
            raise ValueError("ground_truth_s3_uri must not be empty")
        if self.problem_type not in (
            "BinaryClassification", "MulticlassClassification", "Regression"
        ):
            raise ValueError(f"problem_type invalid: {self.problem_type!r}")
        if not 0.0 <= self.probability_threshold <= 1.0:
            raise ValueError("probability_threshold must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "MonitoringScheduleName": self.monitor_name,
            "MonitoringType": "ModelQuality",
            "MonitoringScheduleConfig": {
                "ScheduleConfig": {"ScheduleExpression": self.schedule.value},
                "MonitoringJobDefinition": {
                    "ModelQualityAppSpecification": {
                        "ProblemType": self.problem_type,
                        "InferenceAttribute": self.inference_attribute,
                        "ProbabilityAttribute": self.probability_attribute,
                        "ProbabilityThresholdAttribute": self.probability_threshold,
                    },
                    "MonitoringInputs": [{
                        "EndpointInput": {
                            "EndpointName": self.endpoint_name,
                            "LocalPath": "/opt/ml/processing/input/endpoint",
                            "InferenceAttribute": self.inference_attribute,
                        }
                    }],
                    "GroundTruthS3Input": {"S3Uri": self.ground_truth_s3_uri},
                    "MonitoringOutputConfig": {
                        "MonitoringOutputs": [{
                            "S3Output": {
                                "S3Uri": self.output_s3_uri,
                                "LocalPath": "/opt/ml/processing/output",
                            }
                        }]
                    },
                    "MonitoringResources": {
                        "ClusterConfig": {
                            "InstanceType": self.instance_type,
                            "InstanceCount": 1,
                            "VolumeSizeInGB": 20,
                        }
                    },
                    "RoleArn": self.role_arn,
                },
            },
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }


# ── Clarify (Bias + Explainability) ───────────────────────────────────────────


@dataclass
class SMClarifyBiasConfig:
    """Clarify bias configuration — identifies which feature is the sensitive attribute."""

    label_name: str
    facet_name: str  # sensitive attribute (e.g. "age", "gender")
    facet_values_or_threshold: list[Any]  # values that constitute the disadvantaged group
    label_values_or_threshold: list[Any]  # positive label values (e.g. [1] for credit approval)

    def __post_init__(self) -> None:
        if not self.label_name:
            raise ValueError("label_name must not be empty")
        if not self.facet_name:
            raise ValueError("facet_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "LabelName": self.label_name,
            "Facet": [{"Name": self.facet_name, "ValueOrThreshold": self.facet_values_or_threshold}],
            "LabelValues": self.label_values_or_threshold,
        }


@dataclass
class SMClarifyConfig:
    """SageMaker Clarify configuration — bias detection and feature attribution.

    Clarify runs as a processing job that computes:
    - Pre-training bias metrics (before model training)
    - Post-training bias metrics (after model predictions)
    - SHAP feature importance scores
    """

    job_name: str
    role_arn: str
    model_name: str
    instance_type: str
    instance_count: int
    input_s3_uri: str
    output_s3_uri: str
    headers: list[str]
    label_column: str
    bias_config: SMClarifyBiasConfig | None = None
    enable_shap: bool = True
    shap_num_samples: int = 100
    probability_threshold: float = 0.5
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_name:
            raise ValueError("job_name must not be empty")
        if not self.role_arn:
            raise ValueError("role_arn must not be empty")
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not self.input_s3_uri:
            raise ValueError("input_s3_uri must not be empty")
        if not self.output_s3_uri:
            raise ValueError("output_s3_uri must not be empty")
        if not self.headers:
            raise ValueError("headers must not be empty")
        if not self.label_column:
            raise ValueError("label_column must not be empty")

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "ProcessingJobName": self.job_name,
            "RoleArn": self.role_arn,
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceType": self.instance_type,
                    "InstanceCount": self.instance_count,
                    "VolumeSizeInGB": 20,
                }
            },
            "ProcessingInputs": [{
                "InputName": "dataset",
                "S3Input": {
                    "S3Uri": self.input_s3_uri,
                    "LocalPath": "/opt/ml/processing/input",
                    "S3DataType": "S3Prefix",
                },
            }],
            "ProcessingOutputConfig": {"Outputs": [{
                "OutputName": "analysis_result",
                "S3Output": {
                    "S3Uri": self.output_s3_uri,
                    "LocalPath": "/opt/ml/processing/output",
                },
            }]},
            "ClarifyCheckConfig": {
                "ModelName": self.model_name,
                "Headers": self.headers,
                "Label": self.label_column,
                "ProbabilityThreshold": self.probability_threshold,
            },
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }
        if self.bias_config:
            cfg["ClarifyCheckConfig"]["BiasConfig"] = self.bias_config.to_dict()
        if self.enable_shap:
            cfg["ClarifyCheckConfig"]["ShapConfig"] = {
                "NumberOfSamples": self.shap_num_samples,
                "Seed": 42,
            }
        return cfg

    @classmethod
    def credit_risk_clarify(
        cls,
        role_arn: str,
        model_name: str,
        data_s3: str,
        output_s3: str,
    ) -> "SMClarifyConfig":
        """Factory: Clarify config for credit-risk bias (age as sensitive attribute)."""
        return cls(
            job_name=f"clarify-{model_name}",
            role_arn=role_arn,
            model_name=model_name,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            input_s3_uri=data_s3,
            output_s3_uri=output_s3,
            headers=["age", "income", "loan_amount", "credit_score", "label"],
            label_column="label",
            bias_config=SMClarifyBiasConfig(
                label_name="label",
                facet_name="age",
                facet_values_or_threshold=[30],  # age < 30 is disadvantaged group
                label_values_or_threshold=[1],   # 1 = approved
            ),
            enable_shap=True,
            shap_num_samples=200,
        )
