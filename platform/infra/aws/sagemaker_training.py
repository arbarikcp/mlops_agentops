"""SageMaker training, processing, and experiment builders.

Day 80: SageMaker abstracts away cluster management — managed spot training,
built-in experiment tracking, and automatic retry logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Data channel ──────────────────────────────────────────────────────────────


@dataclass
class DataChannel:
    """Input data channel for a SageMaker training job."""

    channel_name: str
    s3_uri: str
    content_type: str = "text/csv"
    input_mode: str = "File"  # "File" | "Pipe"

    def __post_init__(self) -> None:
        if not self.channel_name:
            raise ValueError("channel_name must not be empty")
        if not self.s3_uri:
            raise ValueError("s3_uri must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ChannelName": self.channel_name,
            "DataSource": {"S3DataSource": {"S3Uri": self.s3_uri, "S3DataType": "S3Prefix"}},
            "ContentType": self.content_type,
            "InputMode": self.input_mode,
        }


# ── Training job ──────────────────────────────────────────────────────────────


@dataclass
class SMTrainingJob:
    """SageMaker training job specification.

    Encapsulates the configuration for a managed training job — instance type,
    container image, hyperparameters, data channels, and output location.
    Spot training support reduces cost by up to 70%.
    """

    job_name: str
    role_arn: str
    image_uri: str
    instance_type: str
    instance_count: int
    output_s3_uri: str
    hyperparameters: dict[str, str] = field(default_factory=dict)
    data_channels: list[DataChannel] = field(default_factory=list)
    use_spot_instances: bool = True
    max_run_seconds: int = 3600
    max_wait_seconds: int = 7200  # must be >= max_run_seconds when spot enabled
    experiment_name: str = ""
    trial_name: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_name:
            raise ValueError("job_name must not be empty")
        if not self.role_arn:
            raise ValueError("role_arn must not be empty")
        if not self.image_uri:
            raise ValueError("image_uri must not be empty")
        if not self.instance_type:
            raise ValueError("instance_type must not be empty")
        if not self.output_s3_uri:
            raise ValueError("output_s3_uri must not be empty")
        if self.instance_count < 1:
            raise ValueError("instance_count must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "TrainingJobName": self.job_name,
            "RoleArn": self.role_arn,
            "AlgorithmSpecification": {
                "TrainingImage": self.image_uri,
                "TrainingInputMode": "File",
            },
            "ResourceConfig": {
                "InstanceType": self.instance_type,
                "InstanceCount": self.instance_count,
                "VolumeSizeInGB": 30,
            },
            "HyperParameters": self.hyperparameters,
            "InputDataConfig": [ch.to_dict() for ch in self.data_channels],
            "OutputDataConfig": {"S3OutputPath": self.output_s3_uri},
            "StoppingCondition": {
                "MaxRuntimeInSeconds": self.max_run_seconds,
                **({"MaxWaitTimeInSeconds": self.max_wait_seconds} if self.use_spot_instances else {}),
            },
            "EnableManagedSpotTraining": self.use_spot_instances,
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
        }
        if self.experiment_name:
            cfg["ExperimentConfig"] = {
                "ExperimentName": self.experiment_name,
                "TrialName": self.trial_name or self.job_name,
            }
        return cfg

    @classmethod
    def credit_risk_training(
        cls,
        job_name: str,
        role_arn: str,
        image_uri: str,
        train_s3: str,
        val_s3: str,
        output_s3: str,
        n_estimators: int = 200,
    ) -> "SMTrainingJob":
        """Factory for the credit-risk gradient-boosted model training job."""
        return cls(
            job_name=job_name,
            role_arn=role_arn,
            image_uri=image_uri,
            instance_type="ml.m5.xlarge",
            instance_count=1,
            output_s3_uri=output_s3,
            hyperparameters={
                "n_estimators": str(n_estimators),
                "max_depth": "6",
                "learning_rate": "0.05",
                "objective": "binary:logistic",
            },
            data_channels=[
                DataChannel("train", train_s3),
                DataChannel("validation", val_s3),
            ],
            use_spot_instances=True,
            tags={"Project": "credit-risk", "Phase": "12"},
        )


# ── Processing job ────────────────────────────────────────────────────────────


@dataclass
class ProcessingInput:
    """Input for a SageMaker processing job."""

    input_name: str
    s3_uri: str
    local_path: str

    def __post_init__(self) -> None:
        if not self.input_name:
            raise ValueError("input_name must not be empty")
        if not self.s3_uri:
            raise ValueError("s3_uri must not be empty")
        if not self.local_path:
            raise ValueError("local_path must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "InputName": self.input_name,
            "S3Input": {
                "S3Uri": self.s3_uri,
                "LocalPath": self.local_path,
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
            },
        }


@dataclass
class ProcessingOutput:
    """Output for a SageMaker processing job."""

    output_name: str
    s3_uri: str
    local_path: str

    def __post_init__(self) -> None:
        if not self.output_name:
            raise ValueError("output_name must not be empty")
        if not self.s3_uri:
            raise ValueError("s3_uri must not be empty")
        if not self.local_path:
            raise ValueError("local_path must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "OutputName": self.output_name,
            "S3Output": {
                "S3Uri": self.s3_uri,
                "LocalPath": self.local_path,
                "S3UploadMode": "EndOfJob",
            },
        }


@dataclass
class SMProcessingJob:
    """SageMaker processing job — data prep, feature engineering, or evaluation.

    SageMaker Processing jobs provide a managed, reproducible execution environment
    for any script that reads from S3 and writes back to S3.
    """

    job_name: str
    role_arn: str
    image_uri: str
    instance_type: str
    script_path: str
    inputs: list[ProcessingInput] = field(default_factory=list)
    outputs: list[ProcessingOutput] = field(default_factory=list)
    arguments: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_name:
            raise ValueError("job_name must not be empty")
        if not self.role_arn:
            raise ValueError("role_arn must not be empty")
        if not self.image_uri:
            raise ValueError("image_uri must not be empty")
        if not self.script_path:
            raise ValueError("script_path must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ProcessingJobName": self.job_name,
            "RoleArn": self.role_arn,
            "AppSpecification": {
                "ImageUri": self.image_uri,
                "ContainerEntrypoint": ["python3", self.script_path],
                "ContainerArguments": self.arguments,
            },
            "ProcessingResources": {
                "ClusterConfig": {
                    "InstanceType": self.instance_type,
                    "InstanceCount": 1,
                    "VolumeSizeInGB": 20,
                }
            },
            "ProcessingInputs": [i.to_dict() for i in self.inputs],
            "ProcessingOutputConfig": {"Outputs": [o.to_dict() for o in self.outputs]},
            "Environment": self.environment,
        }


# ── Experiment ────────────────────────────────────────────────────────────────


@dataclass
class SMTrialComponent:
    """A single component (step) within an SM trial — training, evaluation, etc."""

    component_name: str
    component_type: str  # "Training" | "Processing" | "Evaluation"
    job_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_name:
            raise ValueError("component_name must not be empty")
        if not self.job_name:
            raise ValueError("job_name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "TrialComponentName": self.component_name,
            "Type": self.component_type,
            "JobName": self.job_name,
            "Metrics": self.metrics,
            "Parameters": self.parameters,
        }


@dataclass
class SMExperiment:
    """SageMaker Experiment — groups related trials for comparison.

    An Experiment contains Trials; each Trial contains TrialComponents.
    Built-in integration with SageMaker jobs means metrics are automatically
    captured without explicit logging calls.
    """

    experiment_name: str
    description: str
    trials: dict[str, list[SMTrialComponent]] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_name:
            raise ValueError("experiment_name must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")

    def add_trial(self, trial_name: str, components: list[SMTrialComponent]) -> "SMExperiment":
        self.trials[trial_name] = components
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "ExperimentName": self.experiment_name,
            "Description": self.description,
            "Tags": [{"Key": k, "Value": v} for k, v in self.tags.items()],
            "Trials": {
                name: [c.to_dict() for c in comps]
                for name, comps in self.trials.items()
            },
        }

    @classmethod
    def credit_risk_experiment(cls) -> "SMExperiment":
        """Factory for credit-risk model comparison experiment."""
        return cls(
            experiment_name="credit-risk-model-selection",
            description="Compare XGBoost hyperparameter configurations for credit-risk model",
            tags={"Project": "credit-risk", "Phase": "12"},
        )
