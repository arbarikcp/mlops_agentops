"""GCP Vertex AI builders — 1:1 mapping with SageMaker equivalents.

Day 87: Vertex AI is GCP's managed ML platform. Understanding the 1:1 mapping
enables teams to reason about cloud-specific features without vendor lock-in:
  Vertex AI Training    ↔ SageMaker Training Jobs
  Vertex Model Registry ↔ SageMaker Model Registry
  Vertex Endpoints      ↔ SageMaker Real-Time Endpoints
  Vertex Pipelines      ↔ SageMaker Pipelines (both use KFP SDK)
  Vertex Monitoring     ↔ SageMaker Model Monitor
  Explainable AI        ↔ SageMaker Clarify
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Training ──────────────────────────────────────────────────────────────────


@dataclass
class VertexMachineSpec:
    """Vertex AI machine specification — instance type and accelerators."""

    machine_type: str  # e.g. "n1-standard-4", "a2-highgpu-1g"
    accelerator_type: str = ""  # e.g. "NVIDIA_TESLA_T4"
    accelerator_count: int = 0

    def __post_init__(self) -> None:
        if not self.machine_type:
            raise ValueError("machine_type must not be empty")

    def to_dict(self) -> dict[str, Any]:
        spec: dict[str, Any] = {"machineType": self.machine_type}
        if self.accelerator_type and self.accelerator_count > 0:
            spec["acceleratorConfig"] = {
                "type": self.accelerator_type,
                "count": self.accelerator_count,
            }
        return spec


@dataclass
class VertexTrainingJob:
    """Vertex AI Custom Training Job — equivalent to SageMaker Training Job.

    Key difference: Vertex uses Google Cloud Storage (GCS) for artifacts,
    Artifact Registry for images, and Vertex Experiments for tracking.
    Spot VMs on GCP called "Preemptible VMs" or "Spot VMs" (same 70-90% savings).
    """

    job_name: str
    project: str
    location: str
    image_uri: str  # Artifact Registry image URI
    gcs_output_uri: str
    machine_spec: VertexMachineSpec
    replica_count: int = 1
    args: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    use_spot: bool = True  # Preemptible VM
    experiment_name: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.job_name:
            raise ValueError("job_name must not be empty")
        if not self.project:
            raise ValueError("project must not be empty")
        if not self.image_uri:
            raise ValueError("image_uri must not be empty")
        if not self.gcs_output_uri:
            raise ValueError("gcs_output_uri must not be empty")
        if self.replica_count < 1:
            raise ValueError("replica_count must be >= 1")

    # AWS equivalent reference
    aws_equivalent: str = "SMTrainingJob"

    def to_dict(self) -> dict[str, Any]:
        worker_pool: dict[str, Any] = {
            "machineSpec": self.machine_spec.to_dict(),
            "replicaCount": self.replica_count,
            "containerSpec": {
                "imageUri": self.image_uri,
                "args": self.args,
                "env": [{"name": k, "value": v} for k, v in self.env_vars.items()],
            },
        }
        if self.use_spot:
            worker_pool["diskSpec"] = {"bootDiskType": "pd-ssd", "bootDiskSizeGb": 100}

        cfg: dict[str, Any] = {
            "displayName": self.job_name,
            "jobSpec": {
                "workerPoolSpecs": [worker_pool],
                "baseOutputDirectory": {"outputUriPrefix": self.gcs_output_uri},
                "enableWebAccess": False,
            },
            "labels": self.labels,
        }
        if self.use_spot:
            cfg["jobSpec"]["scheduling"] = {"restartJobOnWorkerRestart": True}
        if self.experiment_name:
            cfg["experimentConfig"] = {"experiment": self.experiment_name}
        return cfg


# ── Model Registry ────────────────────────────────────────────────────────────


@dataclass
class VertexModelPackage:
    """Vertex AI Model — equivalent to SageMaker Model Package.

    In Vertex, a Model resource stores the artifact URI and container image.
    Model Registry versions allow promotion workflows similar to SM registry
    approval gates.
    """

    display_name: str
    project: str
    location: str
    artifact_uri: str  # GCS URI: gs://bucket/model/
    serving_image_uri: str
    description: str = ""
    version_aliases: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.display_name:
            raise ValueError("display_name must not be empty")
        if not self.project:
            raise ValueError("project must not be empty")
        if not self.artifact_uri:
            raise ValueError("artifact_uri must not be empty")
        if not self.serving_image_uri:
            raise ValueError("serving_image_uri must not be empty")

    aws_equivalent: str = "SMModelPackage"

    def to_dict(self) -> dict[str, Any]:
        return {
            "displayName": self.display_name,
            "description": self.description,
            "artifactUri": self.artifact_uri,
            "containerSpec": {
                "imageUri": self.serving_image_uri,
                "predict_route": "/predict",
                "health_route": "/health",
            },
            "versionAliases": self.version_aliases,
            "labels": self.labels,
        }


# ── Endpoints ─────────────────────────────────────────────────────────────────


@dataclass
class VertexEndpoint:
    """Vertex AI Endpoint — equivalent to SageMaker real-time endpoint.

    Vertex Endpoints support traffic splitting for canary deployments
    (same as SageMaker production variants). Dedicated endpoints use
    reserved capacity; shared endpoints are serverless (similar to SM Serverless).
    """

    endpoint_name: str
    project: str
    location: str
    model_display_name: str
    machine_type: str = "n1-standard-2"
    min_replica_count: int = 1
    max_replica_count: int = 3
    traffic_split: dict[str, int] = field(default_factory=lambda: {"0": 100})
    enable_access_logging: bool = True
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.endpoint_name:
            raise ValueError("endpoint_name must not be empty")
        if not self.project:
            raise ValueError("project must not be empty")
        if not self.model_display_name:
            raise ValueError("model_display_name must not be empty")
        if self.min_replica_count < 1:
            raise ValueError("min_replica_count must be >= 1")
        if self.max_replica_count < self.min_replica_count:
            raise ValueError("max_replica_count must be >= min_replica_count")
        if sum(self.traffic_split.values()) != 100:
            raise ValueError("traffic_split values must sum to 100")

    aws_equivalent: str = "SMEndpoint"

    def to_dict(self) -> dict[str, Any]:
        return {
            "displayName": self.endpoint_name,
            "deployedModel": {
                "displayName": self.model_display_name,
                "dedicatedResources": {
                    "machineSpec": {"machineType": self.machine_type},
                    "minReplicaCount": self.min_replica_count,
                    "maxReplicaCount": self.max_replica_count,
                },
            },
            "trafficSplit": self.traffic_split,
            "enableAccessLogging": self.enable_access_logging,
            "labels": self.labels,
        }

    @classmethod
    def canary(
        cls,
        endpoint_name: str,
        project: str,
        location: str,
        model_display_name: str,
        canary_pct: int = 10,
    ) -> "VertexEndpoint":
        """Factory: canary endpoint with traffic split."""
        return cls(
            endpoint_name=endpoint_name,
            project=project,
            location=location,
            model_display_name=model_display_name,
            traffic_split={"0": 100 - canary_pct, "1": canary_pct},
        )


# ── Pipelines ─────────────────────────────────────────────────────────────────


@dataclass
class VertexPipelineComponent:
    """A single component (step) in a Vertex AI Pipeline (KFP-based)."""

    component_name: str
    image_uri: str
    command: list[str]
    args: list[str] = field(default_factory=list)
    input_artifacts: dict[str, str] = field(default_factory=dict)
    output_artifacts: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_name:
            raise ValueError("component_name must not be empty")
        if not self.image_uri:
            raise ValueError("image_uri must not be empty")
        if not self.command:
            raise ValueError("command must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.component_name,
            "implementation": {
                "container": {
                    "image": self.image_uri,
                    "command": self.command,
                    "args": self.args,
                }
            },
            "inputArtifacts": self.input_artifacts,
            "outputArtifacts": self.output_artifacts,
        }


@dataclass
class VertexPipeline:
    """Vertex AI Pipeline — equivalent to SageMaker Pipeline.

    Both use KFP (Kubeflow Pipelines) SDK under the hood, making Vertex
    Pipelines more portable than SageMaker Pipelines. The key difference:
    Vertex compiles pipelines to YAML (IR), SM uses JSON step definitions.
    """

    pipeline_name: str
    project: str
    location: str
    gcs_root: str  # GCS path for pipeline artifacts
    components: list[VertexPipelineComponent] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pipeline_name:
            raise ValueError("pipeline_name must not be empty")
        if not self.project:
            raise ValueError("project must not be empty")
        if not self.gcs_root:
            raise ValueError("gcs_root must not be empty")

    aws_equivalent: str = "SMPipeline"

    def add_component(self, component: VertexPipelineComponent) -> "VertexPipeline":
        self.components.append(component)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipelineName": self.pipeline_name,
            "pipelineRoot": self.gcs_root,
            "components": [c.to_dict() for c in self.components],
            "runtimeConfig": {"parameters": self.parameters},
            "labels": self.labels,
        }

    def pipeline_job_spec(self) -> dict[str, Any]:
        """Vertex PipelineJob creation spec."""
        return {
            "displayName": self.pipeline_name,
            "pipelineSpec": self.to_dict(),
            "runtimeConfig": {"gcsOutputDirectory": self.gcs_root},
        }

    @classmethod
    def credit_risk_pipeline(cls, project: str, gcs_bucket: str) -> "VertexPipeline":
        """Factory: credit-risk training pipeline on Vertex."""
        return cls(
            pipeline_name="credit-risk-vertex-pipeline",
            project=project,
            location="us-central1",
            gcs_root=f"gs://{gcs_bucket}/pipelines/credit-risk",
            parameters={"n_estimators": 200, "model_approval": "PendingManualApproval"},
            labels={"project": "credit-risk", "phase": "12"},
        )
