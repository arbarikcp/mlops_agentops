"""Platform portability — cloud-agnostic core and provider adapters.

Day 88: The ML platform core (MLflow + Feast + K8s) is intentionally
cloud-agnostic. Only the outer "shell" (training compute, managed endpoints,
object storage) is cloud-specific. PortabilityMatrix quantifies what you
keep vs what changes when switching clouds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PortabilityLevel(str, Enum):
    """How portable a component is across cloud providers."""
    FULLY_PORTABLE = "fully_portable"   # identical API, no changes needed
    ADAPTER_NEEDED = "adapter_needed"   # thin wrapper, 1-2 days to swap
    REWRITE_NEEDED = "rewrite_needed"   # significant effort, week+
    CLOUD_SPECIFIC = "cloud_specific"   # inherently tied to one provider


class CloudProvider(str, Enum):
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    LOCAL = "local"


# ── Portability Matrix ─────────────────────────────────────────────────────────


@dataclass
class MatrixEntry:
    """A single row in the portability matrix — one platform component."""

    component: str
    category: str  # e.g. "experiment_tracking", "serving", "storage"
    portability_level: PortabilityLevel
    aws_impl: str
    gcp_impl: str
    azure_impl: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.component:
            raise ValueError("component must not be empty")
        if not self.category:
            raise ValueError("category must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "category": self.category,
            "portabilityLevel": self.portability_level.value,
            "implementations": {
                "aws": self.aws_impl,
                "gcp": self.gcp_impl,
                "azure": self.azure_impl,
            },
            "notes": self.notes,
        }

    @property
    def is_portable(self) -> bool:
        return self.portability_level in (
            PortabilityLevel.FULLY_PORTABLE,
            PortabilityLevel.ADAPTER_NEEDED,
        )


@dataclass
class PortabilityMatrix:
    """Complete portability matrix for the ML platform.

    Documents which components are portable (use anywhere) vs cloud-specific
    (must be replaced when switching providers). Core principle: keep the
    portable core large, cloud-specific shell small.
    """

    entries: list[MatrixEntry] = field(default_factory=list)

    def add_entry(self, entry: MatrixEntry) -> "PortabilityMatrix":
        self.entries.append(entry)
        return self

    def by_category(self) -> dict[str, list[MatrixEntry]]:
        result: dict[str, list[MatrixEntry]] = {}
        for e in self.entries:
            result.setdefault(e.category, []).append(e)
        return result

    def portable_components(self) -> list[MatrixEntry]:
        return [e for e in self.entries if e.is_portable]

    def cloud_specific_components(self) -> list[MatrixEntry]:
        return [e for e in self.entries if not e.is_portable]

    def portability_score(self) -> float:
        """Fraction of components that are portable (0.0–1.0)."""
        if not self.entries:
            return 0.0
        return len(self.portable_components()) / len(self.entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "summary": {
                "total": len(self.entries),
                "portable": len(self.portable_components()),
                "cloudSpecific": len(self.cloud_specific_components()),
                "portabilityScore": round(self.portability_score(), 2),
            },
        }

    @classmethod
    def ml_platform_matrix(cls) -> "PortabilityMatrix":
        """Factory: standard ML platform portability matrix."""
        matrix = cls()
        matrix.add_entry(MatrixEntry(
            "MLflow tracking", "experiment_tracking",
            PortabilityLevel.FULLY_PORTABLE,
            "MLflow on EKS", "MLflow on GKE", "MLflow on AKS",
            "Same API across all clouds; only storage backend changes",
        ))
        matrix.add_entry(MatrixEntry(
            "Feast feature store", "feature_store",
            PortabilityLevel.ADAPTER_NEEDED,
            "Feast + S3 + RDS", "Feast + GCS + BigQuery", "Feast + ADLS + AzureSQL",
            "Feature definitions portable; offline store backend changes",
        ))
        matrix.add_entry(MatrixEntry(
            "Kubernetes workloads", "orchestration",
            PortabilityLevel.FULLY_PORTABLE,
            "EKS", "GKE", "AKS",
            "Standard K8s APIs: Deployments, Services, HPAs, etc.",
        ))
        matrix.add_entry(MatrixEntry(
            "Argo Workflows", "pipeline_orchestration",
            PortabilityLevel.FULLY_PORTABLE,
            "Argo on EKS", "Argo on GKE", "Argo on AKS",
            "K8s CRD — identical across cloud-managed K8s",
        ))
        matrix.add_entry(MatrixEntry(
            "Helm charts", "deployment",
            PortabilityLevel.FULLY_PORTABLE,
            "Helm on EKS", "Helm on GKE", "Helm on AKS",
            "Cloud-agnostic packaging; values files may differ per env",
        ))
        matrix.add_entry(MatrixEntry(
            "Object storage (DVC)", "data_versioning",
            PortabilityLevel.ADAPTER_NEEDED,
            "S3", "GCS", "Azure Blob Storage",
            "DVC supports all; change remote URL and credentials",
        ))
        matrix.add_entry(MatrixEntry(
            "Container registry", "artifact_registry",
            PortabilityLevel.ADAPTER_NEEDED,
            "ECR", "Artifact Registry", "ACR",
            "OCI standard images; change registry URL in CI/CD",
        ))
        matrix.add_entry(MatrixEntry(
            "Managed training", "training",
            PortabilityLevel.CLOUD_SPECIFIC,
            "SageMaker Training", "Vertex AI Training", "Azure ML Training",
            "Rewrite job spec per cloud; consider Argo for portability",
        ))
        matrix.add_entry(MatrixEntry(
            "Managed endpoints", "serving",
            PortabilityLevel.CLOUD_SPECIFIC,
            "SageMaker Endpoints", "Vertex AI Endpoints", "Azure ML Endpoints",
            "KServe on K8s is the portable alternative",
        ))
        matrix.add_entry(MatrixEntry(
            "IAM / Identity", "security",
            PortabilityLevel.REWRITE_NEEDED,
            "AWS IAM + IRSA", "GCP Workload Identity", "Azure Managed Identity",
            "Concepts map 1:1 but APIs are completely different",
        ))
        return matrix


# ── Cloud Adapter ─────────────────────────────────────────────────────────────


@dataclass
class CloudAdapter:
    """Abstract cloud adapter — wraps provider-specific clients behind a common interface.

    Implements the Adapter pattern: the ML platform core calls CloudAdapter
    methods (upload_artifact, get_artifact, etc.) without knowing which cloud
    is underneath. Only the concrete adapter changes when switching providers.
    """

    provider: CloudProvider
    region: str
    credentials_source: str  # "iam_role" | "workload_identity" | "managed_identity" | "env"

    def __post_init__(self) -> None:
        if not self.region:
            raise ValueError("region must not be empty")
        if not self.credentials_source:
            raise ValueError("credentials_source must not be empty")

    def storage_uri(self, bucket: str, key: str) -> str:
        """Return provider-appropriate storage URI."""
        if self.provider == CloudProvider.AWS:
            return f"s3://{bucket}/{key}"
        elif self.provider == CloudProvider.GCP:
            return f"gs://{bucket}/{key}"
        elif self.provider == CloudProvider.AZURE:
            return f"abfs://{bucket}@{self.region}.dfs.core.windows.net/{key}"
        else:
            return f"file://{bucket}/{key}"

    def registry_uri(self, repo: str, tag: str = "latest") -> str:
        """Return provider-appropriate container registry URI."""
        if self.provider == CloudProvider.AWS:
            return f"{self.credentials_source}.dkr.ecr.{self.region}.amazonaws.com/{repo}:{tag}"
        elif self.provider == CloudProvider.GCP:
            return f"us-central1-docker.pkg.dev/{self.credentials_source}/{repo}:{tag}"
        elif self.provider == CloudProvider.AZURE:
            return f"{self.credentials_source}.azurecr.io/{repo}:{tag}"
        else:
            return f"localhost:5000/{repo}:{tag}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "region": self.region,
            "credentialsSource": self.credentials_source,
            "storageScheme": {"aws": "s3", "gcp": "gs", "azure": "abfs", "local": "file"}.get(self.provider.value, "file"),
        }

    @classmethod
    def aws(cls, region: str = "us-east-1", role_arn: str = "iam_role") -> "CloudAdapter":
        return cls(CloudProvider.AWS, region, role_arn)

    @classmethod
    def gcp(cls, project: str, region: str = "us-central1") -> "CloudAdapter":
        return cls(CloudProvider.GCP, region, project)

    @classmethod
    def local(cls) -> "CloudAdapter":
        return cls(CloudProvider.LOCAL, "local", "env")


# ── Portability Score ──────────────────────────────────────────────────────────


@dataclass
class PortabilityScore:
    """Quantified portability assessment for the ML platform.

    Combines the PortabilityMatrix score with migration effort estimate
    to produce an actionable portability report.
    """

    platform_name: str
    matrix: PortabilityMatrix
    migration_target: CloudProvider
    estimated_migration_days: int
    blockers: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.platform_name:
            raise ValueError("platform_name must not be empty")
        if self.estimated_migration_days < 0:
            raise ValueError("estimated_migration_days must be >= 0")

    @property
    def score(self) -> float:
        return self.matrix.portability_score()

    @property
    def grade(self) -> str:
        s = self.score
        if s >= 0.8:
            return "A"
        elif s >= 0.6:
            return "B"
        elif s >= 0.4:
            return "C"
        else:
            return "D"

    def to_dict(self) -> dict[str, Any]:
        return {
            "platformName": self.platform_name,
            "portabilityScore": round(self.score, 2),
            "grade": self.grade,
            "migrationTarget": self.migration_target.value,
            "estimatedMigrationDays": self.estimated_migration_days,
            "matrix": self.matrix.to_dict(),
            "blockers": self.blockers,
            "recommendations": self.recommendations,
        }

    @classmethod
    def assess(cls, platform_name: str, target: CloudProvider) -> "PortabilityScore":
        """Factory: assess the standard ML platform portability."""
        matrix = PortabilityMatrix.ml_platform_matrix()
        return cls(
            platform_name=platform_name,
            matrix=matrix,
            migration_target=target,
            estimated_migration_days=14,
            blockers=["IAM policies must be rewritten per provider"],
            recommendations=[
                "Use KServe instead of managed endpoints for serving portability",
                "Use DVC remote abstraction — only URL changes per provider",
                "Keep Argo Workflows for orchestration (K8s-native, not cloud-specific)",
            ],
        )
