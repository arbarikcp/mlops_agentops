"""Unit tests for infra.portability (Day 88)."""

import pytest

from infra.portability import (
    MatrixEntry,
    PortabilityMatrix,
    CloudAdapter,
    PortabilityScore,
    PortabilityLevel,
    CloudProvider,
)


# ── MatrixEntry ───────────────────────────────────────────────────────────────

class TestMatrixEntry:
    def _make(self, **kwargs):
        defaults = dict(
            component="MLflow tracking",
            category="experiment_tracking",
            portability_level=PortabilityLevel.FULLY_PORTABLE,
            aws_impl="MLflow on EKS",
            gcp_impl="MLflow on GKE",
            azure_impl="MLflow on AKS",
        )
        defaults.update(kwargs)
        return MatrixEntry(**defaults)

    def test_empty_component_raises(self):
        with pytest.raises(ValueError, match="component"):
            self._make(component="")

    def test_empty_category_raises(self):
        with pytest.raises(ValueError, match="category"):
            self._make(category="")

    def test_is_portable_fully(self):
        e = self._make(portability_level=PortabilityLevel.FULLY_PORTABLE)
        assert e.is_portable is True

    def test_is_portable_adapter(self):
        e = self._make(portability_level=PortabilityLevel.ADAPTER_NEEDED)
        assert e.is_portable is True

    def test_not_portable_cloud_specific(self):
        e = self._make(portability_level=PortabilityLevel.CLOUD_SPECIFIC)
        assert e.is_portable is False

    def test_not_portable_rewrite(self):
        e = self._make(portability_level=PortabilityLevel.REWRITE_NEEDED)
        assert e.is_portable is False

    def test_to_dict_structure(self):
        e = self._make()
        d = e.to_dict()
        assert d["component"] == "MLflow tracking"
        assert d["portabilityLevel"] == "fully_portable"
        assert "aws" in d["implementations"]


# ── PortabilityMatrix ─────────────────────────────────────────────────────────

class TestPortabilityMatrix:
    def test_empty_matrix_score_zero(self):
        m = PortabilityMatrix()
        assert m.portability_score() == 0.0

    def test_all_portable_score_one(self):
        m = PortabilityMatrix()
        for _ in range(5):
            m.add_entry(MatrixEntry(
                f"comp-{_}", "cat", PortabilityLevel.FULLY_PORTABLE,
                "aws", "gcp", "azure"
            ))
        assert m.portability_score() == 1.0

    def test_mixed_score(self):
        m = PortabilityMatrix()
        m.add_entry(MatrixEntry("a", "cat", PortabilityLevel.FULLY_PORTABLE, "a", "b", "c"))
        m.add_entry(MatrixEntry("b", "cat", PortabilityLevel.CLOUD_SPECIFIC, "a", "b", "c"))
        assert m.portability_score() == 0.5

    def test_portable_components_list(self):
        m = PortabilityMatrix()
        m.add_entry(MatrixEntry("a", "cat", PortabilityLevel.FULLY_PORTABLE, "a", "b", "c"))
        m.add_entry(MatrixEntry("b", "cat", PortabilityLevel.CLOUD_SPECIFIC, "a", "b", "c"))
        assert len(m.portable_components()) == 1
        assert len(m.cloud_specific_components()) == 1

    def test_by_category(self):
        m = PortabilityMatrix()
        m.add_entry(MatrixEntry("a", "serving", PortabilityLevel.FULLY_PORTABLE, "a", "b", "c"))
        m.add_entry(MatrixEntry("b", "training", PortabilityLevel.CLOUD_SPECIFIC, "a", "b", "c"))
        cats = m.by_category()
        assert "serving" in cats
        assert "training" in cats

    def test_to_dict_summary(self):
        m = PortabilityMatrix()
        m.add_entry(MatrixEntry("a", "cat", PortabilityLevel.FULLY_PORTABLE, "a", "b", "c"))
        d = m.to_dict()
        assert d["summary"]["total"] == 1
        assert d["summary"]["portable"] == 1

    def test_ml_platform_matrix_factory(self):
        m = PortabilityMatrix.ml_platform_matrix()
        assert len(m.entries) >= 8
        # Score should be > 0.5 (more portable than cloud-specific)
        assert m.portability_score() >= 0.5

    def test_add_entry_chaining(self):
        m = PortabilityMatrix()
        result = m.add_entry(MatrixEntry("a", "cat", PortabilityLevel.FULLY_PORTABLE, "a", "b", "c"))
        assert result is m


# ── CloudAdapter ──────────────────────────────────────────────────────────────

class TestCloudAdapter:
    def test_empty_region_raises(self):
        with pytest.raises(ValueError, match="region"):
            CloudAdapter(CloudProvider.AWS, "", "iam_role")

    def test_empty_credentials_source_raises(self):
        with pytest.raises(ValueError, match="credentials_source"):
            CloudAdapter(CloudProvider.AWS, "us-east-1", "")

    def test_aws_storage_uri(self):
        adapter = CloudAdapter.aws()
        uri = adapter.storage_uri("my-bucket", "models/v1/model.tar.gz")
        assert uri.startswith("s3://")

    def test_gcp_storage_uri(self):
        adapter = CloudAdapter.gcp("my-project")
        uri = adapter.storage_uri("my-bucket", "models/v1/")
        assert uri.startswith("gs://")

    def test_azure_storage_uri(self):
        adapter = CloudAdapter(CloudProvider.AZURE, "eastus", "my-account")
        uri = adapter.storage_uri("container", "path/")
        assert "abfs://" in uri

    def test_local_storage_uri(self):
        adapter = CloudAdapter.local()
        uri = adapter.storage_uri("data", "path")
        assert uri.startswith("file://")

    def test_aws_registry_uri(self):
        adapter = CloudAdapter(CloudProvider.AWS, "us-east-1", "123456789012")
        uri = adapter.registry_uri("credit-risk", "v3")
        assert "amazonaws.com" in uri
        assert "v3" in uri

    def test_to_dict_structure(self):
        adapter = CloudAdapter.aws()
        d = adapter.to_dict()
        assert d["provider"] == "aws"
        assert d["storageScheme"] == "s3"

    def test_aws_factory(self):
        adapter = CloudAdapter.aws(region="us-west-2")
        assert adapter.provider == CloudProvider.AWS
        assert adapter.region == "us-west-2"

    def test_local_factory(self):
        adapter = CloudAdapter.local()
        assert adapter.provider == CloudProvider.LOCAL


# ── PortabilityScore ──────────────────────────────────────────────────────────

class TestPortabilityScore:
    def test_empty_platform_name_raises(self):
        with pytest.raises(ValueError, match="platform_name"):
            PortabilityScore("", PortabilityMatrix(), CloudProvider.GCP, 10)

    def test_negative_migration_days_raises(self):
        with pytest.raises(ValueError, match="estimated_migration_days"):
            PortabilityScore("platform", PortabilityMatrix(), CloudProvider.GCP, -1)

    def test_grade_a_high_score(self):
        m = PortabilityMatrix()
        for i in range(10):
            m.add_entry(MatrixEntry(f"c{i}", "cat", PortabilityLevel.FULLY_PORTABLE, "a", "b", "c"))
        ps = PortabilityScore("p", m, CloudProvider.GCP, 5)
        assert ps.grade == "A"

    def test_grade_d_low_score(self):
        m = PortabilityMatrix()
        for i in range(10):
            m.add_entry(MatrixEntry(f"c{i}", "cat", PortabilityLevel.CLOUD_SPECIFIC, "a", "b", "c"))
        ps = PortabilityScore("p", m, CloudProvider.GCP, 30)
        assert ps.grade == "D"

    def test_to_dict_structure(self):
        m = PortabilityMatrix.ml_platform_matrix()
        ps = PortabilityScore("ml-platform", m, CloudProvider.GCP, 14)
        d = ps.to_dict()
        assert "portabilityScore" in d
        assert "grade" in d
        assert "migrationTarget" in d

    def test_assess_factory(self):
        ps = PortabilityScore.assess("ml-platform", CloudProvider.GCP)
        d = ps.to_dict()
        assert d["migrationTarget"] == "gcp"
        assert len(ps.blockers) >= 1
        assert len(ps.recommendations) >= 1

    def test_score_reflects_matrix(self):
        m = PortabilityMatrix.ml_platform_matrix()
        ps = PortabilityScore("p", m, CloudProvider.GCP, 14)
        assert abs(ps.score - m.portability_score()) < 0.001
