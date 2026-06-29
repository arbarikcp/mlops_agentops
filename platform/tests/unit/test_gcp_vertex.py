"""Unit tests for infra.gcp_vertex (Day 87)."""

import pytest

from infra.gcp_vertex import (
    VertexMachineSpec,
    VertexTrainingJob,
    VertexModelPackage,
    VertexEndpoint,
    VertexPipelineComponent,
    VertexPipeline,
)


# ── VertexMachineSpec ─────────────────────────────────────────────────────────

class TestVertexMachineSpec:
    def test_empty_machine_type_raises(self):
        with pytest.raises(ValueError, match="machine_type"):
            VertexMachineSpec("")

    def test_cpu_only_no_accelerator(self):
        spec = VertexMachineSpec("n1-standard-4")
        d = spec.to_dict()
        assert "acceleratorConfig" not in d

    def test_gpu_accelerator_in_dict(self):
        spec = VertexMachineSpec("a2-highgpu-1g", "NVIDIA_TESLA_A100", 1)
        d = spec.to_dict()
        assert d["acceleratorConfig"]["type"] == "NVIDIA_TESLA_A100"
        assert d["acceleratorConfig"]["count"] == 1

    def test_machine_type_in_dict(self):
        spec = VertexMachineSpec("n1-standard-8")
        assert spec.to_dict()["machineType"] == "n1-standard-8"


# ── VertexTrainingJob ─────────────────────────────────────────────────────────

class TestVertexTrainingJob:
    def _make(self, **kwargs):
        defaults = dict(
            job_name="credit-risk-train",
            project="my-gcp-project",
            location="us-central1",
            image_uri="us-central1-docker.pkg.dev/proj/repo/image:v1",
            gcs_output_uri="gs://bucket/models/",
            machine_spec=VertexMachineSpec("n1-standard-4"),
        )
        defaults.update(kwargs)
        return VertexTrainingJob(**defaults)

    def test_empty_job_name_raises(self):
        with pytest.raises(ValueError, match="job_name"):
            self._make(job_name="")

    def test_empty_project_raises(self):
        with pytest.raises(ValueError, match="project"):
            self._make(project="")

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="image_uri"):
            self._make(image_uri="")

    def test_empty_gcs_output_raises(self):
        with pytest.raises(ValueError, match="gcs_output_uri"):
            self._make(gcs_output_uri="")

    def test_zero_replica_count_raises(self):
        with pytest.raises(ValueError, match="replica_count"):
            self._make(replica_count=0)

    def test_to_dict_structure(self):
        job = self._make()
        d = job.to_dict()
        assert d["displayName"] == "credit-risk-train"
        assert "workerPoolSpecs" in d["jobSpec"]

    def test_spot_scheduling_in_dict(self):
        job = self._make(use_spot=True)
        d = job.to_dict()
        assert "scheduling" in d["jobSpec"]

    def test_no_spot_no_scheduling(self):
        job = self._make(use_spot=False)
        d = job.to_dict()
        assert "scheduling" not in d["jobSpec"]

    def test_experiment_config_in_dict(self):
        job = self._make(experiment_name="credit-risk-exp")
        d = job.to_dict()
        assert "experimentConfig" in d

    def test_aws_equivalent_attribute(self):
        job = self._make()
        assert job.aws_equivalent == "SMTrainingJob"

    def test_args_in_container_spec(self):
        job = self._make(args=["--n_estimators", "200"])
        d = job.to_dict()
        container = d["jobSpec"]["workerPoolSpecs"][0]["containerSpec"]
        assert "--n_estimators" in container["args"]


# ── VertexModelPackage ────────────────────────────────────────────────────────

class TestVertexModelPackage:
    def _make(self, **kwargs):
        defaults = dict(
            display_name="credit-risk-v3",
            project="my-gcp-project",
            location="us-central1",
            artifact_uri="gs://bucket/models/v3/",
            serving_image_uri="us-central1-docker.pkg.dev/proj/repo/serve:v1",
        )
        defaults.update(kwargs)
        return VertexModelPackage(**defaults)

    def test_empty_display_name_raises(self):
        with pytest.raises(ValueError, match="display_name"):
            self._make(display_name="")

    def test_empty_project_raises(self):
        with pytest.raises(ValueError, match="project"):
            self._make(project="")

    def test_empty_artifact_uri_raises(self):
        with pytest.raises(ValueError, match="artifact_uri"):
            self._make(artifact_uri="")

    def test_empty_serving_image_raises(self):
        with pytest.raises(ValueError, match="serving_image_uri"):
            self._make(serving_image_uri="")

    def test_to_dict_structure(self):
        pkg = self._make()
        d = pkg.to_dict()
        assert d["displayName"] == "credit-risk-v3"
        assert "containerSpec" in d
        assert d["artifactUri"] == "gs://bucket/models/v3/"

    def test_aws_equivalent(self):
        pkg = self._make()
        assert pkg.aws_equivalent == "SMModelPackage"

    def test_version_aliases_in_dict(self):
        pkg = self._make(version_aliases=["champion", "v3"])
        d = pkg.to_dict()
        assert "champion" in d["versionAliases"]


# ── VertexEndpoint ────────────────────────────────────────────────────────────

class TestVertexEndpoint:
    def _make(self, **kwargs):
        defaults = dict(
            endpoint_name="credit-risk-endpoint",
            project="my-gcp-project",
            location="us-central1",
            model_display_name="credit-risk-v3",
        )
        defaults.update(kwargs)
        return VertexEndpoint(**defaults)

    def test_empty_endpoint_name_raises(self):
        with pytest.raises(ValueError, match="endpoint_name"):
            self._make(endpoint_name="")

    def test_empty_project_raises(self):
        with pytest.raises(ValueError, match="project"):
            self._make(project="")

    def test_empty_model_display_name_raises(self):
        with pytest.raises(ValueError, match="model_display_name"):
            self._make(model_display_name="")

    def test_min_replica_zero_raises(self):
        with pytest.raises(ValueError, match="min_replica_count"):
            self._make(min_replica_count=0)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_replica_count"):
            self._make(min_replica_count=3, max_replica_count=1)

    def test_traffic_split_not_100_raises(self):
        with pytest.raises(ValueError, match="traffic_split"):
            self._make(traffic_split={"0": 50, "1": 30})

    def test_to_dict_structure(self):
        ep = self._make()
        d = ep.to_dict()
        assert d["displayName"] == "credit-risk-endpoint"
        assert "deployedModel" in d
        assert sum(d["trafficSplit"].values()) == 100

    def test_canary_factory_traffic_split(self):
        ep = VertexEndpoint.canary("ep", "proj", "us-central1", "model-v3", canary_pct=10)
        d = ep.to_dict()
        assert d["trafficSplit"]["1"] == 10
        assert d["trafficSplit"]["0"] == 90

    def test_aws_equivalent(self):
        ep = self._make()
        assert ep.aws_equivalent == "SMEndpoint"


# ── VertexPipeline ────────────────────────────────────────────────────────────

class TestVertexPipeline:
    def test_empty_pipeline_name_raises(self):
        with pytest.raises(ValueError, match="pipeline_name"):
            VertexPipeline("", "proj", "us-central1", "gs://bucket/")

    def test_empty_project_raises(self):
        with pytest.raises(ValueError, match="project"):
            VertexPipeline("p", "", "us-central1", "gs://bucket/")

    def test_empty_gcs_root_raises(self):
        with pytest.raises(ValueError, match="gcs_root"):
            VertexPipeline("p", "proj", "us-central1", "")

    def test_add_component(self):
        pipeline = VertexPipeline("p", "proj", "us-central1", "gs://b/")
        comp = VertexPipelineComponent(
            "train", "image:v1", ["python", "train.py"]
        )
        result = pipeline.add_component(comp)
        assert result is pipeline
        assert len(pipeline.components) == 1

    def test_to_dict_structure(self):
        pipeline = VertexPipeline("p", "proj", "us-central1", "gs://b/")
        d = pipeline.to_dict()
        assert d["pipelineName"] == "p"
        assert d["pipelineRoot"] == "gs://b/"

    def test_pipeline_job_spec(self):
        pipeline = VertexPipeline("p", "proj", "us-central1", "gs://b/")
        spec = pipeline.pipeline_job_spec()
        assert "displayName" in spec
        assert "pipelineSpec" in spec

    def test_credit_risk_pipeline_factory(self):
        pipeline = VertexPipeline.credit_risk_pipeline("my-project", "my-bucket")
        d = pipeline.to_dict()
        assert "credit-risk" in d["pipelineName"]
        assert d["runtimeConfig"]["parameters"]["n_estimators"] == 200

    def test_aws_equivalent(self):
        pipeline = VertexPipeline("p", "proj", "us-central1", "gs://b/")
        assert pipeline.aws_equivalent == "SMPipeline"

    def test_component_empty_name_raises(self):
        with pytest.raises(ValueError, match="component_name"):
            VertexPipelineComponent("", "image:v1", ["python", "train.py"])

    def test_component_empty_command_raises(self):
        with pytest.raises(ValueError, match="command"):
            VertexPipelineComponent("comp", "image:v1", [])
