"""Unit tests for infra.aws.sagemaker_training (Day 80)."""

import pytest

from infra.aws.sagemaker_training import (
    DataChannel,
    SMTrainingJob,
    ProcessingInput,
    ProcessingOutput,
    SMProcessingJob,
    SMTrialComponent,
    SMExperiment,
)


# ── DataChannel ───────────────────────────────────────────────────────────────

class TestDataChannel:
    def test_valid_channel(self):
        ch = DataChannel("train", "s3://bucket/data/train/")
        d = ch.to_dict()
        assert d["ChannelName"] == "train"
        assert d["ContentType"] == "text/csv"

    def test_empty_channel_name_raises(self):
        with pytest.raises(ValueError, match="channel_name"):
            DataChannel("", "s3://bucket/")

    def test_empty_s3_uri_raises(self):
        with pytest.raises(ValueError, match="s3_uri"):
            DataChannel("train", "")

    def test_s3_uri_in_dict(self):
        ch = DataChannel("validation", "s3://bucket/val/")
        d = ch.to_dict()
        assert d["DataSource"]["S3DataSource"]["S3Uri"] == "s3://bucket/val/"


# ── SMTrainingJob ─────────────────────────────────────────────────────────────

class TestSMTrainingJob:
    def _make_job(self, **kwargs):
        defaults = dict(
            job_name="test-job",
            role_arn="arn:aws:iam::123:role/SM",
            image_uri="123.dkr.ecr.us-east-1.amazonaws.com/credit-risk:latest",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            output_s3_uri="s3://bucket/output/",
        )
        defaults.update(kwargs)
        return SMTrainingJob(**defaults)

    def test_empty_job_name_raises(self):
        with pytest.raises(ValueError, match="job_name"):
            self._make_job(job_name="")

    def test_empty_role_arn_raises(self):
        with pytest.raises(ValueError, match="role_arn"):
            self._make_job(role_arn="")

    def test_empty_image_uri_raises(self):
        with pytest.raises(ValueError, match="image_uri"):
            self._make_job(image_uri="")

    def test_empty_output_s3_raises(self):
        with pytest.raises(ValueError, match="output_s3_uri"):
            self._make_job(output_s3_uri="")

    def test_instance_count_zero_raises(self):
        with pytest.raises(ValueError, match="instance_count"):
            self._make_job(instance_count=0)

    def test_spot_config_in_dict(self):
        job = self._make_job(use_spot_instances=True)
        d = job.to_dict()
        assert d["EnableManagedSpotTraining"] is True
        assert "MaxWaitTimeInSeconds" in d["StoppingCondition"]

    def test_no_spot_no_wait(self):
        job = self._make_job(use_spot_instances=False)
        d = job.to_dict()
        assert d["EnableManagedSpotTraining"] is False
        assert "MaxWaitTimeInSeconds" not in d["StoppingCondition"]

    def test_hyperparameters_in_dict(self):
        job = self._make_job(hyperparameters={"n_estimators": "200"})
        d = job.to_dict()
        assert d["HyperParameters"]["n_estimators"] == "200"

    def test_data_channels_in_dict(self):
        job = self._make_job(data_channels=[DataChannel("train", "s3://b/train/")])
        d = job.to_dict()
        assert len(d["InputDataConfig"]) == 1

    def test_experiment_config_in_dict(self):
        job = self._make_job(experiment_name="my-exp", trial_name="trial-1")
        d = job.to_dict()
        assert d["ExperimentConfig"]["ExperimentName"] == "my-exp"

    def test_no_experiment_config_when_empty(self):
        job = self._make_job()
        d = job.to_dict()
        assert "ExperimentConfig" not in d

    def test_credit_risk_factory(self):
        job = SMTrainingJob.credit_risk_training(
            "cr-job", "arn:aws:iam::123:role/SM",
            "123.dkr.ecr.us-east-1.amazonaws.com/cr:v1",
            "s3://b/train/", "s3://b/val/", "s3://b/output/",
        )
        d = job.to_dict()
        assert d["HyperParameters"]["objective"] == "binary:logistic"
        assert len(d["InputDataConfig"]) == 2
        assert d["Tags"][0]["Value"] == "credit-risk"


# ── SMProcessingJob ───────────────────────────────────────────────────────────

class TestSMProcessingJob:
    def _make_job(self, **kwargs):
        defaults = dict(
            job_name="proc-job",
            role_arn="arn:aws:iam::123:role/SM",
            image_uri="123.dkr.ecr.us-east-1.amazonaws.com/proc:v1",
            instance_type="ml.m5.xlarge",
            script_path="/opt/ml/code/preprocess.py",
        )
        defaults.update(kwargs)
        return SMProcessingJob(**defaults)

    def test_empty_job_name_raises(self):
        with pytest.raises(ValueError, match="job_name"):
            self._make_job(job_name="")

    def test_empty_script_raises(self):
        with pytest.raises(ValueError, match="script_path"):
            self._make_job(script_path="")

    def test_to_dict_structure(self):
        job = self._make_job()
        d = job.to_dict()
        assert d["ProcessingJobName"] == "proc-job"
        assert d["AppSpecification"]["ContainerEntrypoint"] == ["python3", "/opt/ml/code/preprocess.py"]

    def test_inputs_in_dict(self):
        job = self._make_job(inputs=[ProcessingInput("data", "s3://b/data/", "/opt/ml/processing/input")])
        d = job.to_dict()
        assert len(d["ProcessingInputs"]) == 1

    def test_outputs_in_dict(self):
        job = self._make_job(outputs=[ProcessingOutput("result", "s3://b/out/", "/opt/ml/processing/output")])
        d = job.to_dict()
        assert len(d["ProcessingOutputConfig"]["Outputs"]) == 1

    def test_processing_input_empty_name_raises(self):
        with pytest.raises(ValueError, match="input_name"):
            ProcessingInput("", "s3://b/", "/tmp")

    def test_processing_output_empty_s3_raises(self):
        with pytest.raises(ValueError, match="s3_uri"):
            ProcessingOutput("out", "", "/tmp")


# ── SMExperiment ──────────────────────────────────────────────────────────────

class TestSMExperiment:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="experiment_name"):
            SMExperiment("", "desc")

    def test_empty_description_raises(self):
        with pytest.raises(ValueError, match="description"):
            SMExperiment("exp", "")

    def test_add_trial(self):
        exp = SMExperiment("exp", "desc")
        comp = SMTrialComponent("comp1", "Training", "job-1", metrics={"auc": 0.82})
        exp.add_trial("trial-1", [comp])
        d = exp.to_dict()
        assert "trial-1" in d["Trials"]
        assert d["Trials"]["trial-1"][0]["Metrics"]["auc"] == 0.82

    def test_trial_component_empty_name_raises(self):
        with pytest.raises(ValueError, match="component_name"):
            SMTrialComponent("", "Training", "job-1")

    def test_trial_component_empty_job_raises(self):
        with pytest.raises(ValueError, match="job_name"):
            SMTrialComponent("comp", "Training", "")

    def test_credit_risk_experiment_factory(self):
        exp = SMExperiment.credit_risk_experiment()
        assert "credit-risk" in exp.experiment_name
        d = exp.to_dict()
        assert d["ExperimentName"] == "credit-risk-model-selection"
