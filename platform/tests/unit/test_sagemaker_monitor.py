"""Unit tests for infra.aws.sagemaker_monitor (Day 83)."""

import pytest

from infra.aws.sagemaker_monitor import (
    MonitoringConstraints,
    SMDataQualityMonitor,
    SMModelQualityMonitor,
    SMClarifyBiasConfig,
    SMClarifyConfig,
    MonitorSchedule,
)

ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"


# ── MonitoringConstraints ─────────────────────────────────────────────────────

class TestMonitoringConstraints:
    def test_empty_uri_raises(self):
        with pytest.raises(ValueError, match="s3_uri"):
            MonitoringConstraints("")

    def test_to_dict(self):
        mc = MonitoringConstraints("s3://bucket/baselines/")
        d = mc.to_dict()
        assert "S3Uri" in d
        assert "constraints.json" in d["S3Uri"]


# ── SMDataQualityMonitor ──────────────────────────────────────────────────────

class TestSMDataQualityMonitor:
    def _make(self, **kwargs):
        defaults = dict(
            monitor_name="dq-monitor",
            endpoint_name="credit-risk-prod",
            role_arn=ROLE,
            output_s3_uri="s3://bucket/monitors/",
            baseline_constraints_s3="s3://bucket/baselines/constraints.json",
            baseline_statistics_s3="s3://bucket/baselines/statistics.json",
        )
        defaults.update(kwargs)
        return SMDataQualityMonitor(**defaults)

    def test_empty_monitor_name_raises(self):
        with pytest.raises(ValueError, match="monitor_name"):
            self._make(monitor_name="")

    def test_empty_endpoint_raises(self):
        with pytest.raises(ValueError, match="endpoint_name"):
            self._make(endpoint_name="")

    def test_empty_role_raises(self):
        with pytest.raises(ValueError, match="role_arn"):
            self._make(role_arn="")

    def test_empty_output_s3_raises(self):
        with pytest.raises(ValueError, match="output_s3_uri"):
            self._make(output_s3_uri="")

    def test_empty_constraints_raises(self):
        with pytest.raises(ValueError, match="baseline_constraints_s3"):
            self._make(baseline_constraints_s3="")

    def test_empty_statistics_raises(self):
        with pytest.raises(ValueError, match="baseline_statistics_s3"):
            self._make(baseline_statistics_s3="")

    def test_to_dict_structure(self):
        monitor = self._make()
        d = monitor.to_dict()
        assert d["MonitoringType"] == "DataQuality"
        assert "MonitoringScheduleConfig" in d
        assert "BaselineConfig" in d["MonitoringScheduleConfig"]["MonitoringJobDefinition"]

    def test_schedule_in_dict(self):
        monitor = self._make(schedule=MonitorSchedule.HOURLY)
        d = monitor.to_dict()
        expr = d["MonitoringScheduleConfig"]["ScheduleConfig"]["ScheduleExpression"]
        assert "0 * ?" in expr

    def test_for_endpoint_factory(self):
        monitor = SMDataQualityMonitor.for_endpoint("credit-risk-prod", ROLE, "my-bucket")
        assert "credit-risk-prod" in monitor.monitor_name
        assert "my-bucket" in monitor.output_s3_uri


# ── SMModelQualityMonitor ──────────────────────────────────────────────────────

class TestSMModelQualityMonitor:
    def _make(self, **kwargs):
        defaults = dict(
            monitor_name="mq-monitor",
            endpoint_name="credit-risk-prod",
            role_arn=ROLE,
            output_s3_uri="s3://bucket/monitors/mq/",
            ground_truth_s3_uri="s3://bucket/ground-truth/",
            problem_type="BinaryClassification",
        )
        defaults.update(kwargs)
        return SMModelQualityMonitor(**defaults)

    def test_empty_monitor_name_raises(self):
        with pytest.raises(ValueError, match="monitor_name"):
            self._make(monitor_name="")

    def test_empty_endpoint_raises(self):
        with pytest.raises(ValueError, match="endpoint_name"):
            self._make(endpoint_name="")

    def test_empty_ground_truth_raises(self):
        with pytest.raises(ValueError, match="ground_truth_s3_uri"):
            self._make(ground_truth_s3_uri="")

    def test_invalid_problem_type_raises(self):
        with pytest.raises(ValueError, match="problem_type"):
            self._make(problem_type="TimeSeries")

    def test_invalid_probability_threshold_raises(self):
        with pytest.raises(ValueError, match="probability_threshold"):
            self._make(probability_threshold=1.5)

    def test_to_dict_structure(self):
        monitor = self._make()
        d = monitor.to_dict()
        assert d["MonitoringType"] == "ModelQuality"
        job_def = d["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        assert job_def["ModelQualityAppSpecification"]["ProblemType"] == "BinaryClassification"

    def test_ground_truth_in_dict(self):
        monitor = self._make()
        d = monitor.to_dict()
        job_def = d["MonitoringScheduleConfig"]["MonitoringJobDefinition"]
        assert "GroundTruthS3Input" in job_def


# ── SMClarifyConfig ───────────────────────────────────────────────────────────

class TestSMClarifyConfig:
    def _make(self, **kwargs):
        defaults = dict(
            job_name="clarify-job",
            role_arn=ROLE,
            model_name="credit-risk-model",
            instance_type="ml.m5.xlarge",
            instance_count=1,
            input_s3_uri="s3://bucket/data/",
            output_s3_uri="s3://bucket/clarify-output/",
            headers=["age", "income", "label"],
            label_column="label",
        )
        defaults.update(kwargs)
        return SMClarifyConfig(**defaults)

    def test_empty_job_name_raises(self):
        with pytest.raises(ValueError, match="job_name"):
            self._make(job_name="")

    def test_empty_model_name_raises(self):
        with pytest.raises(ValueError, match="model_name"):
            self._make(model_name="")

    def test_empty_input_s3_raises(self):
        with pytest.raises(ValueError, match="input_s3_uri"):
            self._make(input_s3_uri="")

    def test_empty_headers_raises(self):
        with pytest.raises(ValueError, match="headers"):
            self._make(headers=[])

    def test_empty_label_raises(self):
        with pytest.raises(ValueError, match="label_column"):
            self._make(label_column="")

    def test_to_dict_structure(self):
        cfg = self._make()
        d = cfg.to_dict()
        assert "ProcessingJobName" in d
        assert "ClarifyCheckConfig" in d

    def test_shap_in_dict_when_enabled(self):
        cfg = self._make(enable_shap=True, shap_num_samples=50)
        d = cfg.to_dict()
        assert "ShapConfig" in d["ClarifyCheckConfig"]
        assert d["ClarifyCheckConfig"]["ShapConfig"]["NumberOfSamples"] == 50

    def test_no_shap_when_disabled(self):
        cfg = self._make(enable_shap=False)
        d = cfg.to_dict()
        assert "ShapConfig" not in d["ClarifyCheckConfig"]

    def test_bias_config_in_dict(self):
        bias = SMClarifyBiasConfig("label", "age", [30], [1])
        cfg = self._make(bias_config=bias)
        d = cfg.to_dict()
        assert "BiasConfig" in d["ClarifyCheckConfig"]

    def test_bias_config_empty_label_raises(self):
        with pytest.raises(ValueError, match="label_name"):
            SMClarifyBiasConfig("", "age", [30], [1])

    def test_bias_config_empty_facet_raises(self):
        with pytest.raises(ValueError, match="facet_name"):
            SMClarifyBiasConfig("label", "", [30], [1])

    def test_credit_risk_factory(self):
        cfg = SMClarifyConfig.credit_risk_clarify(ROLE, "cr-model", "s3://b/data/", "s3://b/out/")
        d = cfg.to_dict()
        assert "BiasConfig" in d["ClarifyCheckConfig"]
        assert d["ClarifyCheckConfig"]["ShapConfig"]["NumberOfSamples"] == 200
