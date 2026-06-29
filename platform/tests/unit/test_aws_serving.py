"""Unit tests for infra.aws.serving (Day 84)."""

import pytest

from infra.aws.serving import (
    EKSResourceSpec,
    EKSInferenceConfig,
    BedrockGuardrailConfig,
    BedrockConfig,
)

IMAGE = "123.dkr.ecr.us-east-1.amazonaws.com/credit-risk:v3"
MODEL_S3 = "s3://bucket/models/v3/model.tar.gz"


# ── EKSResourceSpec ───────────────────────────────────────────────────────────

class TestEKSResourceSpec:
    def test_negative_gpu_raises(self):
        with pytest.raises(ValueError, match="gpu_count"):
            EKSResourceSpec(gpu_count=-1)

    def test_cpu_only_no_gpu_in_dict(self):
        spec = EKSResourceSpec()
        d = spec.to_dict()
        assert "nvidia.com/gpu" not in d["limits"]

    def test_gpu_spec_in_dict(self):
        spec = EKSResourceSpec(gpu_count=2)
        d = spec.to_dict()
        assert d["limits"]["nvidia.com/gpu"] == "2"
        assert d["requests"]["nvidia.com/gpu"] == "2"

    def test_defaults(self):
        spec = EKSResourceSpec()
        d = spec.to_dict()
        assert d["requests"]["cpu"] == "500m"
        assert d["limits"]["memory"] == "4Gi"


# ── EKSInferenceConfig ────────────────────────────────────────────────────────

class TestEKSInferenceConfig:
    def _make(self, **kwargs):
        defaults = dict(
            deployment_name="credit-risk",
            namespace="ml-serving",
            image_uri=IMAGE,
            model_s3_uri=MODEL_S3,
            replicas=2,
            resources=EKSResourceSpec(),
        )
        defaults.update(kwargs)
        return EKSInferenceConfig(**defaults)

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="deployment_name"):
            self._make(deployment_name="")

    def test_empty_namespace_raises(self):
        with pytest.raises(ValueError, match="namespace"):
            self._make(namespace="")

    def test_empty_image_raises(self):
        with pytest.raises(ValueError, match="image_uri"):
            self._make(image_uri="")

    def test_empty_model_s3_raises(self):
        with pytest.raises(ValueError, match="model_s3_uri"):
            self._make(model_s3_uri="")

    def test_zero_replicas_raises(self):
        with pytest.raises(ValueError, match="replicas"):
            self._make(replicas=0)

    def test_to_dict_is_deployment(self):
        cfg = self._make()
        d = cfg.to_dict()
        assert d["kind"] == "Deployment"
        assert d["apiVersion"] == "apps/v1"

    def test_replicas_in_dict(self):
        cfg = self._make(replicas=3)
        d = cfg.to_dict()
        assert d["spec"]["replicas"] == 3

    def test_liveness_probe_in_dict(self):
        cfg = self._make()
        d = cfg.to_dict()
        containers = d["spec"]["template"]["spec"]["containers"]
        assert "livenessProbe" in containers[0]

    def test_model_s3_env_var(self):
        cfg = self._make()
        d = cfg.to_dict()
        env = d["spec"]["template"]["spec"]["containers"][0]["env"]
        s3_envs = [e for e in env if e.get("name") == "MODEL_S3_URI"]
        assert len(s3_envs) == 1
        assert s3_envs[0]["value"] == MODEL_S3

    def test_service_dict(self):
        cfg = self._make()
        svc = cfg.to_service_dict()
        assert svc["kind"] == "Service"
        assert svc["spec"]["selector"]["app"] == "credit-risk"

    def test_node_selector_in_spec(self):
        cfg = self._make(node_selector={"accelerator": "gpu"})
        d = cfg.to_dict()
        assert d["spec"]["template"]["spec"]["nodeSelector"] == {"accelerator": "gpu"}

    def test_cpu_inference_factory(self):
        cfg = EKSInferenceConfig.cpu_inference("cr", IMAGE, MODEL_S3)
        assert cfg.resources.gpu_count == 0
        assert cfg.namespace == "ml-serving"

    def test_gpu_inference_factory(self):
        cfg = EKSInferenceConfig.gpu_inference("cr-gpu", IMAGE, MODEL_S3, gpu_count=1)
        d = cfg.to_dict()
        assert cfg.resources.gpu_count == 1
        tolerations = d["spec"]["template"]["spec"].get("tolerations", [])
        assert any(t.get("key") == "nvidia.com/gpu" for t in tolerations)


# ── BedrockGuardrailConfig ────────────────────────────────────────────────────

class TestBedrockGuardrailConfig:
    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="guardrail_name"):
            BedrockGuardrailConfig("")

    def test_invalid_pii_action_raises(self):
        with pytest.raises(ValueError, match="pii_action"):
            BedrockGuardrailConfig("g1", pii_action="REDACT")

    def test_to_dict(self):
        g = BedrockGuardrailConfig("my-guardrail", topics_to_deny=["financial_advice"])
        d = g.to_dict()
        assert d["name"] == "my-guardrail"
        assert d["topicPolicyConfig"]["topicsConfig"][0]["type"] == "DENY"


# ── BedrockConfig ─────────────────────────────────────────────────────────────

class TestBedrockConfig:
    def test_empty_model_id_raises(self):
        with pytest.raises(ValueError, match="model_id"):
            BedrockConfig("")

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            BedrockConfig("anthropic.claude", temperature=1.5)

    def test_invalid_top_p_raises(self):
        with pytest.raises(ValueError, match="top_p"):
            BedrockConfig("anthropic.claude", top_p=-0.1)

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValueError, match="max_tokens"):
            BedrockConfig("anthropic.claude", max_tokens=0)

    def test_to_dict_structure(self):
        cfg = BedrockConfig("anthropic.claude-3-sonnet-20240229-v1:0")
        d = cfg.to_dict()
        assert d["modelId"] == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert "inferenceConfig" in d

    def test_invoke_body_contains_prompt(self):
        cfg = BedrockConfig("model-id", system_prompt="You are helpful.")
        body = cfg.invoke_body("Hello")
        assert body["messages"][0]["content"] == "Hello"
        assert body["system"] == "You are helpful."

    def test_invoke_body_no_system_when_empty(self):
        cfg = BedrockConfig("model-id")
        body = cfg.invoke_body("Hi")
        assert "system" not in body

    def test_guardrail_in_dict(self):
        g = BedrockGuardrailConfig("g1")
        cfg = BedrockConfig("model-id", guardrail=g)
        d = cfg.to_dict()
        assert "guardrailConfig" in d

    def test_claude_sonnet_factory(self):
        cfg = BedrockConfig.claude_sonnet("Be concise.")
        assert "claude-3-sonnet" in cfg.model_id
        assert cfg.system_prompt == "Be concise."

    def test_titan_embedding_factory(self):
        cfg = BedrockConfig.titan_embedding()
        assert "titan-embed" in cfg.model_id
        assert cfg.temperature == 0.0
