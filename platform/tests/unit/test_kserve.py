"""Tests for infra/kserve.py — InferenceServiceSpec, CanaryConfig."""
from __future__ import annotations

import pytest

from infra.kserve import CanaryConfig, InferenceServiceSpec


# ── InferenceServiceSpec ───────────────────────────────────────────────────────

class TestInferenceServiceSpec:
    def _spec(self, **kw) -> InferenceServiceSpec:
        defaults = dict(name="credit-risk", storage_uri="s3://models/v1/")
        return InferenceServiceSpec(**{**defaults, **kw})

    def test_basic_manifest(self) -> None:
        m = self._spec().to_manifest()
        assert m["kind"] == "InferenceService"
        assert m["apiVersion"] == "serving.kserve.io/v1beta1"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            InferenceServiceSpec(name="")

    def test_invalid_model_format_raises(self) -> None:
        with pytest.raises(ValueError, match="model_format"):
            self._spec(model_format="random-forest-custom")

    def test_invalid_scale_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="scale_metric"):
            self._spec(scale_metric="gpu_util")

    def test_negative_min_replicas_raises(self) -> None:
        with pytest.raises(ValueError, match="min_replicas"):
            self._spec(min_replicas=-1)

    def test_min_gt_max_replicas_raises(self) -> None:
        with pytest.raises(ValueError, match="min_replicas"):
            self._spec(min_replicas=5, max_replicas=3)

    def test_scale_to_zero_min_replicas(self) -> None:
        m = self._spec(min_replicas=0).to_manifest()
        assert m["spec"]["predictor"]["minReplicas"] == 0

    def test_predictor_model_format(self) -> None:
        m = self._spec(model_format="pytorch").to_manifest()
        assert m["spec"]["predictor"]["model"]["modelFormat"]["name"] == "pytorch"

    def test_storage_uri_in_manifest(self) -> None:
        m = self._spec(storage_uri="s3://ml-models/v2/").to_manifest()
        assert m["spec"]["predictor"]["model"]["storageUri"] == "s3://ml-models/v2/"

    def test_no_transformer_by_default(self) -> None:
        m = self._spec().to_manifest()
        assert "transformer" not in m["spec"]

    def test_transformer_included(self) -> None:
        m = self._spec(
            has_transformer=True,
            transformer_image="credit-risk-transformer:v1",
        ).to_manifest()
        assert "transformer" in m["spec"]
        assert m["spec"]["transformer"]["containers"][0]["image"] == "credit-risk-transformer:v1"

    def test_transformer_without_image_not_included(self) -> None:
        m = self._spec(has_transformer=True, transformer_image="").to_manifest()
        assert "transformer" not in m["spec"]

    def test_resources_in_predictor(self) -> None:
        m = self._spec(cpu_request="1", memory_limit="4Gi").to_manifest()
        resources = m["spec"]["predictor"]["model"]["resources"]
        assert resources["requests"]["cpu"] == "1"
        assert resources["limits"]["memory"] == "4Gi"

    def test_valid_model_formats(self) -> None:
        for fmt in ["sklearn", "pytorch", "onnx", "tensorflow", "lightgbm", "xgboost"]:
            spec = self._spec(model_format=fmt)
            assert spec.model_format == fmt


# ── CanaryConfig ────────────────────────────────────────────────────────────────

class TestCanaryConfig:
    def _config(self, **kw) -> CanaryConfig:
        defaults = dict(
            name="credit-risk",
            stable_storage_uri="s3://models/v1/",
            canary_storage_uri="s3://models/v2/",
            canary_traffic_pct=10,
        )
        return CanaryConfig(**{**defaults, **kw})

    def test_basic_manifest(self) -> None:
        m = self._config().to_manifest()
        assert m["kind"] == "InferenceService"

    def test_canary_traffic_pct_in_manifest(self) -> None:
        m = self._config(canary_traffic_pct=20).to_manifest()
        assert m["spec"]["predictor"]["canaryTrafficPercent"] == 20

    def test_invalid_traffic_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="canary_traffic_pct"):
            self._config(canary_traffic_pct=110)

    def test_zero_traffic_shadow_mode(self) -> None:
        m = self._config(canary_traffic_pct=0).to_manifest()
        assert m["spec"]["predictor"]["canaryTrafficPercent"] == 0

    def test_to_patch_dict(self) -> None:
        patch = self._config().to_patch_dict(50)
        assert patch["spec"]["predictor"]["canaryTrafficPercent"] == 50

    def test_to_patch_dict_invalid_pct_raises(self) -> None:
        with pytest.raises(ValueError, match="canary_traffic_pct"):
            self._config().to_patch_dict(-5)

    def test_promote(self) -> None:
        promoted = self._config().promote()
        assert promoted.canary_traffic_pct == 100
        assert promoted.stable_storage_uri == "s3://models/v2/"

    def test_rollback(self) -> None:
        rolled_back = self._config().rollback()
        assert rolled_back.canary_traffic_pct == 0

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name"):
            CanaryConfig(name="")

    def test_namespace_in_manifest(self) -> None:
        m = self._config(namespace="custom-ns").to_manifest()
        assert m["metadata"]["namespace"] == "custom-ns"
