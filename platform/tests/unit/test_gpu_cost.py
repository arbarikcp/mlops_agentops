"""Unit tests for platform/llm/gpu_cost.py (Day 97)."""

import pytest
from llm.gpu_cost import (
    GPUCostModel,
    GPUInstance,
    GPUUtilizationReport,
    MIGConfig,
    MIGProfile,
    SpotConfig,
)


# ── MIGConfig ──────────────────────────────────────────────────────────────

class TestMIGConfig:
    def test_memory_1g_5gb(self):
        cfg = MIGConfig(
            gpu_instance=GPUInstance.A100_40GB,
            profile=MIGProfile.MIG_1g_5gb,
            num_instances=7,
        )
        assert cfg.memory_per_instance_gb() == 5.0

    def test_memory_2g_10gb(self):
        cfg = MIGConfig(
            gpu_instance=GPUInstance.A100_80GB,
            profile=MIGProfile.MIG_2g_10gb,
            num_instances=3,
        )
        assert cfg.memory_per_instance_gb() == 10.0

    def test_memory_7g_40gb(self):
        cfg = MIGConfig(
            gpu_instance=GPUInstance.A100_40GB,
            profile=MIGProfile.MIG_7g_40gb,
            num_instances=1,
        )
        assert cfg.memory_per_instance_gb() == 40.0

    def test_invalid_num_instances(self):
        with pytest.raises(ValueError, match="num_instances"):
            MIGConfig(
                gpu_instance=GPUInstance.A100_40GB,
                profile=MIGProfile.MIG_1g_5gb,
                num_instances=0,
            )

    def test_to_dict(self):
        cfg = MIGConfig(
            gpu_instance=GPUInstance.H100_80GB,
            profile=MIGProfile.MIG_3g_20gb,
            num_instances=2,
        )
        d = cfg.to_dict()
        assert d["gpu_instance"] == "H100_80GB"
        assert d["profile"] == "3g.20gb"
        assert d["memory_per_instance_gb"] == 20.0


# ── SpotConfig ─────────────────────────────────────────────────────────────

class TestSpotConfig:
    def test_spot_price(self):
        cfg = SpotConfig(instance_type="p3.2xlarge", on_demand_price_usd=3.06)
        assert cfg.spot_price() == pytest.approx(3.06 * 0.3, rel=1e-4)

    def test_savings(self):
        cfg = SpotConfig(instance_type="p3.2xlarge", on_demand_price_usd=3.06, spot_discount=0.7)
        assert cfg.savings_usd_per_hour() == pytest.approx(3.06 * 0.7, rel=1e-4)

    def test_invalid_price(self):
        with pytest.raises(ValueError, match="on_demand_price"):
            SpotConfig(instance_type="x", on_demand_price_usd=0.0)

    def test_invalid_discount_zero(self):
        with pytest.raises(ValueError, match="spot_discount"):
            SpotConfig(instance_type="x", on_demand_price_usd=3.0, spot_discount=0.0)

    def test_invalid_discount_one(self):
        with pytest.raises(ValueError, match="spot_discount"):
            SpotConfig(instance_type="x", on_demand_price_usd=3.0, spot_discount=1.0)

    def test_to_dict(self):
        cfg = SpotConfig(instance_type="p3.2xlarge", on_demand_price_usd=3.06)
        d = cfg.to_dict()
        assert "spot_price" in d
        assert "savings_usd_per_hour" in d


# ── GPUCostModel ───────────────────────────────────────────────────────────

class TestGPUCostModel:
    def test_effective_cost(self):
        model = GPUCostModel(gpu_instance=GPUInstance.A100_40GB, cost_per_hour_usd=3.0)
        # 3.0 / 0.8 = 3.75
        assert model.effective_cost_per_hour() == pytest.approx(3.75)

    def test_full_utilization(self):
        model = GPUCostModel(
            gpu_instance=GPUInstance.T4,
            cost_per_hour_usd=1.0,
            utilization_target=1.0,
        )
        assert model.effective_cost_per_hour() == pytest.approx(1.0)

    def test_invalid_cost(self):
        with pytest.raises(ValueError, match="cost_per_hour"):
            GPUCostModel(gpu_instance=GPUInstance.T4, cost_per_hour_usd=0.0)

    def test_invalid_utilization(self):
        with pytest.raises(ValueError, match="utilization_target"):
            GPUCostModel(gpu_instance=GPUInstance.T4, cost_per_hour_usd=1.0, utilization_target=0.0)

    def test_to_dict(self):
        model = GPUCostModel(gpu_instance=GPUInstance.L4, cost_per_hour_usd=2.0)
        d = model.to_dict()
        assert d["gpu_instance"] == "L4"
        assert "effective_cost_per_hour" in d


# ── GPUUtilizationReport ───────────────────────────────────────────────────

class TestGPUUtilizationReport:
    def test_healthy_not_underutilized(self):
        r = GPUUtilizationReport(
            gpu_instance=GPUInstance.A100_40GB,
            sm_efficiency=0.8,
            memory_bandwidth_util=0.7,
            tensor_core_util=0.6,
        )
        assert r.is_underutilized() is False

    def test_low_sm_is_underutilized(self):
        r = GPUUtilizationReport(
            gpu_instance=GPUInstance.A100_40GB,
            sm_efficiency=0.3,
            memory_bandwidth_util=0.7,
            tensor_core_util=0.5,
        )
        assert r.is_underutilized() is True

    def test_low_mem_bw_is_underutilized(self):
        r = GPUUtilizationReport(
            gpu_instance=GPUInstance.V100,
            sm_efficiency=0.6,
            memory_bandwidth_util=0.2,
            tensor_core_util=0.5,
        )
        assert r.is_underutilized() is True

    def test_hints_sm_low(self):
        r = GPUUtilizationReport(
            gpu_instance=GPUInstance.A100_40GB,
            sm_efficiency=0.3,
            memory_bandwidth_util=0.6,
            tensor_core_util=0.5,
        )
        hints = r.optimization_hints()
        assert any("batch" in h.lower() for h in hints)

    def test_hints_tensor_low(self):
        r = GPUUtilizationReport(
            gpu_instance=GPUInstance.H100_80GB,
            sm_efficiency=0.7,
            memory_bandwidth_util=0.6,
            tensor_core_util=0.1,
        )
        hints = r.optimization_hints()
        assert any("bf16" in h.lower() or "tf32" in h.lower() for h in hints)

    def test_hints_healthy(self):
        r = GPUUtilizationReport(
            gpu_instance=GPUInstance.A100_80GB,
            sm_efficiency=0.8,
            memory_bandwidth_util=0.75,
            tensor_core_util=0.5,
        )
        hints = r.optimization_hints()
        assert len(hints) == 1
        assert "healthy" in hints[0].lower()

    def test_invalid_metric_range(self):
        with pytest.raises(ValueError):
            GPUUtilizationReport(
                gpu_instance=GPUInstance.T4,
                sm_efficiency=1.1,
                memory_bandwidth_util=0.5,
                tensor_core_util=0.5,
            )

    def test_to_dict(self):
        r = GPUUtilizationReport(
            gpu_instance=GPUInstance.L4,
            sm_efficiency=0.6,
            memory_bandwidth_util=0.5,
            tensor_core_util=0.4,
        )
        d = r.to_dict()
        assert d["gpu_instance"] == "L4"
        assert "is_underutilized" in d
