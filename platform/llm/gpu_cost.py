"""
gpu_cost.py — GPU Utilization & Cost Optimization (Day 97)

Covers MIG partitioning, spot pricing, cost modeling,
and utilization reporting with optimization hints.
No external SDK imports — pure Python dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class MIGProfile(str, Enum):
    """NVIDIA MIG (Multi-Instance GPU) profiles."""

    MIG_1g_5gb = "1g.5gb"
    MIG_2g_10gb = "2g.10gb"
    MIG_3g_20gb = "3g.20gb"
    MIG_4g_20gb = "4g.20gb"
    MIG_7g_40gb = "7g.40gb"


class GPUInstance(str, Enum):
    """GPU instance types."""

    A100_40GB = "A100_40GB"
    A100_80GB = "A100_80GB"
    H100_80GB = "H100_80GB"
    T4 = "T4"
    L4 = "L4"
    V100 = "V100"


@dataclass
class MIGConfig:
    """Multi-Instance GPU configuration."""

    gpu_instance: GPUInstance
    profile: MIGProfile
    num_instances: int

    def __post_init__(self) -> None:
        if self.num_instances < 1:
            raise ValueError(f"num_instances must be >= 1, got {self.num_instances}")

    def memory_per_instance_gb(self) -> float:
        """Parse VRAM per instance from profile string (e.g. '1g.5gb' → 5.0)."""
        match = re.search(r"(\d+)gb", self.profile.value)
        if not match:
            raise ValueError(
                f"Cannot parse memory from profile: {self.profile.value!r}"
            )
        return float(match.group(1))

    def to_dict(self) -> dict:
        return {
            "gpu_instance": self.gpu_instance.value,
            "profile": self.profile.value,
            "num_instances": self.num_instances,
            "memory_per_instance_gb": self.memory_per_instance_gb(),
        }


@dataclass
class SpotConfig:
    """Spot/preemptible GPU pricing configuration."""

    instance_type: str
    on_demand_price_usd: float
    spot_discount: float = 0.7

    def __post_init__(self) -> None:
        if self.on_demand_price_usd <= 0:
            raise ValueError(
                f"on_demand_price_usd must be > 0, got {self.on_demand_price_usd}"
            )
        if not (0 < self.spot_discount < 1):
            raise ValueError(
                f"spot_discount must be in (0, 1), got {self.spot_discount}"
            )

    def spot_price(self) -> float:
        """Spot price per hour."""
        return self.on_demand_price_usd * (1 - self.spot_discount)

    def savings_usd_per_hour(self) -> float:
        """Hourly savings vs on-demand."""
        return self.on_demand_price_usd - self.spot_price()

    def to_dict(self) -> dict:
        return {
            "instance_type": self.instance_type,
            "on_demand_price_usd": self.on_demand_price_usd,
            "spot_discount": self.spot_discount,
            "spot_price": self.spot_price(),
            "savings_usd_per_hour": self.savings_usd_per_hour(),
        }


@dataclass
class GPUCostModel:
    """GPU cost model including idle-time penalty."""

    gpu_instance: GPUInstance
    cost_per_hour_usd: float
    utilization_target: float = 0.8

    def __post_init__(self) -> None:
        if self.cost_per_hour_usd <= 0:
            raise ValueError(
                f"cost_per_hour_usd must be > 0, got {self.cost_per_hour_usd}"
            )
        if not (0 < self.utilization_target <= 1):
            raise ValueError(
                f"utilization_target must be in (0, 1], "
                f"got {self.utilization_target}"
            )

    def effective_cost_per_hour(self) -> float:
        """Effective cost per GPU-hour including idle overhead."""
        return self.cost_per_hour_usd / self.utilization_target

    def to_dict(self) -> dict:
        return {
            "gpu_instance": self.gpu_instance.value,
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "utilization_target": self.utilization_target,
            "effective_cost_per_hour": self.effective_cost_per_hour(),
        }


@dataclass
class GPUUtilizationReport:
    """GPU utilization metrics report."""

    gpu_instance: GPUInstance
    sm_efficiency: float
    memory_bandwidth_util: float
    tensor_core_util: float

    def __post_init__(self) -> None:
        for name, val in [
            ("sm_efficiency", self.sm_efficiency),
            ("memory_bandwidth_util", self.memory_bandwidth_util),
            ("tensor_core_util", self.tensor_core_util),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")

    def is_underutilized(self) -> bool:
        """True if SM efficiency < 50% or memory bandwidth < 40%."""
        return self.sm_efficiency < 0.5 or self.memory_bandwidth_util < 0.4

    def optimization_hints(self) -> list[str]:
        """Return actionable optimization hints based on utilization metrics."""
        hints: list[str] = []

        if self.sm_efficiency < 0.5:
            hints.append(
                f"SM efficiency is {self.sm_efficiency:.0%} — "
                "increase batch size to improve GPU occupancy"
            )

        if self.memory_bandwidth_util < 0.4:
            hints.append(
                f"Memory bandwidth utilization is {self.memory_bandwidth_util:.0%} — "
                "consider a larger model or switch to FP16 to improve memory throughput"
            )

        if self.tensor_core_util < 0.3:
            hints.append(
                f"Tensor core utilization is {self.tensor_core_util:.0%} — "
                "enable TF32 or BF16 to activate tensor cores"
            )

        if not hints:
            hints.append("GPU utilization looks healthy — no immediate action needed")

        return hints

    def to_dict(self) -> dict:
        return {
            "gpu_instance": self.gpu_instance.value,
            "sm_efficiency": self.sm_efficiency,
            "memory_bandwidth_util": self.memory_bandwidth_util,
            "tensor_core_util": self.tensor_core_util,
            "is_underutilized": self.is_underutilized(),
        }
