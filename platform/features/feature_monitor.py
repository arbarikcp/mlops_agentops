"""Feature monitoring: freshness, data quality, and distribution drift.

Classes:
  FreshnessStatus        — FRESH / STALE / MISSING
  FreshnessCheck         — result of one freshness evaluation
  FeatureQualityResult   — per-feature null/range quality result
  FeatureDriftResult     — per-feature PSI + KS drift severity
  FeatureMonitorReport   — combined report across all three pillars
  FreshnessChecker       — compares last_materialized_at vs threshold
  FeatureQualityChecker  — checks null rates and value ranges per column
  FeatureDriftMonitor    — computes PSI and KS between reference and current
  FeatureMonitor         — orchestrates all three checkers in one `.run()` call

See: docs/phase6/day43_feature_monitoring.md
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


# ── Freshness ──────────────────────────────────────────────────────────────────

class FreshnessStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    MISSING = "missing"


@dataclass
class FreshnessCheck:
    """Result of a freshness evaluation for one feature view.

    Attributes:
        feature_view_name:   Name of the feature view checked.
        last_materialized_at: When it was last written to the online store (None if never).
        threshold_hours:     Maximum acceptable age (alert above this).
        status:              FRESH / STALE / MISSING.
        age_hours:           How many hours since last materialization (inf if None).
    """

    feature_view_name: str
    last_materialized_at: datetime | None
    threshold_hours: float
    status: FreshnessStatus
    age_hours: float

    @property
    def is_fresh(self) -> bool:
        return self.status == FreshnessStatus.FRESH


class FreshnessChecker:
    """Evaluates freshness of a feature view against a threshold.

    Args:
        stale_multiplier: Age beyond threshold × this factor is MISSING (not STALE).
                          Default: 3× (e.g. threshold=24h → MISSING if age > 72h).
    """

    def __init__(self, stale_multiplier: float = 3.0) -> None:
        if stale_multiplier <= 1.0:
            raise ValueError("stale_multiplier must be > 1.0")
        self.stale_multiplier = stale_multiplier

    def check(
        self,
        feature_view_name: str,
        last_materialized_at: datetime | None,
        threshold_hours: float = 25.0,
        now: datetime | None = None,
    ) -> FreshnessCheck:
        """Evaluate freshness of a feature view.

        Args:
            feature_view_name:   Name used in the result.
            last_materialized_at: UTC datetime of last successful materialization.
            threshold_hours:     Max acceptable staleness in hours.
            now:                 Override for current time (UTC); default: utcnow.

        Returns:
            FreshnessCheck with status and age_hours.
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if last_materialized_at is None:
            return FreshnessCheck(
                feature_view_name=feature_view_name,
                last_materialized_at=None,
                threshold_hours=threshold_hours,
                status=FreshnessStatus.MISSING,
                age_hours=math.inf,
            )

        # Ensure timezone-aware comparison
        last = last_materialized_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        age_hours = (now - last).total_seconds() / 3600

        if age_hours < threshold_hours:
            status = FreshnessStatus.FRESH
        elif age_hours < threshold_hours * self.stale_multiplier:
            status = FreshnessStatus.STALE
        else:
            status = FreshnessStatus.MISSING

        return FreshnessCheck(
            feature_view_name=feature_view_name,
            last_materialized_at=last_materialized_at,
            threshold_hours=threshold_hours,
            status=status,
            age_hours=age_hours,
        )


# ── Data Quality ───────────────────────────────────────────────────────────────

@dataclass
class FeatureBounds:
    """Expected value constraints for one feature column.

    Attributes:
        name:         Column name.
        min_val:      Minimum allowed value (None = no lower bound).
        max_val:      Maximum allowed value (None = no upper bound).
        max_null_rate: Maximum acceptable fraction of nulls (0.0 = no nulls).
    """

    name: str
    min_val: float | None = None
    max_val: float | None = None
    max_null_rate: float = 0.05


@dataclass
class FeatureQualityResult:
    """Data quality result for one feature column.

    Attributes:
        feature_name:      Column checked.
        null_rate:         Fraction of null values in the checked DataFrame.
        out_of_range_rate: Fraction of values outside [min_val, max_val].
        min_val:           Observed minimum (excluding nulls).
        max_val:           Observed maximum (excluding nulls).
        passed:            True if no quality constraints were violated.
        issues:            Human-readable issue descriptions.
    """

    feature_name: str
    null_rate: float
    out_of_range_rate: float
    min_val: float | None
    max_val: float | None
    passed: bool
    issues: list[str] = field(default_factory=list)


class FeatureQualityChecker:
    """Checks data quality constraints per feature column in a DataFrame."""

    def check(
        self,
        df: pd.DataFrame,
        bounds: list[FeatureBounds],
    ) -> list[FeatureQualityResult]:
        """Evaluate all specified feature bounds against a DataFrame.

        Args:
            df:     Current feature DataFrame (online or offline sample).
            bounds: List of FeatureBounds to enforce.

        Returns:
            One FeatureQualityResult per FeatureBounds entry.
        """
        results: list[FeatureQualityResult] = []
        n = len(df)

        for b in bounds:
            if b.name not in df.columns:
                results.append(FeatureQualityResult(
                    feature_name=b.name,
                    null_rate=1.0,
                    out_of_range_rate=0.0,
                    min_val=None,
                    max_val=None,
                    passed=False,
                    issues=[f"Column '{b.name}' not found in DataFrame"],
                ))
                continue

            col = df[b.name]
            null_count = int(col.isna().sum())
            null_rate = null_count / n if n > 0 else 0.0

            non_null = col.dropna()
            observed_min: float | None = float(non_null.min()) if len(non_null) > 0 else None
            observed_max: float | None = float(non_null.max()) if len(non_null) > 0 else None

            out_of_range = 0
            if b.min_val is not None:
                out_of_range += int((non_null < b.min_val).sum())
            if b.max_val is not None:
                out_of_range += int((non_null > b.max_val).sum())
            out_of_range_rate = out_of_range / n if n > 0 else 0.0

            issues: list[str] = []
            if null_rate > b.max_null_rate:
                issues.append(
                    f"Null rate {null_rate:.2%} exceeds threshold {b.max_null_rate:.2%}"
                )
            if out_of_range_rate > 0:
                issues.append(
                    f"Out-of-range rate {out_of_range_rate:.2%} for bounds "
                    f"[{b.min_val}, {b.max_val}]"
                )

            results.append(FeatureQualityResult(
                feature_name=b.name,
                null_rate=null_rate,
                out_of_range_rate=out_of_range_rate,
                min_val=observed_min,
                max_val=observed_max,
                passed=len(issues) == 0,
                issues=issues,
            ))

        return results


# ── Drift Detection ───────────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    """Drift detection result for one feature.

    Attributes:
        feature_name: Column checked.
        psi:          Population Stability Index (0=no drift, >0.20=high drift).
        ks_stat:      Kolmogorov-Smirnov statistic (0–1).
        severity:     "NONE" / "LOW" / "HIGH".
    """

    feature_name: str
    psi: float
    ks_stat: float
    severity: str  # "NONE" | "LOW" | "HIGH"

    @property
    def is_drifted(self) -> bool:
        return self.severity in ("LOW", "HIGH")


def _compute_psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """Compute Population Stability Index between reference and current distributions."""
    ref_clean = reference.dropna()
    cur_clean = current.dropna()
    if len(ref_clean) < 5 or len(cur_clean) < 5:
        return 0.0

    bins = np.linspace(
        min(ref_clean.min(), cur_clean.min()),
        max(ref_clean.max(), cur_clean.max()) + 1e-9,
        n_bins + 1,
    )
    ref_counts, _ = np.histogram(ref_clean, bins=bins)
    cur_counts, _ = np.histogram(cur_clean, bins=bins)

    ref_pct = ref_counts / len(ref_clean)
    cur_pct = cur_counts / len(cur_clean)

    # Avoid log(0)
    ref_pct = np.where(ref_pct == 0, 1e-9, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-9, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _compute_ks(reference: pd.Series, current: pd.Series) -> float:
    """Compute the Kolmogorov-Smirnov statistic between two distributions."""
    ref_clean = reference.dropna().sort_values().to_numpy()
    cur_clean = current.dropna().sort_values().to_numpy()
    if len(ref_clean) == 0 or len(cur_clean) == 0:
        return 0.0

    all_vals = np.concatenate([ref_clean, cur_clean])
    all_vals = np.unique(all_vals)

    ref_cdf = np.searchsorted(ref_clean, all_vals, side="right") / len(ref_clean)
    cur_cdf = np.searchsorted(cur_clean, all_vals, side="right") / len(cur_clean)
    return float(np.max(np.abs(ref_cdf - cur_cdf)))


class FeatureDriftMonitor:
    """Computes PSI and KS drift between reference and current feature distributions.

    Args:
        psi_low:  PSI threshold for LOW severity (default: 0.10).
        psi_high: PSI threshold for HIGH severity (default: 0.20).
        ks_low:   KS threshold for LOW severity (default: 0.05).
        ks_high:  KS threshold for HIGH severity (default: 0.10).
    """

    def __init__(
        self,
        psi_low: float = 0.10,
        psi_high: float = 0.20,
        ks_low: float = 0.05,
        ks_high: float = 0.10,
    ) -> None:
        self.psi_low = psi_low
        self.psi_high = psi_high
        self.ks_low = ks_low
        self.ks_high = ks_high

    def check(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str],
    ) -> list[FeatureDriftResult]:
        """Compute drift for each feature in feature_names.

        Args:
            reference_df:  Training distribution (baseline).
            current_df:    Current inference distribution.
            feature_names: Columns to check.

        Returns:
            One FeatureDriftResult per feature name.
        """
        results: list[FeatureDriftResult] = []
        for name in feature_names:
            if name not in reference_df.columns or name not in current_df.columns:
                results.append(FeatureDriftResult(
                    feature_name=name, psi=0.0, ks_stat=0.0, severity="NONE"
                ))
                continue

            psi = _compute_psi(reference_df[name], current_df[name])
            ks = _compute_ks(reference_df[name], current_df[name])

            # Severity based on worst signal
            if psi >= self.psi_high or ks >= self.ks_high:
                severity = "HIGH"
            elif psi >= self.psi_low or ks >= self.ks_low:
                severity = "LOW"
            else:
                severity = "NONE"

            results.append(FeatureDriftResult(
                feature_name=name, psi=psi, ks_stat=ks, severity=severity
            ))

        return results


# ── Feature Monitor Report ─────────────────────────────────────────────────────

@dataclass
class FeatureMonitorReport:
    """Combined monitoring report across freshness, quality, and drift.

    Attributes:
        freshness:       Freshness check results per feature view.
        quality:         Data quality results per feature column.
        drift:           Drift results per feature column.
        overall_passed:  True if no MISSING, CRITICAL quality, or HIGH drift found.
    """

    freshness: list[FreshnessCheck] = field(default_factory=list)
    quality: list[FeatureQualityResult] = field(default_factory=list)
    drift: list[FeatureDriftResult] = field(default_factory=list)
    overall_passed: bool = True

    def summary(self) -> str:
        lines = [f"FeatureMonitorReport: {'PASSED ✅' if self.overall_passed else 'FAILED ❌'}"]
        lines.append(f"  Freshness ({len(self.freshness)} views):")
        for fc in self.freshness:
            lines.append(f"    {fc.feature_view_name}: {fc.status.value} ({fc.age_hours:.1f}h)")
        lines.append(f"  Quality ({len(self.quality)} features):")
        for qr in self.quality:
            flag = "✅" if qr.passed else "❌"
            lines.append(f"    {qr.feature_name}: {flag} null={qr.null_rate:.2%}")
        lines.append(f"  Drift ({len(self.drift)} features):")
        for dr in self.drift:
            lines.append(f"    {dr.feature_name}: {dr.severity} (PSI={dr.psi:.3f})")
        return "\n".join(lines)


# ── Feature Monitor ────────────────────────────────────────────────────────────

class FeatureMonitor:
    """Orchestrates freshness, quality, and drift checks in one `.run()` call.

    Args:
        freshness_configs: List of (view_name, last_materialized_at, threshold_hours) tuples.
        quality_bounds:    List of FeatureBounds to enforce on current_df.
        drift_features:    Feature names to check for drift.
    """

    def __init__(
        self,
        freshness_configs: list[tuple[str, Any, float]] | None = None,
        quality_bounds: list[FeatureBounds] | None = None,
        drift_features: list[str] | None = None,
    ) -> None:
        self.freshness_configs = freshness_configs or []
        self.quality_bounds = quality_bounds or []
        self.drift_features = drift_features or []
        self._freshness_checker = FreshnessChecker()
        self._quality_checker = FeatureQualityChecker()
        self._drift_monitor = FeatureDriftMonitor()

    def run(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        now: datetime | None = None,
    ) -> FeatureMonitorReport:
        """Run all three monitoring pillars and return a combined report.

        Args:
            reference_df: Training-time feature distribution (baseline).
            current_df:   Recent inference feature distribution.
            now:          Override for current time (UTC).

        Returns:
            FeatureMonitorReport with all results and overall_passed flag.
        """
        # 1. Freshness
        freshness_results = [
            self._freshness_checker.check(name, last_mat, threshold, now=now)
            for name, last_mat, threshold in self.freshness_configs
        ]

        # 2. Quality
        quality_results = self._quality_checker.check(current_df, self.quality_bounds)

        # 3. Drift
        drift_results = self._drift_monitor.check(reference_df, current_df, self.drift_features)

        # Determine overall_passed
        any_missing = any(fc.status == FreshnessStatus.MISSING for fc in freshness_results)
        any_quality_fail = any(not qr.passed for qr in quality_results)
        any_high_drift = any(dr.severity == "HIGH" for dr in drift_results)

        overall_passed = not (any_missing or any_quality_fail or any_high_drift)

        return FeatureMonitorReport(
            freshness=freshness_results,
            quality=quality_results,
            drift=drift_results,
            overall_passed=overall_passed,
        )
