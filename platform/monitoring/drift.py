"""Drift detection: PSI, KS, MMD, and classifier-based data drift.

Day 47 — four complementary drift metrics, each answering a different question:
  PSI        — how much has the overall distribution shifted? (industry credit-risk standard)
  KS stat    — is there a statistically significant boundary shift in the CDF?
  MMD        — do the feature distributions differ in kernel-space? (no binning needed)
  Classifier — can a model separate reference vs current samples? (AUC > 0.7 → HIGH drift)

Classes:
  DriftMetric        — PSI / KS / MMD / CLASSIFIER
  FeatureDriftResult — outcome for one feature + one metric
  DriftReport        — aggregated drift across all features and metrics
  DriftDetector      — computes all metrics and produces a DriftReport

See: docs/phase7/day47_drift.md
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


# ── Enumerations ──────────────────────────────────────────────────────────────

class DriftMetric(str, Enum):
    PSI        = "psi"
    KS         = "ks"
    MMD        = "mmd"
    CLASSIFIER = "classifier"


# ── Per-feature result ─────────────────────────────────────────────────────────

@dataclass
class FeatureDriftResult:
    """Drift measurement for one feature with one metric.

    Attributes:
        feature_name: Column name.
        metric:       Which drift metric was computed.
        value:        Computed metric value.
        threshold:    Threshold above which drift is flagged.
        drifted:      True if value >= threshold.
        severity:     "NONE" / "LOW" / "HIGH".
    """

    feature_name: str
    metric: DriftMetric
    value: float
    threshold: float
    drifted: bool
    severity: str

    @property
    def is_high(self) -> bool:
        return self.severity == "HIGH"


# ── Aggregate report ──────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    """Aggregated drift report across all features and metrics.

    Attributes:
        results:         All per-feature-per-metric results.
        overall_drifted: True if any HIGH severity drift found.
    """

    results: list[FeatureDriftResult] = field(default_factory=list)
    overall_drifted: bool = False

    def drifted_features(self) -> list[str]:
        return list({r.feature_name for r in self.results if r.drifted})

    def severity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"NONE": 0, "LOW": 0, "HIGH": 0}
        for r in self.results:
            counts[r.severity] = counts.get(r.severity, 0) + 1
        return counts

    def by_metric(self, metric: DriftMetric) -> list[FeatureDriftResult]:
        return [r for r in self.results if r.metric == metric]

    def summary(self) -> str:
        status = "DRIFTED ❌" if self.overall_drifted else "STABLE ✅"
        counts = self.severity_counts()
        lines = [
            f"DriftReport: {status}",
            f"  Results:  {len(self.results)} (HIGH={counts['HIGH']}, LOW={counts['LOW']})",
            f"  Features: {len(self.drifted_features())} drifted",
        ]
        for f in sorted(self.drifted_features()):
            lines.append(f"    {f}")
        return "\n".join(lines)


# ── Drift Detector ────────────────────────────────────────────────────────────

class DriftDetector:
    """Computes PSI, KS, MMD, and classifier-based drift per feature.

    Args:
        psi_low:              PSI threshold for LOW severity (default: 0.10).
        psi_high:             PSI threshold for HIGH severity (default: 0.20).
        ks_low:               KS threshold for LOW severity (default: 0.05).
        ks_high:              KS threshold for HIGH severity (default: 0.10).
        mmd_low:              MMD threshold for LOW severity (default: 0.05).
        mmd_high:             MMD threshold for HIGH severity (default: 0.10).
        classifier_auc_low:   Classifier AUC threshold for LOW (default: 0.65).
        classifier_auc_high:  Classifier AUC threshold for HIGH (default: 0.70).
        n_bins:               Bins for PSI computation (default: 10).
    """

    def __init__(
        self,
        psi_low: float = 0.10,
        psi_high: float = 0.20,
        ks_low: float = 0.05,
        ks_high: float = 0.10,
        mmd_low: float = 0.05,
        mmd_high: float = 0.10,
        classifier_auc_low: float = 0.65,
        classifier_auc_high: float = 0.70,
        n_bins: int = 10,
    ) -> None:
        self.psi_low = psi_low
        self.psi_high = psi_high
        self.ks_low = ks_low
        self.ks_high = ks_high
        self.mmd_low = mmd_low
        self.mmd_high = mmd_high
        self.classifier_auc_low = classifier_auc_low
        self.classifier_auc_high = classifier_auc_high
        self.n_bins = n_bins

    # ── Public per-metric computations ────────────────────────────────────────

    def compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Population Stability Index.  Returns 0.0 if either array is too small."""
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]
        if len(ref) < 5 or len(cur) < 5:
            return 0.0

        lo = min(ref.min(), cur.min())
        hi = max(ref.max(), cur.max()) + 1e-9
        bins = np.linspace(lo, hi, self.n_bins + 1)

        ref_pct = np.histogram(ref, bins=bins)[0] / len(ref)
        cur_pct = np.histogram(cur, bins=bins)[0] / len(cur)
        ref_pct = np.where(ref_pct == 0, 1e-9, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-9, cur_pct)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    def compute_ks(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Kolmogorov-Smirnov statistic (max CDF difference).  Returns 0.0 if < 5 samples."""
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]
        if len(ref) < 5 or len(cur) < 5:
            return 0.0

        # Merge and compute empirical CDFs
        all_vals = np.sort(np.concatenate([ref, cur]))
        cdf_ref = np.searchsorted(np.sort(ref), all_vals, side="right") / len(ref)
        cdf_cur = np.searchsorted(np.sort(cur), all_vals, side="right") / len(cur)
        return float(np.max(np.abs(cdf_ref - cdf_cur)))

    def compute_mmd(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        gamma: float = 1.0,
    ) -> float:
        """Maximum Mean Discrepancy with RBF kernel.  Returns 0.0 if < 5 samples.

        For scalars, gamma=1/σ² where σ is the feature std. Uses unbiased estimate.
        """
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]
        if len(ref) < 5 or len(cur) < 5:
            return 0.0

        # Subsample for speed if large
        rng = np.random.default_rng(42)
        if len(ref) > 500:
            ref = rng.choice(ref, 500, replace=False)
        if len(cur) > 500:
            cur = rng.choice(cur, 500, replace=False)

        def rbf_kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            diff = a[:, None] - b[None, :]
            return np.exp(-gamma * diff ** 2)

        k_rr = rbf_kernel(ref, ref)
        k_cc = rbf_kernel(cur, cur)
        k_rc = rbf_kernel(ref, cur)

        n, m = len(ref), len(cur)
        np.fill_diagonal(k_rr, 0)
        np.fill_diagonal(k_cc, 0)

        term1 = k_rr.sum() / (n * (n - 1)) if n > 1 else 0.0
        term2 = k_cc.sum() / (m * (m - 1)) if m > 1 else 0.0
        term3 = k_rc.mean()
        mmd2 = term1 + term2 - 2 * term3
        return float(max(0.0, math.sqrt(abs(mmd2))))

    def compute_classifier_auc(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """Classifier-based drift: AUC of a classifier trained to separate ref vs cur.

        Uses logistic regression on z-score normalised data. Returns AUC.
        AUC ≈ 0.5 → distributions indistinguishable; AUC > 0.7 → HIGH drift.
        Returns 0.5 if < 10 samples in either set.
        """
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]
        if len(ref) < 10 or len(cur) < 10:
            return 0.5

        # Subsample
        rng = np.random.default_rng(0)
        n = min(len(ref), len(cur), 300)
        ref_s = rng.choice(ref, n, replace=False)
        cur_s = rng.choice(cur, n, replace=False)

        X = np.concatenate([ref_s, cur_s]).reshape(-1, 1)
        y = np.array([0] * n + [1] * n)

        # Z-score normalise
        mu, sigma = X.mean(), X.std() + 1e-9
        X_norm = (X - mu) / sigma

        # Logistic regression — closed-form sigmoid on single feature
        # Use simple threshold sweep (no sklearn dependency)
        return self._logistic_auc(X_norm.ravel(), y)

    # ── Per-feature multi-metric check ────────────────────────────────────────

    def check_feature(
        self,
        name: str,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> list[FeatureDriftResult]:
        """Run all four drift metrics on a single feature array."""
        results: list[FeatureDriftResult] = []

        psi_val = self.compute_psi(reference, current)
        results.append(self._make_result(name, DriftMetric.PSI, psi_val, self.psi_low, self.psi_high))

        ks_val = self.compute_ks(reference, current)
        results.append(self._make_result(name, DriftMetric.KS, ks_val, self.ks_low, self.ks_high))

        mmd_val = self.compute_mmd(reference, current)
        results.append(self._make_result(name, DriftMetric.MMD, mmd_val, self.mmd_low, self.mmd_high))

        auc = self.compute_classifier_auc(reference, current)
        results.append(self._make_result(name, DriftMetric.CLASSIFIER, auc, self.classifier_auc_low, self.classifier_auc_high))

        return results

    def run(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> DriftReport:
        """Run all drift metrics on all shared numeric features.

        Args:
            reference_df:  Training / baseline DataFrame.
            current_df:    Current serving sample DataFrame.
            feature_names: Columns to check (default: all shared numeric columns).

        Returns:
            DriftReport with overall_drifted=True if any HIGH drift found.
        """
        if feature_names is None:
            shared = set(reference_df.columns) & set(current_df.columns)
            feature_names = [
                c for c in sorted(shared)
                if pd.api.types.is_numeric_dtype(reference_df[c])
            ]

        all_results: list[FeatureDriftResult] = []
        for name in feature_names:
            if name not in reference_df.columns or name not in current_df.columns:
                continue
            ref_arr = reference_df[name].to_numpy(dtype=float)
            cur_arr = current_df[name].to_numpy(dtype=float)
            all_results.extend(self.check_feature(name, ref_arr, cur_arr))

        overall_drifted = any(r.is_high for r in all_results)
        return DriftReport(results=all_results, overall_drifted=overall_drifted)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _make_result(
        self,
        name: str,
        metric: DriftMetric,
        value: float,
        low_threshold: float,
        high_threshold: float,
    ) -> FeatureDriftResult:
        if value >= high_threshold:
            severity = "HIGH"
            drifted = True
        elif value >= low_threshold:
            severity = "LOW"
            drifted = True
        else:
            severity = "NONE"
            drifted = False
        return FeatureDriftResult(
            feature_name=name,
            metric=metric,
            value=value,
            threshold=high_threshold,
            drifted=drifted,
            severity=severity,
        )

    @staticmethod
    def _logistic_auc(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute AUC via Mann-Whitney U — no sklearn dependency."""
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        u = sum(1 for p in pos for n in neg if p > n) + \
            0.5 * sum(1 for p in pos for n in neg if p == n)
        return float(u / (len(pos) * len(neg)))
