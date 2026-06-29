"""Train/serve skew detection using PSI, KS test, and Jensen-Shannon divergence.

Three metrics, three perspectives on the same question: "has this feature's
distribution changed since training?"

PSI (Population Stability Index):
    Industry-standard credit risk metric. Quantifies the magnitude of shift.
    Computed by binning both distributions and comparing bin fractions.

KS test (Kolmogorov-Smirnov two-sample):
    Non-parametric significance test. Measures max CDF difference.
    Use for: statistical confirmation that shift is not sampling noise.

Jensen-Shannon divergence:
    Symmetric, bounded [0, 1] divergence. Works on histograms / discrete distributions.
    Use for: categorical features; complementary severity measure to PSI.

PSI thresholds: < 0.10 stable | 0.10–0.20 slight | > 0.20 major
KS threshold:   p-value < 0.05 → significant shift

See: docs/phase3/day21_train_serve_skew.md for theory.

Usage:
    from monitoring.skew_detector import detect_skew, skew_summary
    from monitoring.reference_stats import load_reference_stats

    ref = load_reference_stats(Path("models/reference_stats.json"))
    report = detect_skew(serving_df, ref)
    print(skew_summary(report).to_string())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from monitoring.reference_stats import ReferenceStats

log = logging.getLogger(__name__)

# PSI severity thresholds
PSI_STABLE = 0.10
PSI_MODERATE = 0.20

# KS significance threshold
KS_PVALUE_THRESHOLD = 0.05

# Bin count for PSI computation
_N_BINS = 10
_EPSILON = 1e-4  # avoids ln(0) in PSI when a bin is empty


@dataclass
class FeatureSkewResult:
    """Per-feature skew assessment across all three metrics.

    Attributes:
        feature:       Column name.
        psi:           Population Stability Index.
        ks_stat:       KS test statistic (None for categorical).
        ks_pvalue:     KS test p-value (None for categorical).
        js_divergence: Jensen-Shannon divergence.
        severity:      "stable" | "moderate" | "major".
        flag:          True if PSI > PSI_STABLE or KS p-value < 0.05.
    """

    feature: str
    psi: float
    ks_stat: float | None
    ks_pvalue: float | None
    js_divergence: float
    severity: str
    flag: bool


@dataclass
class SkewReport:
    """Skew detection results for a full serving batch.

    Attributes:
        feature_results: List of FeatureSkewResult, one per feature.
        overall_severity: Worst severity across all features.
        n_flagged: Number of features with flag=True.
        n_features: Total features evaluated.
        serving_n_rows: Number of rows in the serving batch.
        reference_model_version: Model version the reference corresponds to.
    """

    feature_results: list[FeatureSkewResult]
    overall_severity: str
    n_flagged: int
    n_features: int
    serving_n_rows: int
    reference_model_version: str


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = _N_BINS,
) -> float:
    """Compute the Population Stability Index between two 1-D arrays.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    where expected = reference distribution, actual = current distribution.

    Bins are defined by the reference data range; values outside this range
    fall into the edge bins (clipped to prevent index-out-of-range).

    Args:
        reference: 1-D array of training feature values (non-null).
        current:   1-D array of serving feature values (non-null).
        n_bins:    Number of equal-width bins.

    Returns:
        PSI value (float). 0 = identical distributions.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    # Edge case: constant feature (no variation)
    if ref.std() == 0:
        return 0.0 if cur.std() == 0 else 1.0

    bin_edges = np.linspace(ref.min(), ref.max(), n_bins + 1)
    # Extend edge bins slightly to include the boundary values
    bin_edges[0] -= 1e-8
    bin_edges[-1] += 1e-8

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = ref_counts / max(len(ref), 1) + _EPSILON
    cur_pct = cur_counts / max(len(cur), 1) + _EPSILON

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_ks(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Args:
        reference: 1-D array of training feature values.
        current:   1-D array of serving feature values.

    Returns:
        (ks_stat, pvalue). ks_stat = max CDF difference. pvalue = significance.
    """
    result = stats.ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)


def compute_js(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = _N_BINS,
) -> float:
    """Jensen-Shannon divergence between two 1-D arrays.

    Uses the same binning as PSI. Returns a value in [0, 1].
    0 = identical distributions, 1 = completely disjoint support.

    The scipy jensenshannon function returns the *square root* of the JS
    divergence (i.e. the JS distance), which is bounded in [0, 1].

    Args:
        reference: 1-D training feature values.
        current:   1-D serving feature values.
        n_bins:    Number of histogram bins.

    Returns:
        JS distance in [0, 1].
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    if ref.std() == 0 and cur.std() == 0:
        return 0.0

    all_vals = np.concatenate([ref, cur])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    bin_edges[0] -= 1e-8
    bin_edges[-1] += 1e-8

    ref_hist, _ = np.histogram(ref, bins=bin_edges)
    cur_hist, _ = np.histogram(cur, bins=bin_edges)

    # Add epsilon to avoid zero-probability bins
    ref_hist = ref_hist + _EPSILON
    cur_hist = cur_hist + _EPSILON

    ref_prob = ref_hist / ref_hist.sum()
    cur_prob = cur_hist / cur_hist.sum()

    return float(jensenshannon(ref_prob, cur_prob))


def _classify_severity(psi: float, ks_pvalue: float | None) -> tuple[str, bool]:
    """Map PSI + KS p-value to severity string and flag."""
    if psi > PSI_MODERATE:
        return "major", True
    if psi > PSI_STABLE:
        return "moderate", True
    if ks_pvalue is not None and ks_pvalue < KS_PVALUE_THRESHOLD:
        return "moderate", True
    return "stable", False


def detect_skew(
    serving_df: pd.DataFrame,
    reference: ReferenceStats,
    *,
    features: list[str] | None = None,
) -> SkewReport:
    """Detect distribution shift between serving data and training reference.

    Evaluates only numeric features. Categorical features are skipped with a
    warning (use JS divergence on value-count histograms for those separately).

    Args:
        serving_df:  Incoming serving batch (DataFrame with feature columns).
        reference:   Training-time reference stats (ReferenceStats).
        features:    Optional explicit list of features to evaluate.
                     Defaults to the intersection of reference.feature_names
                     and serving_df numeric columns.

    Returns:
        SkewReport with per-feature results and overall severity.
    """
    ref_stats = reference.dataset_stats

    if features is None:
        candidate_features = reference.feature_names
    else:
        candidate_features = features

    numeric_cols = set(serving_df.select_dtypes(include="number").columns)
    eval_features = [f for f in candidate_features if f in numeric_cols and f in ref_stats.columns]

    if not eval_features:
        log.warning("No common numeric features to evaluate for skew")

    results: list[FeatureSkewResult] = []

    for feat in eval_features:
        ref_col = ref_stats.columns[feat]

        # Re-construct a synthetic reference sample from percentiles
        # (We store stats, not raw data — use known percentiles as representative values)
        if ref_col.mean is None:
            log.debug("Skipping non-numeric column %r", feat)
            continue

        serving_vals = serving_df[feat].dropna().to_numpy(dtype=float)
        if len(serving_vals) == 0:
            log.warning("No non-null serving values for feature %r — skipping", feat)
            continue

        # Build reference sample from stored percentiles
        # This approximation is sufficient for PSI / KS / JS estimation
        ref_sample = np.array([
            ref_col.p5 or ref_col.mean,
            ref_col.p25 or ref_col.mean,
            ref_col.p50 or ref_col.mean,
            ref_col.p75 or ref_col.mean,
            ref_col.p95 or ref_col.mean,
        ])
        # Expand by sampling a Gaussian approximation centred on the reference mean
        rng = np.random.default_rng(42)
        ref_std = max(ref_col.std or 1.0, 1e-8)
        ref_expanded = rng.normal(loc=ref_col.mean, scale=ref_std, size=len(serving_vals))

        psi = compute_psi(ref_expanded, serving_vals)
        ks_stat, ks_pvalue = compute_ks(ref_expanded, serving_vals)
        js_div = compute_js(ref_expanded, serving_vals)
        severity, flag = _classify_severity(psi, ks_pvalue)

        results.append(FeatureSkewResult(
            feature=feat,
            psi=psi,
            ks_stat=ks_stat,
            ks_pvalue=ks_pvalue,
            js_divergence=js_div,
            severity=severity,
            flag=flag,
        ))

    n_flagged = sum(1 for r in results if r.flag)
    severities = [r.severity for r in results]
    if "major" in severities:
        overall = "major"
    elif "moderate" in severities:
        overall = "moderate"
    else:
        overall = "stable"

    log.info(
        "Skew detection: %d/%d features flagged | overall_severity=%s | serving_rows=%d",
        n_flagged, len(results), overall, len(serving_df),
    )

    return SkewReport(
        feature_results=results,
        overall_severity=overall,
        n_flagged=n_flagged,
        n_features=len(results),
        serving_n_rows=len(serving_df),
        reference_model_version=reference.model_version,
    )


def skew_summary(report: SkewReport) -> pd.DataFrame:
    """Convert a SkewReport to a DataFrame sorted by PSI descending.

    Returns columns: feature, psi, ks_stat, ks_pvalue, js_divergence, severity, flag.
    Suitable for logging to MLflow as a CSV artifact or displaying in a notebook.
    """
    if not report.feature_results:
        return pd.DataFrame(columns=["feature", "psi", "ks_stat", "ks_pvalue",
                                     "js_divergence", "severity", "flag"])

    rows = [
        {
            "feature": r.feature,
            "psi": r.psi,
            "ks_stat": r.ks_stat,
            "ks_pvalue": r.ks_pvalue,
            "js_divergence": r.js_divergence,
            "severity": r.severity,
            "flag": r.flag,
        }
        for r in report.feature_results
    ]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
