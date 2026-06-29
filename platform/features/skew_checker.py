"""Train-serve skew detection: schema, feature distribution, and prediction skew.

Consolidation for Phase 6 (Day 45) — proves zero train-serve skew by checking:
  1. Schema skew    — column set differences between training and serving DataFrames
  2. Feature skew   — distribution divergence (PSI + mean delta) per feature
  3. Prediction skew — score differences on shared test inputs

Classes:
  SkewType               — SCHEMA / FEATURE / LABEL / PREDICTION
  SkewEvidence           — one detected skew instance with severity
  TrainServeSkewReport   — aggregated report across all skew types
  TrainServeSkewChecker  — runs all three checks and produces the report

See: docs/phase6/day45_zero_skew.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd


# ── Skew Type ─────────────────────────────────────────────────────────────────

class SkewType(str, Enum):
    SCHEMA = "schema"
    FEATURE = "feature"
    LABEL = "label"
    PREDICTION = "prediction"


# ── Skew Evidence ─────────────────────────────────────────────────────────────

@dataclass
class SkewEvidence:
    """One detected skew instance.

    Attributes:
        feature_name: Column or metric where skew was found.
        skew_type:    Type of skew.
        train_value:  Observed value in training (e.g. mean, presence flag).
        serve_value:  Observed value in serving.
        delta:        Absolute difference between train and serve values.
        severity:     "NONE" / "LOW" / "HIGH".
        description:  Human-readable description.
    """

    feature_name: str
    skew_type: SkewType
    train_value: float
    serve_value: float
    delta: float
    severity: str
    description: str = ""

    @property
    def is_high(self) -> bool:
        return self.severity == "HIGH"


# ── Report ─────────────────────────────────────────────────────────────────────

@dataclass
class TrainServeSkewReport:
    """Aggregated train-serve skew report across all checks.

    Attributes:
        evidences: All detected skew instances (all severity levels).
        zero_skew: True if no HIGH severity skew was found.
    """

    evidences: list[SkewEvidence] = field(default_factory=list)
    zero_skew: bool = True

    def high_severity_count(self) -> int:
        return sum(1 for e in self.evidences if e.is_high)

    def by_type(self, skew_type: SkewType) -> list[SkewEvidence]:
        return [e for e in self.evidences if e.skew_type == skew_type]

    def summary(self) -> str:
        status = "ZERO SKEW ✅" if self.zero_skew else "SKEW DETECTED ❌"
        lines = [f"TrainServeSkewReport: {status}"]
        lines.append(f"  Total evidences: {len(self.evidences)}")
        lines.append(f"  HIGH severity:   {self.high_severity_count()}")
        for stype in SkewType:
            items = self.by_type(stype)
            if items:
                lines.append(f"  {stype.value}: {len(items)} items")
                for ev in items:
                    lines.append(f"    [{ev.severity}] {ev.feature_name}: {ev.description}")
        return "\n".join(lines)


# ── Checker ───────────────────────────────────────────────────────────────────

class TrainServeSkewChecker:
    """Runs three skew checks and returns a combined TrainServeSkewReport.

    Args:
        psi_threshold_low:   PSI above this is LOW severity (default: 0.10).
        psi_threshold_high:  PSI above this is HIGH severity (default: 0.20).
        mean_delta_low:      Relative mean delta (%) above this is LOW (default: 0.10).
        mean_delta_high:     Relative mean delta (%) above this is HIGH (default: 0.25).
        pred_delta_threshold: Max allowed score delta for prediction skew check (default: 1e-3).
    """

    def __init__(
        self,
        psi_threshold_low: float = 0.10,
        psi_threshold_high: float = 0.20,
        mean_delta_low: float = 0.10,
        mean_delta_high: float = 0.25,
        pred_delta_threshold: float = 1e-3,
    ) -> None:
        self.psi_low = psi_threshold_low
        self.psi_high = psi_threshold_high
        self.mean_delta_low = mean_delta_low
        self.mean_delta_high = mean_delta_high
        self.pred_delta_threshold = pred_delta_threshold

    # -- Schema Skew --

    def check_schema_skew(
        self,
        train_df: pd.DataFrame,
        serve_df: pd.DataFrame,
    ) -> list[SkewEvidence]:
        """Detect columns present in training but missing at serving and vice versa.

        Returns:
            List of SkewEvidence — one per missing/extra column.
            All schema mismatches are HIGH severity.
        """
        train_cols = set(train_df.columns)
        serve_cols = set(serve_df.columns)

        evidences: list[SkewEvidence] = []

        for col in sorted(train_cols - serve_cols):
            evidences.append(SkewEvidence(
                feature_name=col,
                skew_type=SkewType.SCHEMA,
                train_value=1.0,
                serve_value=0.0,
                delta=1.0,
                severity="HIGH",
                description=f"Column '{col}' in training but MISSING at serving",
            ))

        for col in sorted(serve_cols - train_cols):
            evidences.append(SkewEvidence(
                feature_name=col,
                skew_type=SkewType.SCHEMA,
                train_value=0.0,
                serve_value=1.0,
                delta=1.0,
                severity="LOW",  # extra columns at serving = warning, not critical
                description=f"Column '{col}' at serving but NOT in training",
            ))

        return evidences

    # -- Feature Skew --

    def check_feature_skew(
        self,
        train_df: pd.DataFrame,
        serve_df: pd.DataFrame,
        feature_names: list[str] | None = None,
    ) -> list[SkewEvidence]:
        """Compute PSI and mean delta for each shared numeric feature.

        Args:
            train_df:      Training feature DataFrame.
            serve_df:      Serving feature DataFrame.
            feature_names: Columns to check (default: all shared numeric columns).

        Returns:
            List of SkewEvidence with PSI-based severity.
        """
        if feature_names is None:
            shared = set(train_df.columns) & set(serve_df.columns)
            feature_names = [
                c for c in sorted(shared)
                if pd.api.types.is_numeric_dtype(train_df[c])
            ]

        evidences: list[SkewEvidence] = []
        for name in feature_names:
            if name not in train_df.columns or name not in serve_df.columns:
                continue

            psi = self._psi(train_df[name], serve_df[name])
            train_mean = float(train_df[name].dropna().mean()) if len(train_df[name].dropna()) else 0.0
            serve_mean = float(serve_df[name].dropna().mean()) if len(serve_df[name].dropna()) else 0.0
            mean_delta = abs(train_mean - serve_mean) / (abs(train_mean) + 1e-9)

            severity = "NONE"
            if psi >= self.psi_high or mean_delta >= self.mean_delta_high:
                severity = "HIGH"
            elif psi >= self.psi_low or mean_delta >= self.mean_delta_low:
                severity = "LOW"

            if severity != "NONE":
                evidences.append(SkewEvidence(
                    feature_name=name,
                    skew_type=SkewType.FEATURE,
                    train_value=train_mean,
                    serve_value=serve_mean,
                    delta=abs(train_mean - serve_mean),
                    severity=severity,
                    description=f"PSI={psi:.3f}, mean delta={mean_delta:.1%}",
                ))

        return evidences

    # -- Prediction Skew --

    def check_prediction_skew(
        self,
        train_scores: list[float],
        serve_scores: list[float],
    ) -> list[SkewEvidence]:
        """Detect score discrepancies on the same inputs between training and serving.

        Args:
            train_scores: Scores produced by the training-path model.
            serve_scores: Scores produced by the serving-path model (same inputs).

        Returns:
            List of SkewEvidence — one if max delta exceeds threshold.
        """
        if not train_scores or not serve_scores:
            return []
        if len(train_scores) != len(serve_scores):
            return [SkewEvidence(
                feature_name="prediction",
                skew_type=SkewType.PREDICTION,
                train_value=float(len(train_scores)),
                serve_value=float(len(serve_scores)),
                delta=abs(len(train_scores) - len(serve_scores)),
                severity="HIGH",
                description="Score arrays have different lengths",
            )]

        deltas = [abs(t - s) for t, s in zip(train_scores, serve_scores)]
        max_delta = max(deltas)
        mean_delta = sum(deltas) / len(deltas)

        if max_delta <= self.pred_delta_threshold:
            return []

        severity = "HIGH" if max_delta > self.pred_delta_threshold * 10 else "LOW"
        return [SkewEvidence(
            feature_name="prediction",
            skew_type=SkewType.PREDICTION,
            train_value=float(np.mean(train_scores)),
            serve_value=float(np.mean(serve_scores)),
            delta=max_delta,
            severity=severity,
            description=f"Max score delta={max_delta:.6f}, mean={mean_delta:.6f}",
        )]

    # -- Full Run --

    def run(
        self,
        train_df: pd.DataFrame,
        serve_df: pd.DataFrame,
        feature_names: list[str] | None = None,
        train_scores: list[float] | None = None,
        serve_scores: list[float] | None = None,
    ) -> TrainServeSkewReport:
        """Run all skew checks and return a combined report.

        Args:
            train_df:      Training feature DataFrame.
            serve_df:      Serving feature DataFrame.
            feature_names: Feature columns to check for distribution skew.
            train_scores:  Training-path model scores (optional, for prediction skew).
            serve_scores:  Serving-path model scores (optional, for prediction skew).

        Returns:
            TrainServeSkewReport with zero_skew=True if no HIGH severity found.
        """
        evidences: list[SkewEvidence] = []

        evidences.extend(self.check_schema_skew(train_df, serve_df))
        evidences.extend(self.check_feature_skew(train_df, serve_df, feature_names))

        if train_scores is not None and serve_scores is not None:
            evidences.extend(self.check_prediction_skew(train_scores, serve_scores))

        zero_skew = not any(e.is_high for e in evidences)
        return TrainServeSkewReport(evidences=evidences, zero_skew=zero_skew)

    # -- Internals --

    @staticmethod
    def _psi(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
        """Population Stability Index between two numeric series."""
        ref = reference.dropna()
        cur = current.dropna()
        if len(ref) < 5 or len(cur) < 5:
            return 0.0

        bins = np.linspace(
            min(ref.min(), cur.min()),
            max(ref.max(), cur.max()) + 1e-9,
            n_bins + 1,
        )
        ref_pct = np.histogram(ref, bins=bins)[0] / len(ref)
        cur_pct = np.histogram(cur, bins=bins)[0] / len(cur)
        ref_pct = np.where(ref_pct == 0, 1e-9, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-9, cur_pct)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
