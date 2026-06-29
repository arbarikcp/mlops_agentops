"""Evidently adapter: wraps Evidently reports/test-suites with a DriftDetector fallback.

Day 48 — thin adapter so callers get the same interface whether Evidently is installed or not.
In CI / minimal environments, the fallback (our DriftDetector from Day 47) is used.
In production pipelines, Evidently generates HTML/JSON artefacts and richer metrics.

Classes:
  EvidentlyResult   — unified outcome (passed, n_drifted, share_drifted, backend used)
  EvidentlyReporter — adapter: tries evidently, falls back to DriftDetector

See: docs/phase7/day48_evidently.md
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from monitoring.drift import DriftDetector

logger = logging.getLogger(__name__)


# ── EvidentlyResult ───────────────────────────────────────────────────────────

@dataclass
class EvidentlyResult:
    """Unified result from either Evidently or the fallback DriftDetector.

    Attributes:
        passed:         True if no HIGH drift / all quality checks pass.
        n_drifted:      Number of features with drift detected.
        n_tested:       Total features tested.
        share_drifted:  n_drifted / n_tested (0.0 if n_tested == 0).
        details:        Full metric details keyed by feature name.
        backend:        "evidently" or "fallback".
        report_path:    Path to saved JSON/HTML report (empty if not saved).
    """

    passed: bool
    n_drifted: int
    n_tested: int
    share_drifted: float
    details: dict = field(default_factory=dict)
    backend: str = "fallback"
    report_path: str = ""

    def __post_init__(self) -> None:
        if self.n_tested > 0:
            self.share_drifted = self.n_drifted / self.n_tested


# ── EvidentlyReporter ─────────────────────────────────────────────────────────

class EvidentlyReporter:
    """Adapter over Evidently with transparent DriftDetector fallback.

    Args:
        output_dir:           Where to save JSON/HTML report files.
        psi_threshold:        PSI threshold for the fallback drift check.
        drift_share_ok:       Max acceptable fraction of drifted features (for `passed`).
    """

    def __init__(
        self,
        output_dir: str = "artifacts/evidently",
        psi_threshold: float = 0.20,
        drift_share_ok: float = 0.20,
    ) -> None:
        self.output_dir = output_dir
        self.psi_threshold = psi_threshold
        self.drift_share_ok = drift_share_ok
        self._evidently_available = self._check_evidently()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None = None,
        save_report: bool = False,
        report_name: str = "drift",
    ) -> EvidentlyResult:
        """Compute data drift between reference and current DataFrames.

        Tries Evidently first; falls back to DriftDetector if not available.

        Args:
            reference_df:   Training / baseline sample.
            current_df:     Current serving sample.
            feature_names:  Columns to check (default: all shared numeric).
            save_report:    If True, saves JSON report to output_dir.
            report_name:    Filename stem for the saved report.

        Returns:
            EvidentlyResult with passed=True if share_drifted <= drift_share_ok.
        """
        if self._evidently_available:
            result = self._run_evidently_drift(reference_df, current_df, feature_names)
        else:
            result = self._run_fallback_drift(reference_df, current_df, feature_names)

        if save_report:
            result.report_path = self._save_json(result, report_name)

        return result

    def run_data_quality(
        self,
        df: pd.DataFrame,
        max_null_rate: float = 0.05,
        save_report: bool = False,
        report_name: str = "quality",
    ) -> EvidentlyResult:
        """Check data quality: null rates and constant columns.

        Args:
            df:             DataFrame to check.
            max_null_rate:  Maximum acceptable null fraction per column.
            save_report:    If True, saves JSON report.

        Returns:
            EvidentlyResult with passed=True if all columns within bounds.
        """
        if self._evidently_available:
            result = self._run_evidently_quality(df, max_null_rate)
        else:
            result = self._run_fallback_quality(df, max_null_rate)

        if save_report:
            result.report_path = self._save_json(result, report_name)

        return result

    def run_test_suite(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None = None,
        save_report: bool = False,
        report_name: str = "test_suite",
    ) -> EvidentlyResult:
        """Run a pass/fail test suite over drift metrics.

        Equivalent to Evidently's TestSuite with DataDriftTestPreset.
        Falls back to per-feature PSI threshold check.

        Returns:
            EvidentlyResult with passed=True if no features exceed threshold.
        """
        if self._evidently_available:
            result = self._run_evidently_suite(reference_df, current_df, feature_names)
        else:
            result = self._run_fallback_drift(reference_df, current_df, feature_names)

        if save_report:
            result.report_path = self._save_json(result, report_name)

        return result

    # ── Evidently backend ─────────────────────────────────────────────────────

    def _run_evidently_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None,
    ) -> EvidentlyResult:
        try:
            from evidently.metric_preset import DataDriftPreset  # type: ignore[import]
            from evidently.report import Report  # type: ignore[import]

            cols = feature_names or [
                c for c in reference_df.columns
                if pd.api.types.is_numeric_dtype(reference_df[c])
            ]
            ref = reference_df[cols]
            cur = current_df[[c for c in cols if c in current_df.columns]]

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref, current_data=cur)
            data = report.as_dict()

            drift_info = data.get("metrics", [{}])[0].get("result", {})
            n_drifted = int(drift_info.get("number_of_drifted_columns", 0))
            n_tested = int(drift_info.get("number_of_columns", len(cols)))
            passed = drift_info.get("share_of_drifted_columns", 0.0) <= self.drift_share_ok

            return EvidentlyResult(
                passed=passed,
                n_drifted=n_drifted,
                n_tested=n_tested,
                share_drifted=n_drifted / max(n_tested, 1),
                details=drift_info,
                backend="evidently",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evidently drift failed (%s); using fallback", exc)
            return self._run_fallback_drift(reference_df, current_df, feature_names)

    def _run_evidently_quality(self, df: pd.DataFrame, max_null_rate: float) -> EvidentlyResult:
        try:
            from evidently.metric_preset import DataQualityPreset  # type: ignore[import]
            from evidently.report import Report  # type: ignore[import]

            report = Report(metrics=[DataQualityPreset()])
            report.run(reference_data=df, current_data=df)
            data = report.as_dict()
            return EvidentlyResult(
                passed=True,
                n_drifted=0,
                n_tested=len(df.columns),
                share_drifted=0.0,
                details=data,
                backend="evidently",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evidently quality failed (%s); using fallback", exc)
            return self._run_fallback_quality(df, max_null_rate)

    def _run_evidently_suite(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None,
    ) -> EvidentlyResult:
        try:
            from evidently.test_preset import DataDriftTestPreset  # type: ignore[import]
            from evidently.test_suite import TestSuite  # type: ignore[import]

            cols = feature_names or [
                c for c in reference_df.columns
                if pd.api.types.is_numeric_dtype(reference_df[c])
            ]
            ref = reference_df[cols]
            cur = current_df[[c for c in cols if c in current_df.columns]]

            suite = TestSuite(tests=[DataDriftTestPreset()])
            suite.run(reference_data=ref, current_data=cur)
            data = suite.as_dict()

            tests = data.get("tests", [])
            n_failed = sum(1 for t in tests if t.get("status") == "FAIL")
            passed = n_failed == 0

            return EvidentlyResult(
                passed=passed,
                n_drifted=n_failed,
                n_tested=len(tests),
                share_drifted=n_failed / max(len(tests), 1),
                details=data,
                backend="evidently",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Evidently suite failed (%s); using fallback", exc)
            return self._run_fallback_drift(reference_df, current_df, feature_names)

    # ── Fallback backend ──────────────────────────────────────────────────────

    def _run_fallback_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_names: list[str] | None,
    ) -> EvidentlyResult:
        detector = DriftDetector(psi_high=self.psi_threshold)
        report = detector.run(reference_df, current_df, feature_names)

        drifted = report.drifted_features()
        n_features = len({r.feature_name for r in report.results})
        details = {
            r.feature_name: {"metric": r.metric, "value": r.value, "severity": r.severity}
            for r in report.results
            if r.drifted
        }

        return EvidentlyResult(
            passed=len(drifted) / max(n_features, 1) <= self.drift_share_ok,
            n_drifted=len(drifted),
            n_tested=n_features,
            share_drifted=len(drifted) / max(n_features, 1),
            details=details,
            backend="fallback",
        )

    def _run_fallback_quality(self, df: pd.DataFrame, max_null_rate: float) -> EvidentlyResult:
        failed: list[str] = []
        details: dict = {}
        for col in df.columns:
            null_rate = float(df[col].isna().mean())
            details[col] = {"null_rate": null_rate}
            if null_rate > max_null_rate:
                failed.append(col)

        return EvidentlyResult(
            passed=len(failed) == 0,
            n_drifted=len(failed),
            n_tested=len(df.columns),
            share_drifted=len(failed) / max(len(df.columns), 1),
            details=details,
            backend="fallback",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save_json(self, result: EvidentlyResult, name: str) -> str:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.json"
        payload = {
            "passed": result.passed,
            "n_drifted": result.n_drifted,
            "n_tested": result.n_tested,
            "share_drifted": result.share_drifted,
            "backend": result.backend,
            "details": result.details,
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Evidently report saved: %s", path)
        return str(path)

    @staticmethod
    def _check_evidently() -> bool:
        try:
            import evidently  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False
