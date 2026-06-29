"""Tests for monitoring/evidently_reporter.py — EvidentlyReporter with fallback backend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoring.evidently_reporter import EvidentlyReporter, EvidentlyResult


def _make_df(shift: float = 0.0, n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "pay_ratio": rng.normal(0.5 + shift, 0.1, n),
        "util_rate": rng.normal(0.4 + shift, 0.1, n),
    })


# ── EvidentlyResult ────────────────────────────────────────────────────────────

class TestEvidentlyResult:
    def test_share_drifted_computed(self) -> None:
        r = EvidentlyResult(passed=True, n_drifted=1, n_tested=4, share_drifted=0.0)
        assert r.share_drifted == pytest.approx(0.25)

    def test_passed_reflected(self) -> None:
        r = EvidentlyResult(passed=False, n_drifted=3, n_tested=4, share_drifted=0.75)
        assert not r.passed

    def test_backend_default(self) -> None:
        r = EvidentlyResult(passed=True, n_drifted=0, n_tested=2, share_drifted=0.0)
        assert r.backend == "fallback"


# ── EvidentlyReporter — fallback data drift ───────────────────────────────────

class TestEvidentlyReporterDrift:
    def test_stable_on_identical_data(self) -> None:
        df = _make_df()
        reporter = EvidentlyReporter()
        result = reporter.run_data_drift(df, df)
        assert result.passed

    def test_drifted_on_large_shift(self) -> None:
        ref = _make_df(shift=0.0, n=300)
        cur = _make_df(shift=0.8, n=300)
        reporter = EvidentlyReporter(drift_share_ok=0.0)
        result = reporter.run_data_drift(ref, cur)
        assert not result.passed
        assert result.n_drifted > 0

    def test_backend_is_fallback_without_evidently(self) -> None:
        df = _make_df()
        reporter = EvidentlyReporter()
        result = reporter.run_data_drift(df, df)
        assert result.backend == "fallback"

    def test_custom_feature_names(self) -> None:
        ref = _make_df(shift=0.0, n=300)
        cur = _make_df(shift=0.8, n=300)
        reporter = EvidentlyReporter()
        result = reporter.run_data_drift(ref, cur, feature_names=["pay_ratio"])
        assert result.n_tested == 1

    def test_n_tested_equals_feature_count(self) -> None:
        df = _make_df()
        reporter = EvidentlyReporter()
        result = reporter.run_data_drift(df, df)
        assert result.n_tested == 2  # pay_ratio + util_rate

    def test_details_populated(self) -> None:
        ref = _make_df(shift=0.0, n=300)
        cur = _make_df(shift=0.8, n=300)
        reporter = EvidentlyReporter(drift_share_ok=0.0)
        result = reporter.run_data_drift(ref, cur)
        assert isinstance(result.details, dict)


# ── EvidentlyReporter — fallback quality ──────────────────────────────────────

class TestEvidentlyReporterQuality:
    def test_passes_clean_data(self) -> None:
        df = _make_df()
        reporter = EvidentlyReporter()
        result = reporter.run_data_quality(df)
        assert result.passed

    def test_fails_high_null_data(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "pay_ratio": rng.normal(0.5, 0.1, 100),
            "util_rate": [None] * 100,  # 100% null
        })
        reporter = EvidentlyReporter()
        result = reporter.run_data_quality(df, max_null_rate=0.05)
        assert not result.passed
        assert result.n_drifted >= 1

    def test_n_tested_equals_column_count(self) -> None:
        df = _make_df()
        reporter = EvidentlyReporter()
        result = reporter.run_data_quality(df)
        assert result.n_tested == len(df.columns)


# ── EvidentlyReporter — test suite ────────────────────────────────────────────

class TestEvidentlyReporterSuite:
    def test_passes_on_identical(self) -> None:
        df = _make_df()
        reporter = EvidentlyReporter()
        result = reporter.run_test_suite(df, df)
        assert result.passed

    def test_fails_on_large_drift(self) -> None:
        ref = _make_df(shift=0.0, n=300)
        cur = _make_df(shift=0.8, n=300)
        reporter = EvidentlyReporter(drift_share_ok=0.0)
        result = reporter.run_test_suite(ref, cur)
        assert not result.passed


# ── Save report ───────────────────────────────────────────────────────────────

class TestEvidentlyReporterSave:
    def test_save_creates_json(self, tmp_path: Path) -> None:
        df = _make_df()
        reporter = EvidentlyReporter(output_dir=str(tmp_path))
        result = reporter.run_data_drift(df, df, save_report=True, report_name="test_drift")
        assert result.report_path != ""
        report_file = Path(result.report_path)
        assert report_file.exists()

    def test_saved_json_has_required_keys(self, tmp_path: Path) -> None:
        df = _make_df()
        reporter = EvidentlyReporter(output_dir=str(tmp_path))
        result = reporter.run_data_drift(df, df, save_report=True, report_name="test")
        data = json.loads(Path(result.report_path).read_text())
        assert "passed" in data
        assert "n_drifted" in data
        assert "backend" in data
