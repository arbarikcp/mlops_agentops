"""Tests for monitoring/reference_stats.py."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from monitoring.reference_stats import (
    ReferenceStats,
    check_feature_alignment,
    compute_reference_stats,
    load_reference_stats,
    save_reference_stats,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_feature_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feature_a": rng.standard_normal(n),
        "feature_b": rng.uniform(0, 100, n),
        "feature_c": rng.integers(1, 5, n).astype(float),
    })


# ── compute_reference_stats ───────────────────────────────────────────────────

class TestComputeReferenceStats:
    def test_returns_reference_stats(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        assert isinstance(ref, ReferenceStats)

    def test_model_version_stored(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v2.3")
        assert ref.model_version == "v2.3"

    def test_n_training_rows_correct(self) -> None:
        df = _make_feature_df(n=150)
        ref = compute_reference_stats(df, model_version="v1")
        assert ref.n_training_rows == 150

    def test_feature_names_default_to_columns(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        assert set(ref.feature_names) == set(df.columns)

    def test_feature_names_explicit(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1", feature_names=["feature_a", "feature_b"])
        assert ref.feature_names == ["feature_a", "feature_b"]

    def test_dataset_stats_columns_match(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        for col in df.columns:
            assert col in ref.dataset_stats.columns

    def test_training_date_set(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        assert ref.training_date is not None
        assert len(ref.training_date) > 10  # ISO-8601 string


# ── save / load roundtrip ─────────────────────────────────────────────────────

class TestSaveLoadReferenceStats:
    def test_save_creates_file(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reference_stats.json"
            save_reference_stats(ref, path)
            assert path.exists()

    def test_saved_file_is_valid_json(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "reference_stats.json"
            save_reference_stats(ref, path)
            with open(path) as f:
                data = json.load(f)
            assert "model_version" in data

    def test_load_roundtrip(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v_roundtrip")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            save_reference_stats(ref, path)
            loaded = load_reference_stats(path)
        assert loaded.model_version == "v_roundtrip"
        assert loaded.n_training_rows == len(df)
        assert set(loaded.feature_names) == set(ref.feature_names)

    def test_load_preserves_feature_means(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stats.json"
            save_reference_stats(ref, path)
            loaded = load_reference_stats(path)
        orig_mean = ref.dataset_stats.columns["feature_a"].mean
        loaded_mean = loaded.dataset_stats.columns["feature_a"].mean
        assert abs(orig_mean - loaded_mean) < 1e-10

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_reference_stats(Path("/nonexistent/path/stats.json"))

    def test_save_creates_parent_dirs(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "stats.json"
            save_reference_stats(ref, path)
            assert path.exists()


# ── check_feature_alignment ───────────────────────────────────────────────────

class TestCheckFeatureAlignment:
    def test_aligned_returns_true(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        result = check_feature_alignment(ref, df)
        assert result["aligned"] is True
        assert result["missing_in_serving"] == []

    def test_missing_feature_detected(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        serving = df.drop(columns=["feature_a"])
        result = check_feature_alignment(ref, serving)
        assert result["aligned"] is False
        assert "feature_a" in result["missing_in_serving"]

    def test_extra_feature_in_serving_ok(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        serving = df.copy()
        serving["extra_col"] = 1.0
        result = check_feature_alignment(ref, serving)
        assert result["aligned"] is True
        assert "extra_col" in result["extra_in_serving"]

    def test_result_has_expected_keys(self) -> None:
        df = _make_feature_df()
        ref = compute_reference_stats(df, model_version="v1")
        result = check_feature_alignment(ref, df)
        assert set(result.keys()) == {"aligned", "missing_in_serving", "extra_in_serving"}
