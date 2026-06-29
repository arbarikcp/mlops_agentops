"""Tests for data/contracts/label_contract.py."""
from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest

from data.contracts.label_contract import (
    DEFAULT_LABEL_METADATA,
    LabelMetadata,
    check_correction_rate,
    check_label_arrival,
    check_single_policy_version,
    validate_label_batch,
)


# ── Fixture: valid label batch ────────────────────────────────────────────────

def _make_label_df(
    n: int = 50,
    all_confirmed: bool = True,
    policy_version: str = "v1.0",
    label_source: str = "core_banking",
    n_corrected: int = 0,
) -> pd.DataFrame:
    """Build a synthetic label batch."""
    import numpy as np
    rng = np.random.default_rng(0)
    obs_dates = pd.date_range("2005-01-01", periods=n, freq="D")
    # confirmed = outcome 91 days after observation
    if all_confirmed:
        out_dates = obs_dates + pd.Timedelta(days=91)
    else:
        # half confirmed, half not (only 10 days after)
        out_dates = [obs + pd.Timedelta(days=91 if i % 2 == 0 else 10)
                     for i, obs in enumerate(obs_dates)]
    is_corrected = [True] * n_corrected + [False] * (n - n_corrected)

    return pd.DataFrame({
        "applicant_id": range(1, n + 1),
        "label": rng.choice([0, 1], n).tolist(),
        "label_source": [label_source] * n,
        "policy_version": [policy_version] * n,
        "label_timestamp": ["2005-04-01T00:00:00"] * n,
        "observation_date": [str(d.date()) for d in obs_dates],
        "outcome_date": [str(pd.Timestamp(d).date()) for d in out_dates],
        "is_corrected": is_corrected,
    })


# ── LabelMetadata tests ───────────────────────────────────────────────────────

class TestLabelMetadata:
    def test_valid_construction(self) -> None:
        meta = LabelMetadata(
            label_source="core_banking",
            policy_version="v1.0",
            observation_window_days=180,
            outcome_delay_days=90,
        )
        assert meta.outcome_delay_days == 90

    def test_invalid_label_source_raises(self) -> None:
        with pytest.raises(ValueError, match="label_source must be"):
            LabelMetadata(
                label_source="unknown_system",
                policy_version="v1.0",
                observation_window_days=180,
                outcome_delay_days=90,
            )

    def test_outcome_delay_below_1_raises(self) -> None:
        with pytest.raises(ValueError, match="outcome_delay_days must be"):
            LabelMetadata(
                label_source="core_banking",
                policy_version="v1.0",
                observation_window_days=180,
                outcome_delay_days=0,
            )

    def test_invalid_confirmed_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="min_confirmed_fraction"):
            LabelMetadata(
                label_source="core_banking",
                policy_version="v1.0",
                observation_window_days=180,
                outcome_delay_days=90,
                min_confirmed_fraction=1.5,
            )

    def test_default_metadata_valid(self) -> None:
        assert DEFAULT_LABEL_METADATA.outcome_delay_days == 90
        assert DEFAULT_LABEL_METADATA.label_source == "core_banking"


# ── validate_label_batch tests ────────────────────────────────────────────────

class TestValidateLabelBatch:
    def test_valid_batch_passes(self) -> None:
        df = _make_label_df()
        result = validate_label_batch(df)
        assert len(result) == len(df)

    def test_invalid_label_value_fails(self) -> None:
        df = _make_label_df()
        df.loc[0, "label"] = 2  # must be 0 or 1
        with pytest.raises(pa.errors.SchemaErrors):
            validate_label_batch(df)

    def test_invalid_label_source_fails(self) -> None:
        df = _make_label_df(label_source="unknown")
        with pytest.raises(pa.errors.SchemaErrors):
            validate_label_batch(df)

    def test_null_label_fails(self) -> None:
        df = _make_label_df()
        df.loc[0, "label"] = None
        with pytest.raises(pa.errors.SchemaErrors):
            validate_label_batch(df)

    def test_null_applicant_id_fails(self) -> None:
        df = _make_label_df()
        df.loc[0, "applicant_id"] = None
        with pytest.raises(pa.errors.SchemaErrors):
            validate_label_batch(df)


# ── check_label_arrival tests ─────────────────────────────────────────────────

class TestCheckLabelArrival:
    def test_all_confirmed_sufficient(self) -> None:
        df = _make_label_df(all_confirmed=True)
        result = check_label_arrival(df, DEFAULT_LABEL_METADATA)
        assert result["sufficient_for_training"] is True
        assert result["pct_confirmed"] > 0.9

    def test_half_confirmed_insufficient(self) -> None:
        df = _make_label_df(n=100, all_confirmed=False)
        result = check_label_arrival(df, DEFAULT_LABEL_METADATA)
        # ~50% confirmed; below 90% threshold
        assert result["pct_confirmed"] < 0.9
        assert result["sufficient_for_training"] is False

    def test_result_has_expected_keys(self) -> None:
        df = _make_label_df()
        result = check_label_arrival(df, DEFAULT_LABEL_METADATA)
        required = {"n_total", "n_confirmed", "pct_confirmed", "sufficient_for_training",
                    "outcome_delay_days", "min_required_fraction"}
        assert required.issubset(result.keys())

    def test_n_total_matches_df_length(self) -> None:
        df = _make_label_df(n=30)
        result = check_label_arrival(df, DEFAULT_LABEL_METADATA)
        assert result["n_total"] == 30

    def test_pct_confirmed_in_range(self) -> None:
        df = _make_label_df(all_confirmed=True)
        result = check_label_arrival(df, DEFAULT_LABEL_METADATA)
        assert 0.0 <= result["pct_confirmed"] <= 1.0


# ── check_single_policy_version tests ────────────────────────────────────────

class TestCheckSinglePolicyVersion:
    def test_single_version_consistent(self) -> None:
        df = _make_label_df(policy_version="v1.0")
        result = check_single_policy_version(df)
        assert result["is_consistent"] is True
        assert result["policy_versions"] == ["v1.0"]

    def test_mixed_versions_inconsistent(self) -> None:
        df1 = _make_label_df(n=25, policy_version="v1.0")
        df2 = _make_label_df(n=25, policy_version="v2.0")
        df2["applicant_id"] = df2["applicant_id"] + 25
        mixed = pd.concat([df1, df2], ignore_index=True)
        result = check_single_policy_version(mixed)
        assert result["is_consistent"] is False
        assert len(result["policy_versions"]) == 2

    def test_missing_column_returns_inconsistent(self) -> None:
        df = _make_label_df().drop(columns=["policy_version"])
        result = check_single_policy_version(df)
        assert result["is_consistent"] is False


# ── check_correction_rate tests ───────────────────────────────────────────────

class TestCheckCorrectionRate:
    def test_no_corrections_no_flag(self) -> None:
        df = _make_label_df(n_corrected=0)
        result = check_correction_rate(df, max_rate=0.05)
        assert result["flag"] is False
        assert result["correction_rate"] == 0.0

    def test_high_correction_rate_flagged(self) -> None:
        df = _make_label_df(n=100, n_corrected=10)  # 10% > 5% threshold
        result = check_correction_rate(df, max_rate=0.05)
        assert result["flag"] is True

    def test_correction_rate_matches_expectation(self) -> None:
        df = _make_label_df(n=100, n_corrected=3)  # 3%
        result = check_correction_rate(df, max_rate=0.05)
        assert abs(result["correction_rate"] - 0.03) < 0.001

    def test_result_has_expected_keys(self) -> None:
        df = _make_label_df()
        result = check_correction_rate(df)
        assert set(result.keys()) == {"n_corrected", "n_total", "correction_rate", "flag"}
