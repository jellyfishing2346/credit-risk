import numpy as np
import pytest

from credit_risk.data.pipeline import (
    DAYS_EMPLOYED_SENTINEL,
    CreditRiskPreprocessor,
    HighMissingnessDropper,
    SentinelReplacer,
)


# ---------------------------------------------------------------------------
# SentinelReplacer
# ---------------------------------------------------------------------------

def test_sentinel_replaced_with_nan(sample_app_df):
    result = SentinelReplacer().fit_transform(sample_app_df)
    sentinel_mask = sample_app_df["days_employed"] == DAYS_EMPLOYED_SENTINEL
    assert sentinel_mask.any(), "fixture must contain at least one sentinel value"
    assert result.loc[sentinel_mask, "days_employed"].isna().all()


def test_non_sentinel_values_unchanged(sample_app_df):
    result = SentinelReplacer().fit_transform(sample_app_df)
    normal_mask = sample_app_df["days_employed"] != DAYS_EMPLOYED_SENTINEL
    assert (
        result.loc[normal_mask, "days_employed"].values
        == sample_app_df.loc[normal_mask, "days_employed"].values
    ).all()


def test_anomaly_flag_added(sample_app_df):
    result = SentinelReplacer().fit_transform(sample_app_df)
    assert "days_employed_anom" in result.columns


def test_anomaly_flag_matches_sentinel_rows(sample_app_df):
    sentinel_count = (sample_app_df["days_employed"] == DAYS_EMPLOYED_SENTINEL).sum()
    result = SentinelReplacer().fit_transform(sample_app_df)
    assert result["days_employed_anom"].sum() == sentinel_count


def test_sentinel_replacer_is_idempotent_on_fit(sample_app_df):
    replacer = SentinelReplacer()
    out1 = replacer.fit_transform(sample_app_df)
    out2 = replacer.fit_transform(sample_app_df)
    assert out1.equals(out2)


# ---------------------------------------------------------------------------
# HighMissingnessDropper
# ---------------------------------------------------------------------------

def test_high_missing_col_dropped(sample_app_df):
    dropper = HighMissingnessDropper(threshold=0.6)
    result = dropper.fit_transform(sample_app_df)
    assert "high_missing_col" not in result.columns


def test_low_missing_cols_kept(sample_app_df):
    dropper = HighMissingnessDropper(threshold=0.6)
    result = dropper.fit_transform(sample_app_df)
    for col in ["amt_income_total", "amt_credit", "ext_source_2"]:
        assert col in result.columns


def test_dropper_threshold_respected(sample_app_df):
    # With threshold=1.0 nothing should be dropped
    dropper = HighMissingnessDropper(threshold=1.0)
    result = dropper.fit_transform(sample_app_df)
    assert set(result.columns) == set(sample_app_df.columns)


def test_dropper_stores_dropped_cols(sample_app_df):
    dropper = HighMissingnessDropper(threshold=0.6)
    dropper.fit(sample_app_df)
    assert "high_missing_col" in dropper.cols_to_drop_


def test_dropper_transform_uses_fit_state(sample_app_df):
    dropper = HighMissingnessDropper(threshold=0.6)
    dropper.fit(sample_app_df)
    # A new DataFrame without the missing column should still not have it after transform
    df2 = sample_app_df.copy()
    df2["high_missing_col"] = 0.5  # fill it — but dropper was fitted to drop it
    result = dropper.transform(df2)
    assert "high_missing_col" not in result.columns


# ---------------------------------------------------------------------------
# CreditRiskPreprocessor
# ---------------------------------------------------------------------------

def test_preprocessor_drops_id_and_target(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    pre = CreditRiskPreprocessor()
    pre.fit_transform(X)
    feature_names = pre.get_feature_names_out()
    assert "sk_id_curr" not in feature_names
    assert "target" not in feature_names


def test_preprocessor_output_is_all_numeric(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    result = CreditRiskPreprocessor().fit_transform(X)
    assert np.issubdtype(result.dtype, np.floating)


def test_preprocessor_no_nans_in_output(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    result = CreditRiskPreprocessor().fit_transform(X)
    assert not np.isnan(result).any()


def test_preprocessor_feature_names_match_output_columns(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    pre = CreditRiskPreprocessor()
    result = pre.fit_transform(X)
    assert len(pre.get_feature_names_out()) == result.shape[1]


def test_preprocessor_transform_matches_fit_transform(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    pre = CreditRiskPreprocessor()
    out_fit_transform = pre.fit_transform(X)
    out_transform = pre.transform(X)
    np.testing.assert_array_almost_equal(out_fit_transform, out_transform)
