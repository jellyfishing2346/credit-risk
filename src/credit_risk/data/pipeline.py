"""
Cleaning and encoding pipeline for the Home Credit applications table.

Usage:
    from credit_risk.data.pipeline import CreditRiskPreprocessor

    pre = CreditRiskPreprocessor()
    X_train = pre.fit_transform(train_df.drop(columns=["target"]))
    X_test  = pre.transform(test_df.drop(columns=["target"]))
    feature_names = pre.get_feature_names_out()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# DAYS_EMPLOYED uses 365243 as a sentinel for "applicant has never been employed"
DAYS_EMPLOYED_SENTINEL = 365243

# Columns that identify a row rather than describe an applicant — always dropped before modelling
_ID_COLS = {"sk_id_curr", "target"}

# Binary categoricals — label-encoded as 0/1 rather than one-hot to avoid redundant columns
_BINARY_CATS = ["code_gender", "flag_own_car", "flag_own_realty", "emergencystate_mode"]

# Low-cardinality nominals — one-hot encoded (bounded column explosion)
_LOW_CARD_CATS = [
    "name_contract_type",
    "name_income_type",
    "name_education_type",
    "name_family_status",
    "name_housing_type",
    "weekday_appr_process_start",
    "fondkapremont_mode",
    "housetype_mode",
    "wallsmaterial_mode",
]

# High-cardinality nominals — one-hot with a max_categories cap to control dimensionality
_HIGH_CARD_CATS = ["organization_type", "occupation_type", "name_type_suite"]


class SentinelReplacer(BaseEstimator, TransformerMixin):
    """Replace the DAYS_EMPLOYED sentinel with NaN and add a binary anomaly-indicator column."""

    def fit(self, X: pd.DataFrame, y=None) -> "SentinelReplacer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "days_employed" in X.columns:
            X["days_employed_anom"] = (X["days_employed"] == DAYS_EMPLOYED_SENTINEL).astype(np.int8)
            X["days_employed"] = X["days_employed"].replace(DAYS_EMPLOYED_SENTINEL, np.nan)
        return X


class HighMissingnessDropper(BaseEstimator, TransformerMixin):
    """Drop columns where the fraction of missing values exceeds `threshold` (default 60%)."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "HighMissingnessDropper":
        missing_frac = X.isnull().mean()
        self.cols_to_drop_: list[str] = missing_frac[missing_frac > self.threshold].index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.cols_to_drop_, errors="ignore")


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Full preprocessing pipeline for the Home Credit applications feature matrix.

    Steps (in order):
      1. Drop ID / label columns
      2. Replace DAYS_EMPLOYED sentinel and add anomaly flag
      3. Drop columns with >60% missing values
      4. ColumnTransformer:
           - numeric      → median imputation
           - binary cats  → mode imputation + ordinal encoding
           - other cats   → 'Unknown' fill + one-hot (max_categories=20 for high-card)
    """

    def __init__(self, missing_threshold: float = 0.6) -> None:
        self.missing_threshold = missing_threshold

    def fit(self, X: pd.DataFrame, y=None) -> "CreditRiskPreprocessor":
        X = X.drop(columns=list(_ID_COLS & set(X.columns)), errors="ignore")

        self._sentinel = SentinelReplacer()
        X = self._sentinel.fit_transform(X)

        self._dropper = HighMissingnessDropper(threshold=self.missing_threshold)
        X = self._dropper.fit_transform(X)

        bin_cols = [c for c in _BINARY_CATS if c in X.columns]
        cat_cols = [
            c
            for c in X.select_dtypes(include="object").columns
            if c not in set(_BINARY_CATS)
        ]
        num_cols = [
            c
            for c in X.select_dtypes(include="number").columns
            if c not in set(_BINARY_CATS)
        ]

        self._ct = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                (
                    "bin",
                    Pipeline([
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("enc", OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        )),
                    ]),
                    bin_cols,
                ),
                (
                    "cat",
                    Pipeline([
                        ("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("enc", OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            max_categories=20,
                        )),
                    ]),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )
        self._ct.fit(X)
        # Store the column order seen at fit time so inference-time DataFrames
        # (which may be missing auxiliary columns) can be realigned before transform.
        self._fitted_columns_: list[str] = X.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.drop(columns=list(_ID_COLS & set(X.columns)), errors="ignore")
        X = self._sentinel.transform(X)
        X = self._dropper.transform(X)
        # Realign: add any columns the CT expects but are absent (filled with NaN),
        # drop any extras that appeared post-fit (e.g. columns added only in test data).
        X = X.reindex(columns=self._fitted_columns_, fill_value=np.nan)
        return self._ct.transform(X)

    def get_feature_names_out(self) -> list[str]:
        return self._ct.get_feature_names_out().tolist()
