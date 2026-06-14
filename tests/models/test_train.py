import numpy as np
import pytest

from credit_risk.data.pipeline import CreditRiskPreprocessor
from credit_risk.models.train import split_data


def test_split_is_stratified(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    y = sample_app_df["target"]
    _, _, _, y_train, y_val, y_test = split_data(X, y)

    base_rate = y.mean()
    for split_y in [y_train, y_val, y_test]:
        assert abs(split_y.mean() - base_rate) < 0.08


def test_split_sizes(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    y = sample_app_df["target"]
    X_train, X_val, X_test, *_ = split_data(X, y, val_size=0.2, test_size=0.2)

    total = len(sample_app_df)
    assert abs(len(X_test) / total - 0.2) < 0.03
    assert abs(len(X_val) / total - 0.2) < 0.03
    assert abs(len(X_train) / total - 0.6) < 0.03


def test_split_no_overlap(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    y = sample_app_df["target"]
    X_train, X_val, X_test, *_ = split_data(X, y)

    train_idx = set(X_train.index)
    val_idx = set(X_val.index)
    test_idx = set(X_test.index)
    assert not (train_idx & val_idx)
    assert not (train_idx & test_idx)
    assert not (val_idx & test_idx)


def test_split_covers_all_rows(sample_app_df):
    X = sample_app_df.drop(columns=["target"])
    y = sample_app_df["target"]
    X_train, X_val, X_test, *_ = split_data(X, y)

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(sample_app_df)


def test_preprocessor_fit_on_train_only(sample_app_df):
    """Preprocessor must be fit on train only — fitting on val/test would leak distribution info."""
    X = sample_app_df.drop(columns=["target"])
    y = sample_app_df["target"]
    X_train, X_val, X_test, *_ = split_data(X, y)

    pre = CreditRiskPreprocessor()
    X_tr = pre.fit_transform(X_train)
    X_v = pre.transform(X_val)
    X_te = pre.transform(X_test)

    assert X_tr.shape[1] == X_v.shape[1] == X_te.shape[1]
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_v).any()
    assert not np.isnan(X_te).any()
