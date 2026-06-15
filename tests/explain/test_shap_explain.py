"""
Tests for SHAP explainability functions.

Uses a tiny synthetic XGBoost model — no real data or MLflow needed.
"""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from credit_risk.explain.shap_explain import (
    compute_shap_values,
    explain_application,
    global_importance,
)


@pytest.fixture(scope="module")
def tiny_model_and_data():
    """Train a tiny XGBoost model on random binary classification data."""
    rng = np.random.default_rng(0)
    n, p = 200, 10
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (rng.random(n) < 0.2).astype(int)

    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        objective="binary:logistic",
        random_state=0,
        eval_metric="auc",
    )
    model.fit(X, y)
    return model, X, y


@pytest.fixture(scope="module")
def shap_vals(tiny_model_and_data):
    model, X, _ = tiny_model_and_data
    return compute_shap_values(model, X)


# ---------------------------------------------------------------------------
# compute_shap_values
# ---------------------------------------------------------------------------

def test_shap_shape(tiny_model_and_data, shap_vals):
    _, X, _ = tiny_model_and_data
    assert shap_vals.shape == X.shape


def test_shap_values_are_finite(shap_vals):
    assert np.isfinite(shap_vals).all()


def test_shap_sum_approximates_logodds(tiny_model_and_data, shap_vals):
    """
    SHAP values (contribs) + bias should sum to the raw log-odds prediction.
    We verify this by comparing sigmoid(sum) to predict_proba.
    """
    model, X, _ = tiny_model_and_data
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X)
    # Full contribs including bias column
    contribs = booster.predict(dmatrix, pred_contribs=True)
    logodds = contribs.sum(axis=1)
    proba_from_shap = 1.0 / (1.0 + np.exp(-logodds))
    proba_from_model = model.predict_proba(X)[:, 1]
    np.testing.assert_allclose(proba_from_shap, proba_from_model, atol=1e-5)


# ---------------------------------------------------------------------------
# global_importance
# ---------------------------------------------------------------------------

def test_global_importance_shape(shap_vals):
    names = [f"feature_{i}" for i in range(shap_vals.shape[1])]
    imp = global_importance(shap_vals, names, top_n=5)
    assert len(imp) == 5
    assert set(imp.columns) == {"feature", "mean_abs_shap"}


def test_global_importance_sorted(shap_vals):
    names = [f"feature_{i}" for i in range(shap_vals.shape[1])]
    imp = global_importance(shap_vals, names)
    assert imp["mean_abs_shap"].is_monotonic_decreasing


def test_global_importance_non_negative(shap_vals):
    names = [f"feature_{i}" for i in range(shap_vals.shape[1])]
    imp = global_importance(shap_vals, names)
    assert (imp["mean_abs_shap"] >= 0).all()


# ---------------------------------------------------------------------------
# explain_application
# ---------------------------------------------------------------------------

def test_explain_application_shape(shap_vals, tiny_model_and_data):
    _, X, _ = tiny_model_and_data
    names = [f"feature_{i}" for i in range(X.shape[1])]
    result = explain_application(shap_vals[0], names, X[0], top_n=5)
    assert len(result) == 5
    assert set(result.columns) == {"feature", "preprocessed_value", "shap_value"}


def test_explain_application_sorted_by_abs(shap_vals, tiny_model_and_data):
    _, X, _ = tiny_model_and_data
    names = [f"feature_{i}" for i in range(X.shape[1])]
    result = explain_application(shap_vals[0], names, X[0])
    assert result["shap_value"].abs().is_monotonic_decreasing


def test_explain_application_top_n_respected(shap_vals, tiny_model_and_data):
    _, X, _ = tiny_model_and_data
    names = [f"feature_{i}" for i in range(X.shape[1])]
    for top_n in [1, 3, 7]:
        result = explain_application(shap_vals[0], names, X[0], top_n=top_n)
        assert len(result) == top_n
