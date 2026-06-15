"""
SHAP explainability using XGBoost's native TreeSHAP (pred_contribs=True).

No external shap library required — XGBoost ships the same TreeSHAP algorithm
natively in C++, which is faster and avoids the llvmlite/Python version constraint.

Two levels of explanation:
  - Global  : mean |SHAP| per feature across a dataset — shows what the model
              uses overall and is the primary artefact for model cards / audits.
  - Local   : per-application SHAP breakdown — "why did this applicant get this
              score?" — the key output for Phase 4's API /explain endpoint.

Usage:
    from credit_risk.explain.shap_explain import load_artifacts, global_importance, explain_application

    model, preprocessor, feature_names = load_artifacts(run_id="<mlflow-run-id>")
    # or omit run_id to use the latest run in the default experiment
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

_TRACKING_URI = "sqlite:///mlruns.db"
_EXPERIMENT_NAME = "credit-risk-xgboost"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_shap_values(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    """
    Return SHAP values with shape (n_samples, n_features).

    XGBoost's pred_contribs returns (n_samples, n_features + 1); the final
    column is the bias term (base score), which we strip here so callers
    always get one value per named feature.
    """
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X)
    contribs = booster.predict(dmatrix, pred_contribs=True)
    return contribs[:, :-1]


def global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return a DataFrame of the top_n features ranked by mean |SHAP| value.

    This is the standard global feature importance plot for model cards and
    regulatory explanations.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.nlargest(top_n, "mean_abs_shap").reset_index(drop=True)


def explain_application(
    shap_row: np.ndarray,
    feature_names: list[str],
    feature_values: np.ndarray,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the top_n features driving a single application's score.

    Positive SHAP → pushes score toward default (increases risk).
    Negative SHAP → pushes score toward repayment (decreases risk).

    Args:
        shap_row      : 1-D array of SHAP values for one application (n_features,)
        feature_names : output of preprocessor.get_feature_names_out()
        feature_values: preprocessed feature vector for the same application
        top_n         : how many top contributors to return
    """
    df = pd.DataFrame({
        "feature": feature_names,
        "preprocessed_value": feature_values,
        "shap_value": shap_row,
    })
    df["abs_shap"] = df["shap_value"].abs()
    return (
        df.nlargest(top_n, "abs_shap")
        .drop(columns=["abs_shap"])
        .reset_index(drop=True)
    )


def load_artifacts(
    run_id: str | None = None,
    experiment_name: str = _EXPERIMENT_NAME,
) -> tuple[xgb.XGBClassifier, object, list[str]]:
    """
    Load model + preprocessor from MLflow and return (model, preprocessor, feature_names).

    If run_id is None, the most recent run in the experiment is used.
    mlflow is imported lazily so this module can be imported in the container
    (which has no mlflow) without failing at import time.
    """
    import cloudpickle
    import mlflow
    import mlflow.xgboost

    mlflow.set_tracking_uri(_TRACKING_URI)

    if run_id is None:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError(
                f"No runs found in experiment '{experiment_name}'. Run 'make train' first."
            )
        run_id = runs.iloc[0]["run_id"]
        print(f"Using latest run: {run_id}")

    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")

    preprocessor_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessor"
    )
    pkl_path = next(Path(preprocessor_dir).glob("*.pkl"))
    with open(pkl_path, "rb") as f:
        preprocessor = cloudpickle.load(f)

    feature_names = preprocessor.get_feature_names_out()
    return model, preprocessor, feature_names


def run_global_report(run_id: str | None = None, top_n: int = 20) -> pd.DataFrame:
    """
    Load the latest trained model and print global feature importance.
    Returns the importance DataFrame for further use.
    """
    from credit_risk.features.engineer import build_full_feature_matrix
    from credit_risk.models.train import split_data

    model, preprocessor, feature_names = load_artifacts(run_id)

    print("Loading data for SHAP background sample...")
    X, y = build_full_feature_matrix()
    _, _, X_test, _, _, _ = split_data(X, y)

    print("Preprocessing test set...")
    X_te = preprocessor.transform(X_test)

    print(f"Computing SHAP values for {len(X_te):,} test samples (may take a minute)...")
    shap_vals = compute_shap_values(model, X_te)

    importance = global_importance(shap_vals, feature_names, top_n=top_n)

    print(f"\n{'─' * 52}")
    print(f"  Global feature importance — top {top_n}")
    print(f"{'─' * 52}")
    for _, row in importance.iterrows():
        bar = "█" * int(row["mean_abs_shap"] / importance["mean_abs_shap"].max() * 30)
        print(f"  {row['feature']:<35} {bar}")
    print(f"{'─' * 52}")

    return importance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="MLflow run ID (default: latest)")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()
    run_global_report(run_id=args.run_id, top_n=args.top_n)
