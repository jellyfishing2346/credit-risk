"""
XGBoost training pipeline with MLflow experiment tracking.

Usage (from project root):
    uv run python -m credit_risk.models.train               # baseline run
    uv run python -m credit_risk.models.train --tune        # + Optuna HPO (30 trials)
    uv run python -m credit_risk.models.train --tune --trials 50
"""
from __future__ import annotations

import argparse

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk.data.pipeline import CreditRiskPreprocessor
from credit_risk.features.engineer import build_full_feature_matrix

EXPERIMENT_NAME = "credit-risk-xgboost"
RANDOM_STATE = 42

# Baseline hyperparameters — solid starting point for Home Credit
# scale_pos_weight is set dynamically from training set class balance
_BASE_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
}


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 60/20/20 split — preserves the ~8% default rate in all three sets."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    val_frac_of_temp = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac_of_temp, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _fit_model(
    params: dict,
    X_tr: np.ndarray,
    X_v: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=100)
    return model


def _evaluate(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    y: pd.Series,
    prefix: str,
) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    return {
        f"{prefix}_auc": auc,
        f"{prefix}_gini": 2 * auc - 1,
        f"{prefix}_avg_precision": average_precision_score(y, proba),
    }


def _tune(
    base_params: dict,
    X_tr: np.ndarray,
    X_v: np.ndarray,
    y_train: pd.Series,
    y_val: pd.Series,
    n_trials: int = 30,
) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_params,
            "n_estimators": 500,
            "early_stopping_rounds": 30,
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_train, eval_set=[(X_v, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_v)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best val AUC from tuning: {study.best_value:.4f}")
    return {**base_params, **study.best_params}


def run_training(tune: bool = False, n_trials: int = 30) -> tuple[xgb.XGBClassifier, dict]:
    # MLflow 3.x requires a database backend; SQLite is the right local choice.
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Building feature matrix (application + bureau + previous_application)...")
    X, y = build_full_feature_matrix()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    print(
        f"Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}"
        f"  |  default rate: train {y_train.mean():.3%}  val {y_val.mean():.3%}  test {y_test.mean():.3%}"
    )

    params = {**_BASE_PARAMS, "scale_pos_weight": neg / pos}

    print("Preprocessing features...")
    preprocessor = CreditRiskPreprocessor()
    X_tr = preprocessor.fit_transform(X_train)
    X_v = preprocessor.transform(X_val)
    X_te = preprocessor.transform(X_test)

    if tune:
        print(f"Running Optuna HPO ({n_trials} trials)...")
        params = _tune(params, X_tr, X_v, y_train, y_val, n_trials)

    with mlflow.start_run() as run:
        log_params = {k: v for k, v in params.items() if k not in {"objective", "eval_metric"}}
        mlflow.log_params(log_params)
        mlflow.log_params({"n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)})

        print("\nTraining XGBoost...")
        model = _fit_model(params, X_tr, X_v, y_train, y_val)

        val_metrics = _evaluate(model, X_v, y_val, "val")
        test_metrics = _evaluate(model, X_te, y_test, "test")
        metrics = {**val_metrics, **test_metrics}
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")

        run_id = run.info.run_id

    print(f"\n{'─' * 40}")
    print(f"  val  ROC-AUC       {val_metrics['val_auc']:.4f}")
    print(f"  test ROC-AUC       {test_metrics['test_auc']:.4f}   (benchmark: 0.79)")
    print(f"  test Gini          {test_metrics['test_gini']:.4f}")
    print(f"  test Avg Precision {test_metrics['test_avg_precision']:.4f}")
    print(f"{'─' * 40}")
    print(f"MLflow run: {run_id}")
    print("View experiments: make mlflow-ui")

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials")
    args = parser.parse_args()
    run_training(tune=args.tune, n_trials=args.trials)
