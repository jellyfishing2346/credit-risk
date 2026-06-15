"""
Singleton model store — loaded once at API startup, reused across all requests.

Holds the XGBoost model, fitted preprocessor, and feature names in memory so
every warm request avoids disk / MLflow I/O entirely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from credit_risk.api.schemas import (
    ScoreRequest,
    ScoreResponse,
    ShapDriver,
    ShapExplanation,
    classify_risk,
)
from credit_risk.explain.shap_explain import compute_shap_values, load_artifacts
from credit_risk.features.engineer import add_domain_features


class ModelStore:
    def __init__(self) -> None:
        self._model: xgb.XGBClassifier | None = None
        self._preprocessor = None
        self._feature_names: list[str] = []
        self._run_id: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def load(self, run_id: str | None = None) -> None:
        """Load model + preprocessor from MLflow (local dev path)."""
        print("Loading model and preprocessor from MLflow...")
        self._model, self._preprocessor, self._feature_names = load_artifacts(run_id)
        self._run_id = run_id or self._run_id
        print(f"Model ready — {len(self._feature_names)} features")

    def load_from_path(self, path: str = "model_artifacts") -> None:
        """Load baked model + preprocessor from a local directory (container path)."""
        import cloudpickle
        from pathlib import Path
        p = Path(path)
        print(f"Loading model from {p}/...")
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(p / "model.json"))
        with open(p / "preprocessor.pkl", "rb") as f:
            self._preprocessor = cloudpickle.load(f)
        self._feature_names = self._preprocessor.get_feature_names_out()
        run_id_file = p / "run_id.txt"
        self._run_id = run_id_file.read_text().strip() if run_id_file.exists() else "baked"
        print(f"Model ready — {len(self._feature_names)} features (run: {self._run_id})")

    def predict(self, request: ScoreRequest, top_n: int = 10) -> ScoreResponse:
        # Build a one-row DataFrame from the request
        raw = request.model_dump(exclude={"application_id"}, exclude_none=False)
        raw.pop("application_id", None)
        df = pd.DataFrame([raw])
        df.columns = [c.lower() for c in df.columns]

        # Add engineered features (EXT_SOURCE interactions, credit ratios)
        df = add_domain_features(df)

        # Transform — the preprocessor realigns columns internally
        X = self._preprocessor.transform(df)

        # Predict
        prob = float(self._model.predict_proba(X)[0, 1])

        # SHAP
        shap_row = compute_shap_values(self._model, X)[0]
        top_idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
        drivers = [
            ShapDriver(
                feature=self._feature_names[i],
                shap_value=round(float(shap_row[i]), 6),
                direction="INCREASES_RISK" if shap_row[i] > 0 else "DECREASES_RISK",
            )
            for i in top_idx
        ]

        return ScoreResponse(
            application_id=request.application_id,
            default_probability=round(prob, 6),
            risk_band=classify_risk(prob),
            model_run_id=self._run_id or "unknown",
            shap_explanation=ShapExplanation(top_drivers=drivers),
        )
