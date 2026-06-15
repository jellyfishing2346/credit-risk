"""
Export the trained model + preprocessor from MLflow to model_artifacts/.

Run this before `make docker-build` to bake the model into the container image.
The container loads from these plain files at startup — no MLflow dependency at runtime.

Usage:
    uv run python scripts/export_model.py              # latest run
    uv run python scripts/export_model.py --run-id <id>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cloudpickle
import mlflow
import mlflow.xgboost
import xgboost as xgb

TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "credit-risk-xgboost"
OUTPUT_DIR = Path("model_artifacts")


def export(run_id: str | None = None) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    if run_id is None:
        runs = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise RuntimeError("No runs found. Run 'make train' first.")
        run_id = runs.iloc[0]["run_id"]

    print(f"Exporting run: {run_id}")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # XGBoost model — save as JSON (portable, version-independent)
    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
    model_path = OUTPUT_DIR / "model.json"
    model.save_model(str(model_path))
    print(f"  model     → {model_path}")

    # Preprocessor — cloudpickle preserves lambda closures used in engineer.py
    preprocessor_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessor"
    )
    src_pkl = next(Path(preprocessor_dir).glob("*.pkl"))
    dest_pkl = OUTPUT_DIR / "preprocessor.pkl"
    shutil.copy(src_pkl, dest_pkl)
    print(f"  preprocessor → {dest_pkl}")

    # Record which run was baked in for traceability
    (OUTPUT_DIR / "run_id.txt").write_text(run_id)
    print(f"  run_id    → {OUTPUT_DIR}/run_id.txt")
    print(f"Done — model_artifacts/ ready for 'make docker-build'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    export(args.run_id)
