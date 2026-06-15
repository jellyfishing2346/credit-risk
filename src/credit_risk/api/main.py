"""
Credit Risk Scoring API

Endpoints:
  GET  /health   — liveness + model-loaded status
  POST /score    — score a loan application, returns probability + SHAP explanation

Start locally:
  make serve
  # or: uv run uvicorn credit_risk.api.main:app --reload --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from credit_risk.api.model_store import ModelStore
from credit_risk.api.schemas import HealthResponse, ScoreRequest, ScoreResponse

_store = ModelStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from pathlib import Path
    if Path("model_artifacts/model.json").exists():
        _store.load_from_path("model_artifacts")
    else:
        _store.load()
    yield


app = FastAPI(
    title="Credit Risk Scoring API",
    description=(
        "Score loan applications using an XGBoost model trained on Home Credit data. "
        "Every score includes a SHAP explanation of the top drivers."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_store.is_loaded,
        run_id=_store.run_id,
    )


@app.post("/score", response_model=ScoreResponse, tags=["scoring"])
def score(request: ScoreRequest) -> ScoreResponse:
    if not _store.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    result = _store.predict(request)
    result.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return result
