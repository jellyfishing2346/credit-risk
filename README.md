# Credit Risk Scoring Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud_Run-Deployed-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade credit risk scoring system built on the Home Credit Default Risk dataset.**
Score loan applications in real time, with per-decision SHAP explanations of every prediction.

[Live API](https://credit-risk-scorer-218846370015.europe-west1.run.app/docs) · [Dashboard](https://github.com/jellyfishing2346/credit-risk) · [GitHub](https://github.com/jellyfishing2346/credit-risk)

</div>

---

## Overview

Most ML tutorials stop at model training. This project goes further — from raw CSVs to a containerised REST API running in production, with full explainability on every prediction.

| Metric | Value |
|---|---|
| Training records | 307,511 |
| Source tables | 8 CSV files |
| Validation ROC-AUC | 0.7743 |
| Test ROC-AUC | **0.7795** |
| API warm latency | **< 35ms** |
| Default rate (class imbalance) | 8.1% |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Layer (Phase 1)                         │
│                                                                     │
│  Kaggle CSVs  ──►  PostgreSQL Schema  ──►  Leakage Checks           │
│  (8 tables)         (SQLAlchemy Core)       (pattern + correlation) │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    Feature Engineering (Phase 2)                    │
│                                                                     │
│  Bureau · Previous Apps · Installments · POS Cash · Credit Cards    │
│  ──► 12–9 aggregations each ──► Domain ratios + EXT_SOURCE          │
│                                  interactions ──► 176 features      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                      Model Training (Phase 2)                       │
│                                                                     │
│  XGBoost  +  Optuna HPO (30 trials)  +  MLflow tracking             │
│  Stratified 60/20/20 split  ·  scale_pos_weight for imbalance       │
│  Early stopping (50 rounds)  ·  SQLite artifact store               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                    Explainability (Phase 3)                         │
│                                                                     │
│  XGBoost native TreeSHAP (pred_contribs=True)                       │
│  Global importance  ·  Per-application top-N drivers                │
│  No shap/llvmlite dependency — same algorithm, zero overhead        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│                       REST API (Phase 4)                            │
│                                                                     │
│  FastAPI  ·  POST /score  ──►  probability + risk band + SHAP       │
│  GET /health  ·  Pydantic v2 validation  ·  <35ms warm latency      │
└───────────────┬─────────────────────────────┬───────────────────────┘
                │                             │
┌───────────────▼───────────┐   ┌────────────▼──────────────────────┐
│   Deployment (Phase 5)    │   │        Dashboard                  │
│                           │   │                                   │
│  Docker container image   │   │  Streamlit · probability gauge    │
│  Model baked in at build  │   │  SHAP waterfall chart             │
│  Google Cloud Run         │   │  Live API calls                   │
│  Mangum Lambda adapter    │   │                                   │
└───────────────────────────┘   └───────────────────────────────────┘
```

---

## Project Structure

```
credit-risk/
├── src/credit_risk/
│   ├── data/
│   │   ├── schema.py          # SQLAlchemy Core table definitions (7 tables)
│   │   ├── ingest.py          # Kaggle download + PostgreSQL ingest
│   │   ├── pipeline.py        # Cleaning pipeline (sentinel, missingness, encoding)
│   │   └── leakage.py         # Pattern + correlation leakage checks
│   ├── features/
│   │   └── engineer.py        # Bureau, prev apps, installments, POS, credit card aggs
│   ├── models/
│   │   └── train.py           # XGBoost training + Optuna HPO + MLflow logging
│   ├── explain/
│   │   └── shap_explain.py    # Native TreeSHAP, global + local explanations
│   └── api/
│       ├── main.py            # FastAPI app + lifespan model loading
│       ├── model_store.py     # Singleton store, MLflow + path-based loading
│       ├── schemas.py         # Pydantic v2 request/response models
│       └── lambda_handler.py  # Mangum ASGI adapter for Lambda
├── dashboard/
│   ├── app.py                 # Streamlit dashboard
│   └── requirements.txt
├── scripts/
│   └── export_model.py        # Export MLflow artifacts → model_artifacts/
├── tests/
│   ├── test_pipeline.py       # 15 pipeline unit tests
│   ├── models/test_train.py   # 5 training tests
│   ├── explain/test_shap_explain.py  # 9 SHAP tests
│   └── api/test_api.py        # 12 API tests
├── Dockerfile                 # Lambda container image (public.ecr.aws/lambda/python:3.13)
├── docker-compose.yml         # Local testing via uvicorn
├── pyproject.toml             # uv-managed dependencies
└── Makefile                   # All workflow commands
```

---

## Quickstart

### Prerequisites

- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Python 3.13
- Docker Desktop
- PostgreSQL (for data ingestion only)

### Install

```bash
git clone https://github.com/jellyfishing2346/credit-risk.git
cd credit-risk
make install-dev
```

### Download data

Create a Kaggle account, accept the [Home Credit competition rules](https://www.kaggle.com/competitions/home-credit-default-risk), then add your credentials to `~/.kaggle/kaggle.json`:

```json
{"username": "your_username", "key": "your_api_key"}
```

```bash
make download
```

### Run the full pipeline

```bash
make ingest          # load CSVs into PostgreSQL
make leakage-check   # verify no data leakage
make train           # train XGBoost, log to MLflow
make explain         # global SHAP feature importance report
make serve           # start API locally on :8000
make dashboard       # start Streamlit dashboard on :8502
```

---

## Model Details

### Feature Engineering

Five auxiliary tables are aggregated and joined to the main applications table:

| Table | Features built | Examples |
|---|---|---|
| `bureau` | 13 | active credit ratio, max overdue days |
| `previous_applications` | 10 | approval rate, avg credit |
| `installments_payments` | 8 | payment ratio, days late |
| `pos_cash_balance` | 7 | DPD, completed ratio |
| `credit_card_balance` | 8 | utilisation, DPD |

**Domain features** added on top:
- `ext_source_prod23` — product of EXT_SOURCE_2 × EXT_SOURCE_3 (top global feature)
- `credit_income_ratio`, `annuity_income_ratio`, `credit_term`, `goods_credit_ratio`
- `employed_to_age_ratio`

Final feature matrix: **176 columns**, 307,511 rows.

### Preprocessing Pipeline

```
SentinelReplacer          — DAYS_EMPLOYED=365243 → NaN + anomaly flag
HighMissingnessDropper    — drop columns with >60% missing
ColumnTransformer
  ├── numeric   → median imputation
  ├── binary    → mode imputation + ordinal encoding
  └── categorical → Unknown fill + one-hot (max_categories=20)
```

### Training

```
Algorithm       XGBoost (gradient boosted trees)
Split           60% train / 20% val / 20% test  (stratified)
Imbalance       scale_pos_weight = (1 - 0.081) / 0.081 ≈ 11.3
Early stopping  50 rounds on validation AUC
HPO             Optuna, 30 trials, TPE sampler
Tracking        MLflow with SQLite backend (mlruns.db)
```

### Results

| Split | ROC-AUC | Gini | Avg Precision |
|---|---|---|---|
| Validation | 0.7743 | — | — |
| Test | 0.7795 | 0.5591 | 0.2763 |

Top global features by mean |SHAP|:

```
ext_source_prod23          ████████████████████████████████
ext_source_2               ████████████████████████
ext_source_3               ████████████████████
days_birth                 ████████████████
credit_term                █████████████
goods_credit_ratio         ████████████
code_gender                ████████████
annuity_income_ratio       ██████████
```

---

## API Reference

**Base URL:** `https://credit-risk-scorer-218846370015.europe-west1.run.app`

Interactive docs: `/docs`

### GET /health

```json
{
  "status": "ok",
  "model_loaded": true,
  "run_id": "91a4f51bd49c46b397134a77d9482dfa"
}
```

### POST /score

All fields are optional — the model handles missing values through median imputation. Pass whatever fields are available.

**Request**
```json
{
  "application_id": "APP-001",
  "amt_credit": 250000,
  "amt_income_total": 90000,
  "amt_annuity": 20000,
  "amt_goods_price": 230000,
  "days_birth": -12775,
  "ext_source_2": 0.62,
  "ext_source_3": 0.55,
  "code_gender": "M",
  "name_contract_type": "Cash loans"
}
```

**Response**
```json
{
  "application_id": "APP-001",
  "default_probability": 0.401652,
  "risk_band": "HIGH",
  "model_run_id": "91a4f51bd49c46b397134a77d9482dfa",
  "latency_ms": 30.98,
  "shap_explanation": {
    "top_drivers": [
      {
        "feature": "num__ext_source_2",
        "shap_value": -0.167671,
        "direction": "DECREASES_RISK"
      }
    ]
  }
}
```

**Risk bands**

| Band | Default probability |
|---|---|
| LOW | < 5% |
| MEDIUM | 5% – 15% |
| HIGH | ≥ 15% |

---

## Deployment

### Local Docker

```bash
make docker-build    # export model + build image (native arch)
make docker-run      # run on :8000 via uvicorn
```

### Google Cloud Run

```bash
make docker-build-lambda   # cross-compile linux/amd64
make gar-create            # create Artifact Registry repo (one-time)
make cloudrun-deploy       # push image + deploy service
```

### Redeploy after retraining

```bash
make train
make cloudrun-redeploy
```

### Environment variables

No environment variables are required at runtime. The model artifacts are baked into the container image at build time via `make export-model`.

---

## Testing

```bash
make test
```

```
41 passed in 21s

tests/test_pipeline.py          15 tests  — sentinel, dropper, preprocessor
tests/models/test_train.py       5 tests  — stratified split, no leakage
tests/explain/test_shap_explain.py  9 tests  — SHAP shape, sum, sorting
tests/api/test_api.py           12 tests  — endpoints, schema, 503 handling
```

---

## Makefile Reference

| Command | Description |
|---|---|
| `make install` | Install base dependencies |
| `make install-dev` | Install all dependencies including dev |
| `make test` | Run full test suite with coverage |
| `make lint` | Ruff lint check |
| `make format` | Ruff auto-format |
| `make download` | Download dataset from Kaggle |
| `make ingest` | Load CSVs into PostgreSQL |
| `make leakage-check` | Run leakage detection report |
| `make train` | Train XGBoost baseline |
| `make train-tune` | Train with Optuna HPO (30 trials) |
| `make mlflow-ui` | Open MLflow experiment tracker |
| `make explain` | Print global SHAP importance chart |
| `make serve` | Run API locally on :8000 |
| `make dashboard` | Run Streamlit dashboard on :8502 |
| `make export-model` | Export MLflow artifacts to model_artifacts/ |
| `make docker-build` | Build container image (native arch) |
| `make docker-build-lambda` | Build container image (linux/amd64) |
| `make docker-run` | Run container locally via docker-compose |
| `make gar-create` | Create Google Artifact Registry repo |
| `make cloudrun-deploy` | Full deploy to Google Cloud Run |
| `make cloudrun-redeploy` | Push updated image to existing service |

---

## Development Guide

### Adding new features

Feature engineering lives in [src/credit_risk/features/engineer.py](src/credit_risk/features/engineer.py). Each auxiliary table has its own `build_*_features()` function returning a DataFrame indexed on `sk_id_curr`. Add your aggregations there, then call `build_full_feature_matrix()` to verify the shape before retraining.

### Retraining

```bash
make train-tune      # Optuna HPO — takes ~20 min
make mlflow-ui       # compare runs
make cloudrun-redeploy
```

### Changing the risk bands

Risk band thresholds are defined in [src/credit_risk/api/schemas.py](src/credit_risk/api/schemas.py) in the `classify_risk()` function. Adjust the thresholds and redeploy — no model retraining needed.

### Running without PostgreSQL

The training pipeline (`make train`) loads directly from CSVs in `data/raw/`. PostgreSQL is only needed for `make ingest`. If you skip ingestion, training still works.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Package manager | uv |
| Data | pandas, numpy, SQLAlchemy |
| Database | PostgreSQL |
| ML | XGBoost, scikit-learn |
| HPO | Optuna |
| Experiment tracking | MLflow (SQLite backend) |
| Explainability | XGBoost native TreeSHAP |
| API | FastAPI, Pydantic v2, uvicorn |
| Serialization | cloudpickle |
| Container | Docker, Mangum |
| Cloud | Google Cloud Run, Artifact Registry |
| Dashboard | Streamlit, Plotly |
| Testing | pytest, pytest-cov |
| Linting | Ruff |

---

## License

[MIT](LICENSE) © 2026 Faizan Khan
