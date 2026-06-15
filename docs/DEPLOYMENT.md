# Deployment Guide

This document covers local development, Docker, and Google Cloud Run deployment. It also covers the model export workflow and how the container loads model artefacts at startup.

---

## How the container loads the model

At build time, `make export-model` exports the trained XGBoost model and fitted preprocessor from MLflow into `model_artifacts/`:

```
model_artifacts/
├── model.json          # XGBoost model in portable JSON format
├── preprocessor.pkl    # Fitted CreditRiskPreprocessor (cloudpickle)
└── run_id.txt          # MLflow run ID for traceability
```

These files are baked into the container image via `COPY model_artifacts/ /var/task/model_artifacts/`. At startup, the FastAPI lifespan context manager detects the `model_artifacts/model.json` path and loads directly from disk — no MLflow, no network calls.

```python
# src/credit_risk/api/main.py
@asynccontextmanager
async def lifespan(app):
    if Path("model_artifacts/model.json").exists():
        _store.load_from_path("model_artifacts")   # container path
    else:
        _store.load()                               # local dev via MLflow
    yield
```

This means warm requests after the first are purely in-memory — no disk I/O per request.

---

## Local development

### Run with uvicorn (recommended for development)

```bash
make serve
```

The API starts on `http://localhost:8000` with hot reload. This path loads the model from MLflow (`mlruns.db`), so you need a trained run first:

```bash
make train    # or make train-tune
make serve
```

### Run with Docker (matches production)

```bash
make docker-build    # export model + build image
make docker-run      # start on :8000 via docker-compose
```

`docker-compose.yml` overrides the Lambda entrypoint and CMD to use uvicorn:

```yaml
entrypoint: []
command: ["python", "-m", "uvicorn", "credit_risk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This is functionally identical to the Cloud Run deployment.

---

## Google Cloud Run

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) — `brew install --cask google-cloud-sdk`
- A GCP project with billing enabled
- A trained model — `make train`

### First-time setup

```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable artifactregistry.googleapis.com run.googleapis.com \
  --project YOUR_PROJECT_ID

# Create Artifact Registry repository (one-time)
make gar-create
```

### Deploy

```bash
make cloudrun-deploy
```

This runs the following steps in order:

1. `make export-model` — exports the latest MLflow run to `model_artifacts/`
2. `make docker-build-lambda` — builds a `linux/amd64` image using the Lambda base image
3. `gcloud auth configure-docker` — configures Docker to push to Artifact Registry
4. `docker push` — pushes the image to `europe-west1-docker.pkg.dev/PROJECT/credit-risk/api:latest`
5. `gcloud run deploy` — creates or updates the Cloud Run service

The deploy command overrides the Lambda entrypoint to run uvicorn:

```bash
--command=python
--args=-m,uvicorn,credit_risk.api.main:app,--host,0.0.0.0,--port,8000
```

At the end of the deploy, gcloud prints the service URL.

### Redeploy after retraining

```bash
make train               # retrain
make cloudrun-redeploy   # rebuild image + push + update service
```

`cloudrun-redeploy` skips creating the service (it already exists) and only updates the container image.

### Configuration

All Cloud Run configuration is in the Makefile under `## Google Cloud Run deployment`:

| Variable | Default | Description |
|---|---|---|
| `GCR_REGION` | `europe-west1` | Cloud Run and Artifact Registry region |
| `GCP_PROJECT` | `solar-panel-469619` | GCP project ID |
| `CLOUDRUN_NAME` | `credit-risk-scorer` | Cloud Run service name |

Override at the command line:

```bash
make cloudrun-deploy GCR_REGION=us-central1 GCP_PROJECT=my-project
```

### Memory and scaling

The current configuration:

```
--memory 512Mi       # 149MB measured usage, 512MB gives 3.4× headroom
--min-instances 0    # scale to zero when idle (cold starts possible)
--max-instances 5    # cap concurrent instances
--timeout 30         # 30s request timeout
```

To eliminate cold starts (first request after ~15 min idle takes 8–15s):

```bash
gcloud run services update credit-risk-scorer \
  --min-instances 1 \
  --region europe-west1 \
  --project YOUR_PROJECT_ID
```

This keeps one instance always warm at a cost of approximately $10–15/month.

---

## AWS Lambda (alternative)

The Dockerfile is built on `public.ecr.aws/lambda/python:3.13` and includes a Mangum adapter in `src/credit_risk/api/lambda_handler.py`. This means the same image can be deployed to Lambda without modification.

### Deploy to Lambda

```bash
# Push to ECR
make ecr-create
make ecr-push

# Create Lambda function
make lambda-create

# Expose via Function URL
make lambda-url
```

The Lambda handler is `credit_risk.api.lambda_handler.handler`. Mangum handles the translation between API Gateway / Function URL event format and the FastAPI ASGI interface.

### Lambda vs Cloud Run

| | AWS Lambda | Google Cloud Run |
|---|---|---|
| Cold start | 8–15s | 8–15s |
| Warm latency | ~26ms | ~32ms |
| Min memory | 128MB (set to 512MB) | 128MB (set to 512MB) |
| Free tier | 1M req/month | 2M req/month |
| Entrypoint | Mangum handler | uvicorn (CMD override) |
| Auth | Function URL IAM | Cloud Run IAM |

---

## Makefile reference (deployment targets)

| Target | Description |
|---|---|
| `make export-model` | Export latest MLflow run to `model_artifacts/` |
| `make docker-build` | Build native-arch image for local testing |
| `make docker-build-lambda` | Build `linux/amd64` image for Lambda/Cloud Run |
| `make docker-run` | Run image locally via docker-compose on :8000 |
| `make gar-create` | Create Artifact Registry repo (one-time) |
| `make cloudrun-deploy` | Full first-time deploy to Cloud Run |
| `make cloudrun-redeploy` | Update existing Cloud Run service |
| `make ecr-create` | Create ECR repo (one-time, AWS) |
| `make ecr-push` | Push image to ECR |
| `make lambda-create` | Create Lambda function |
| `make lambda-url` | Create Lambda Function URL |
| `make lambda-update` | Update Lambda with new image |
| `make deploy` | AWS full deploy (build + push + create + url) |
| `make redeploy` | AWS redeploy (build + push + update) |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'mlflow'` in container

The container does not include mlflow — it loads the model from baked artefacts. This error means `shap_explain.py` or another module is importing mlflow at the top level instead of lazily. All mlflow imports must be inside functions:

```python
def load_artifacts(...):
    import mlflow        # lazy — only runs when called
    import mlflow.xgboost
    ...
```

### `entrypoint requires the handler name to be the first argument`

The Lambda base image's entrypoint (`/lambda-entrypoint.sh`) intercepted the CMD. For local docker-compose, ensure the entrypoint is cleared:

```yaml
entrypoint: []
command: ["python", "-m", "uvicorn", ...]
```

### Container exits immediately with code 1

Check logs with `docker compose logs`. Most common causes:
- Missing `model_artifacts/` — run `make export-model` first
- Port mismatch — `--port 8000` in the uvicorn command must match the exposed port

### Cloud Run deploy: `Billing account not found`

The GCP project does not have billing enabled. Go to `console.cloud.google.com/billing/projects` and link a billing account, or use a project that already has billing enabled.

### `gcloud run deploy --args: expected one argument`

Use `=` syntax without quotes:

```bash
--args=-m,uvicorn,credit_risk.api.main:app,--host,0.0.0.0,--port,8000
```

Not:

```bash
--args "-m,uvicorn,..."   # wrong — gcloud treats this as two arguments
```
