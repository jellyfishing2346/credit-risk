.PHONY: install install-dev lint format test setup-db download ingest leakage-check train train-tune mlflow-ui explain serve export-model docker-build docker-run docker-build-lambda ecr-create ecr-login ecr-push lambda-create lambda-url lambda-update deploy redeploy gar-create cloudrun-push cloudrun-deploy cloudrun-redeploy

# ── AWS config ────────────────────────────────────────────────────────────────
AWS_REGION     ?= us-east-1
AWS_ACCOUNT_ID ?= $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
ECR_REGISTRY    = $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
ECR_REPO        = $(ECR_REGISTRY)/credit-risk-api
LAMBDA_NAME    ?= credit-risk-scorer
LAMBDA_ROLE    ?= arn:aws:iam::$(AWS_ACCOUNT_ID):role/lambda-basic-execution

install:
	uv sync

install-dev:
	uv sync --extra dev

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

test:
	uv run pytest tests/ -v --cov=credit_risk --cov-report=term-missing

setup-db:
	uv run python -c "from credit_risk.data.ingest import get_engine, create_schema; create_schema(get_engine()); print('Schema created.')"

download:
	uv run python -c "from credit_risk.data.ingest import download_from_kaggle; download_from_kaggle()"

ingest: setup-db
	uv run python -m credit_risk.data.ingest

leakage-check:
	uv run python -c "\
	import pandas as pd; \
	from credit_risk.data.leakage import report_leakage; \
	report_leakage(pd.read_csv('data/raw/application_train.csv'))"

train:
	uv run python -m credit_risk.models.train

train-tune:
	uv run python -m credit_risk.models.train --tune

mlflow-ui:
	uv run mlflow ui

explain:
	uv run python -m credit_risk.explain.shap_explain

serve:
	uv run uvicorn credit_risk.api.main:app --reload --port 8000

export-model:
	uv run python scripts/export_model.py

docker-build: export-model
	docker build -t credit-risk-api .

# For Lambda x86_64 deployment (cross-compile from Apple Silicon):
docker-build-lambda: export-model
	docker build --platform linux/amd64 -t credit-risk-api-lambda .

docker-run:
	docker compose up

# ── AWS deployment ────────────────────────────────────────────────────────────

ecr-create:
	aws ecr create-repository --repository-name credit-risk-api --region $(AWS_REGION)

ecr-login:
	aws ecr get-login-password --region $(AWS_REGION) | \
		docker login --username AWS --password-stdin $(ECR_REGISTRY)

ecr-push: ecr-login
	docker tag credit-risk-api-lambda $(ECR_REPO):latest
	docker push $(ECR_REPO):latest

lambda-create:
	aws lambda create-function \
		--function-name $(LAMBDA_NAME) \
		--package-type Image \
		--code ImageUri=$(ECR_REPO):latest \
		--role $(LAMBDA_ROLE) \
		--memory-size 512 \
		--timeout 30 \
		--region $(AWS_REGION)

lambda-url:
	aws lambda create-function-url-config \
		--function-name $(LAMBDA_NAME) \
		--auth-type NONE \
		--region $(AWS_REGION)
	aws lambda add-permission \
		--function-name $(LAMBDA_NAME) \
		--statement-id FunctionURLAllowPublicAccess \
		--action lambda:InvokeFunctionUrl \
		--principal "*" \
		--function-url-auth-type NONE \
		--region $(AWS_REGION)

lambda-update:
	aws lambda update-function-code \
		--function-name $(LAMBDA_NAME) \
		--image-uri $(ECR_REPO):latest \
		--region $(AWS_REGION)

# First-time full deploy: build → push → create function → expose URL
deploy: docker-build-lambda ecr-push lambda-create lambda-url

# Retrain and redeploy: rebuild image → push → update function code
redeploy: docker-build-lambda ecr-push lambda-update

# ── Google Cloud Run deployment ───────────────────────────────────────────────

GCR_REGION   ?= europe-west1
GCP_PROJECT  ?= solar-panel-469619
GAR_REPO      = $(GCR_REGION)-docker.pkg.dev/$(GCP_PROJECT)/credit-risk
CLOUDRUN_NAME ?= credit-risk-scorer

gar-create:
	gcloud artifacts repositories create credit-risk \
		--repository-format=docker \
		--location=$(GCR_REGION) \
		--project=$(GCP_PROJECT)

cloudrun-push: docker-build-lambda
	gcloud auth configure-docker $(GCR_REGION)-docker.pkg.dev --quiet
	docker tag credit-risk-api-lambda $(GAR_REPO)/api:latest
	docker push $(GAR_REPO)/api:latest

cloudrun-deploy: cloudrun-push
	gcloud run deploy $(CLOUDRUN_NAME) \
		--image $(GAR_REPO)/api:latest \
		--region $(GCR_REGION) \
		--memory 512Mi \
		--timeout 30 \
		--min-instances 0 \
		--max-instances 5 \
		--allow-unauthenticated \
		--port 8000 \
		--command=python \
		--args=-m,uvicorn,credit_risk.api.main:app,--host,0.0.0.0,--port,8000 \
		--project $(GCP_PROJECT)

cloudrun-redeploy: cloudrun-push
	gcloud run deploy $(CLOUDRUN_NAME) \
		--image $(GAR_REPO)/api:latest \
		--region $(GCR_REGION) \
		--project $(GCP_PROJECT)
