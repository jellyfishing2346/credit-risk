.PHONY: install install-dev lint format test setup-db download ingest leakage-check train train-tune mlflow-ui explain serve

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
