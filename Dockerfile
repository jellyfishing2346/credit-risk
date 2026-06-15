FROM public.ecr.aws/lambda/python:3.13

# Runtime-only deps — no mlflow/optuna/kaggle/psycopg2 needed at inference
RUN pip install --no-cache-dir \
    "pandas>=2.2" \
    "numpy>=1.26" \
    "scikit-learn>=1.5" \
    "xgboost>=2.0" \
    "cloudpickle>=3.0" \
    "fastapi>=0.111" \
    "uvicorn>=0.30" \
    "pydantic>=2.7" \
    "mangum>=0.17"

# Application source
COPY src/ ${LAMBDA_TASK_ROOT}/

# Baked model artefacts — run `make export-model` before `make docker-build`
COPY model_artifacts/ ${LAMBDA_TASK_ROOT}/model_artifacts/

CMD ["credit_risk.api.lambda_handler.handler"]
