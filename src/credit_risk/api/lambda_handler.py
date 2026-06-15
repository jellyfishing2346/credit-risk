"""
AWS Lambda entrypoint — wraps the FastAPI app with Mangum's ASGI adapter.

The lifespan context manager in main.py fires at Lambda cold start,
loading the model from model_artifacts/ (baked into the container image).
Subsequent warm invocations skip loading entirely.
"""
from mangum import Mangum

from credit_risk.api.main import app

handler = Mangum(app, lifespan="on")
