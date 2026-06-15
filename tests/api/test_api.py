"""
API tests — mock the model store so no real model or data is needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from credit_risk.api.schemas import (
    RiskBand,
    ScoreResponse,
    ShapDriver,
    ShapExplanation,
    classify_risk,
)


# ---------------------------------------------------------------------------
# classify_risk
# ---------------------------------------------------------------------------

def test_classify_low():
    assert classify_risk(0.01) == RiskBand.LOW
    assert classify_risk(0.049) == RiskBand.LOW


def test_classify_medium():
    assert classify_risk(0.05) == RiskBand.MEDIUM
    assert classify_risk(0.10) == RiskBand.MEDIUM
    assert classify_risk(0.149) == RiskBand.MEDIUM


def test_classify_high():
    assert classify_risk(0.15) == RiskBand.HIGH
    assert classify_risk(0.80) == RiskBand.HIGH


# ---------------------------------------------------------------------------
# API endpoints (model store mocked)
# ---------------------------------------------------------------------------

def _make_mock_store(prob: float = 0.12):
    store = MagicMock()
    store.is_loaded = True
    store.run_id = "test-run-abc123"
    store.predict.return_value = ScoreResponse(
        application_id="TEST-001",
        default_probability=prob,
        risk_band=classify_risk(prob),
        model_run_id="test-run-abc123",
        shap_explanation=ShapExplanation(
            top_drivers=[
                ShapDriver(feature="num__ext_source_2", shap_value=-0.5, direction="DECREASES_RISK"),
                ShapDriver(feature="num__inst_days_late_mean", shap_value=0.3, direction="INCREASES_RISK"),
            ]
        ),
        latency_ms=None,
    )
    return store


@pytest.fixture
def client():
    from credit_risk.api.main import app, _store
    mock = _make_mock_store()
    with patch("credit_risk.api.main._store", mock):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock


def test_health_ok(client):
    c, mock = client
    mock.is_loaded = True
    mock.run_id = "run-xyz"
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_score_returns_200(client):
    c, _ = client
    resp = c.post("/score", json={
        "application_id": "TEST-001",
        "ext_source_2": 0.45,
        "ext_source_3": 0.73,
        "amt_income_total": 202500.0,
        "amt_credit": 406597.5,
        "amt_annuity": 24700.5,
    })
    assert resp.status_code == 200


def test_score_response_shape(client):
    c, _ = client
    resp = c.post("/score", json={"application_id": "TEST-001"})
    body = resp.json()
    assert "default_probability" in body
    assert "risk_band" in body
    assert "shap_explanation" in body
    assert "top_drivers" in body["shap_explanation"]


def test_score_probability_range(client):
    c, _ = client
    resp = c.post("/score", json={})
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0


def test_score_risk_band_values(client):
    c, _ = client
    resp = c.post("/score", json={})
    body = resp.json()
    assert body["risk_band"] in ("LOW", "MEDIUM", "HIGH")


def test_score_shap_drivers_have_direction(client):
    c, _ = client
    resp = c.post("/score", json={})
    drivers = resp.json()["shap_explanation"]["top_drivers"]
    assert len(drivers) > 0
    for d in drivers:
        assert d["direction"] in ("INCREASES_RISK", "DECREASES_RISK")


def test_score_empty_body_accepted(client):
    """All fields are optional — empty body must not 422."""
    c, _ = client
    resp = c.post("/score", json={})
    assert resp.status_code == 200


def test_score_extra_fields_accepted(client):
    """extra='allow' — unknown fields should pass through without 422."""
    c, _ = client
    resp = c.post("/score", json={"some_future_field": 42.0})
    assert resp.status_code == 200


def test_score_503_when_model_not_loaded():
    from credit_risk.api.main import app
    unloaded = MagicMock()
    unloaded.is_loaded = False
    with patch("credit_risk.api.main._store", unloaded):
        with TestClient(app, raise_server_exceptions=False) as c:
            resp = c.post("/score", json={})
    assert resp.status_code == 503
