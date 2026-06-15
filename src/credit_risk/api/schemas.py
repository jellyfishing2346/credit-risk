"""
Pydantic schemas for the Credit Risk Scoring API.

Request: the most important application fields are defined explicitly so the
OpenAPI docs are useful; any additional field passes through via extra="allow".

Response: score + risk band + top SHAP contributors.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RiskBand(str, Enum):
    LOW = "LOW"        # prob < 0.05
    MEDIUM = "MEDIUM"  # 0.05 ≤ prob < 0.15
    HIGH = "HIGH"      # prob ≥ 0.15


def classify_risk(prob: float) -> RiskBand:
    if prob < 0.05:
        return RiskBand.LOW
    if prob < 0.15:
        return RiskBand.MEDIUM
    return RiskBand.HIGH


class ScoreRequest(BaseModel):
    """
    Loan application features.

    All fields are optional — missing values are imputed by the preprocessing
    pipeline using training-set medians / modes.  Auxiliary pre-aggregated
    features (bureau_*, inst_*, prev_*, pos_*, cc_*) can be included when
    available from a feature store.
    """
    model_config = ConfigDict(extra="allow")

    application_id: str | None = Field(default=None, description="Caller-supplied reference ID")

    # Core application fields (most impactful per SHAP analysis)
    amt_income_total: float | None = None
    amt_credit: float | None = None
    amt_annuity: float | None = None
    amt_goods_price: float | None = None
    days_birth: int | None = Field(default=None, description="Days before application (negative)")
    days_employed: int | None = Field(default=None, description="Days before application (negative); 365243 = never employed")
    code_gender: str | None = Field(default=None, description="M or F")
    flag_own_car: str | None = Field(default=None, description="Y or N")
    flag_own_realty: str | None = Field(default=None, description="Y or N")
    name_income_type: str | None = None
    name_education_type: str | None = None
    name_family_status: str | None = None
    name_housing_type: str | None = None
    name_contract_type: str | None = None
    ext_source_1: float | None = Field(default=None, ge=0, le=1)
    ext_source_2: float | None = Field(default=None, ge=0, le=1)
    ext_source_3: float | None = Field(default=None, ge=0, le=1)
    region_rating_client: int | None = None
    organization_type: str | None = None
    occupation_type: str | None = None
    cnt_children: int | None = None
    cnt_fam_members: float | None = None


class ShapDriver(BaseModel):
    feature: str
    shap_value: float = Field(description="Contribution to log-odds (positive = increases default risk)")
    direction: str = Field(description="INCREASES_RISK or DECREASES_RISK")


class ShapExplanation(BaseModel):
    top_drivers: list[ShapDriver]


class ScoreResponse(BaseModel):
    application_id: str | None
    default_probability: float = Field(description="Predicted probability of default (0–1)")
    risk_band: RiskBand
    model_run_id: str
    shap_explanation: ShapExplanation
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    run_id: str | None
