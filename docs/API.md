# API Reference

The Credit Risk Scoring API is a REST API built with FastAPI. It exposes two endpoints: a health check and a scoring endpoint that returns a default probability with SHAP-based explanations.

**Base URL (production):** `https://credit-risk-scorer-218846370015.europe-west1.run.app`

**Interactive docs:** `/docs` (Swagger UI) · `/redoc` (ReDoc)

---

## Authentication

The current deployment uses no authentication (`--allow-unauthenticated` on Google Cloud Run). For production use, restrict access using Cloud Run IAM or place an API Gateway with key validation in front of the service.

---

## Endpoints

### GET /health

Returns the current liveness status of the service and whether the model is loaded.

**Request**

```
GET /health
```

**Response 200**

```json
{
  "status": "ok",
  "model_loaded": true,
  "run_id": "91a4f51bd49c46b397134a77d9482dfa"
}
```

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` if the service is running |
| `model_loaded` | boolean | Whether the XGBoost model is loaded in memory |
| `run_id` | string \| null | MLflow run ID of the loaded model |

**Example**

```bash
curl https://credit-risk-scorer-218846370015.europe-west1.run.app/health
```

---

### POST /score

Score a loan application. Returns the probability of default, a risk band, and a SHAP-based explanation of the top factors driving the prediction.

**Request**

```
POST /score
Content-Type: application/json
```

All fields are optional. The model handles missing values through the fitted preprocessing pipeline (median imputation for numerics, mode for binary categoricals, "Unknown" fill for categoricals). Pass whatever fields are available — more fields generally produce more accurate scores.

**Request body**

```json
{
  "application_id": "APP-001",
  "amt_credit": 250000,
  "amt_income_total": 90000,
  "amt_annuity": 20000,
  "amt_goods_price": 230000,
  "days_birth": -12775,
  "days_employed": -3650,
  "ext_source_1": 0.50,
  "ext_source_2": 0.62,
  "ext_source_3": 0.55,
  "code_gender": "M",
  "flag_own_car": "N",
  "flag_own_realty": "Y",
  "name_contract_type": "Cash loans",
  "name_income_type": "Working",
  "name_education_type": "Secondary / secondary special",
  "name_family_status": "Married",
  "name_housing_type": "House / apartment",
  "occupation_type": "Laborers",
  "organization_type": "Business Entity Type 3",
  "region_population_relative": 0.019,
  "days_registration": -2000,
  "days_id_publish": -1500,
  "cnt_children": 0,
  "cnt_fam_members": 2
}
```

**Key fields**

| Field | Type | Notes |
|---|---|---|
| `application_id` | string | Returned in the response for correlation. Not used by the model. |
| `amt_credit` | float | Total loan amount requested |
| `amt_income_total` | float | Applicant annual income |
| `amt_annuity` | float | Loan annuity (monthly payment × 12) |
| `amt_goods_price` | float | Price of goods the loan is financing |
| `days_birth` | int | Age in days, negative (e.g. -12775 = ~35 years old) |
| `days_employed` | int | Employment duration in days, negative. Use 365243 to indicate never employed — the pipeline handles this sentinel automatically. |
| `ext_source_1/2/3` | float [0–1] | External credit bureau scores. These are the strongest predictors — include them when available. |
| `code_gender` | string | `"M"` or `"F"` |
| `name_contract_type` | string | `"Cash loans"` or `"Revolving loans"` |

Any additional fields present in the original Home Credit dataset can be passed and will be used by the model.

**Response 200**

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
      },
      {
        "feature": "cat__name_contract_type_Cash loans",
        "shap_value": -0.14542,
        "direction": "DECREASES_RISK"
      },
      {
        "feature": "num__days_birth",
        "shap_value": 0.10764,
        "direction": "INCREASES_RISK"
      }
    ]
  }
}
```

**Response fields**

| Field | Type | Description |
|---|---|---|
| `application_id` | string | Echoed from the request |
| `default_probability` | float [0–1] | Predicted probability of loan default |
| `risk_band` | string | `LOW`, `MEDIUM`, or `HIGH` (see thresholds below) |
| `model_run_id` | string | MLflow run ID of the model that produced this score |
| `latency_ms` | float | Server-side processing time in milliseconds |
| `shap_explanation.top_drivers` | array | Top 10 features by absolute SHAP value |

**SHAP driver fields**

| Field | Type | Description |
|---|---|---|
| `feature` | string | Preprocessed feature name. Prefix indicates transformer: `num__` = numeric, `cat__` = categorical (one-hot), `bin__` = binary ordinal |
| `shap_value` | float | SHAP contribution in log-odds space. Positive = increases risk, negative = decreases risk |
| `direction` | string | `INCREASES_RISK` or `DECREASES_RISK` |

**Risk band thresholds**

| Band | Condition | Interpretation |
|---|---|---|
| `LOW` | probability < 0.05 | Low likelihood of default |
| `MEDIUM` | 0.05 ≤ probability < 0.15 | Moderate risk — warrants review |
| `HIGH` | probability ≥ 0.15 | High likelihood of default |

**Response 503**

Returned if the model failed to load at startup.

```json
{
  "detail": "Model not loaded"
}
```

---

## Error Codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 404 | Route not found (e.g. GET / has no handler) |
| 405 | Method not allowed (e.g. GET /score — must be POST) |
| 422 | Validation error — request body is malformed JSON |
| 503 | Model not loaded |

---

## Examples

### Minimal request (no optional fields)

```bash
curl -X POST https://credit-risk-scorer-218846370015.europe-west1.run.app/score \
  -H "Content-Type: application/json" \
  -d '{}'
```

The model will impute all missing values using training-set medians. The score will be less informative but will not error.

### High-risk applicant

```bash
curl -X POST https://credit-risk-scorer-218846370015.europe-west1.run.app/score \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": "HIGH-001",
    "amt_credit": 500000,
    "amt_income_total": 45000,
    "ext_source_2": 0.15,
    "ext_source_3": 0.10,
    "days_birth": -9125,
    "name_contract_type": "Cash loans"
  }'
```

### Low-risk applicant

```bash
curl -X POST https://credit-risk-scorer-218846370015.europe-west1.run.app/score \
  -H "Content-Type: application/json" \
  -d '{
    "application_id": "LOW-001",
    "amt_credit": 100000,
    "amt_income_total": 200000,
    "ext_source_2": 0.85,
    "ext_source_3": 0.80,
    "days_birth": -18250,
    "name_contract_type": "Revolving loans"
  }'
```

### Python client

```python
import requests

API_URL = "https://credit-risk-scorer-218846370015.europe-west1.run.app"

payload = {
    "application_id": "APP-001",
    "amt_credit": 250000,
    "amt_income_total": 90000,
    "ext_source_2": 0.62,
    "ext_source_3": 0.55,
    "days_birth": -12775,
}

response = requests.post(f"{API_URL}/score", json=payload)
response.raise_for_status()

data = response.json()
print(f"Default probability: {data['default_probability']:.1%}")
print(f"Risk band: {data['risk_band']}")
print(f"Latency: {data['latency_ms']}ms")
print("\nTop drivers:")
for driver in data["shap_explanation"]["top_drivers"]:
    sign = "▲" if driver["direction"] == "INCREASES_RISK" else "▼"
    print(f"  {sign} {driver['feature']:<40} {driver['shap_value']:+.4f}")
```

---

## Performance

| Environment | Warm latency (p50) | Notes |
|---|---|---|
| Local (uvicorn) | ~25ms | Apple Silicon M-series |
| Docker (local) | ~30ms | linux/amd64 emulated |
| Google Cloud Run | ~32ms | europe-west1, 512MB |

Cold start (first request after ~15 min idle): 8–15 seconds. Set `--min-instances 1` on Cloud Run to eliminate cold starts.

---

## Local Development

```bash
# Start API with hot reload
make serve

# Run against local API
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"application_id": "LOCAL-001", "amt_credit": 250000}'
```
