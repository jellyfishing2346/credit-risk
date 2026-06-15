# Model Card — Credit Risk Scoring Engine

This model card follows the framework proposed by Mitchell et al. (2019) and is intended to provide transparency about the model's design, training, performance, and limitations.

---

## Model Details

| Field | Value |
|---|---|
| Model type | XGBoost (gradient boosted trees) |
| Version | 1.0 |
| Training date | June 2026 |
| Framework | XGBoost 2.0+, scikit-learn 1.5+ |
| MLflow run ID | `91a4f51bd49c46b397134a77d9482dfa` |
| License | MIT |
| Author | Faizan Khan |

### Architecture

A gradient boosted tree ensemble trained on tabular loan application data. The pipeline consists of:

1. **Preprocessing** — sentinel replacement, high-missingness column dropping, median imputation for numerics, one-hot encoding for categoricals
2. **Feature engineering** — 176 features derived from 8 source tables, including bureau history, previous applications, instalment behaviour, POS cash records, and credit card usage
3. **Model** — XGBoost classifier with scale_pos_weight to handle class imbalance; hyperparameters tuned via Optuna (30 Bayesian trials)
4. **Explainability** — per-prediction SHAP values computed using XGBoost's native TreeSHAP implementation

---

## Intended Use

### Primary use case

Scoring individual loan applications to estimate the probability of default. The model returns a continuous probability (0–1) and a discrete risk band (LOW / MEDIUM / HIGH), alongside the top features driving each prediction.

### Intended users

- Data scientists evaluating credit risk modelling approaches
- ML engineers building risk scoring pipelines
- Researchers studying explainable AI in the financial domain

### Out-of-scope uses

- **Production credit decisioning without human review.** This model is a research and portfolio project. It has not undergone the regulatory review, fairness auditing, or validation required for deployment in any real lending context.
- **Any jurisdiction subject to fair lending laws** (ECOA, FCRA in the US; GDPR Article 22 in the EU) without appropriate legal and ethical review.
- **Real-time fraud detection.** The model is trained to predict loan default, not fraudulent intent.

---

## Training Data

### Dataset

**Home Credit Default Risk** — a publicly available competition dataset released by Home Credit Group on Kaggle.

| Table | Rows | Description |
|---|---|---|
| `application_train` | 307,511 | Main application table with target |
| `bureau` | 1,716,428 | Credit bureau records |
| `bureau_balance` | 27,299,925 | Monthly bureau balances |
| `previous_application` | 1,670,214 | Prior Home Credit applications |
| `pos_cash_balance` | 10,001,358 | POS and cash loan history |
| `installments_payments` | 13,605,401 | Instalment payment records |
| `credit_card_balance` | 3,840,312 | Credit card balance history |

### Target variable

`TARGET` — binary flag: 1 = client had payment difficulties (late by 3+ days on first 3 instalments), 0 = no difficulties.

### Class distribution

| Class | Count | Proportion |
|---|---|---|
| 0 (no default) | 282,686 | 91.9% |
| 1 (default) | 24,825 | 8.1% |

The 8.1% default rate represents significant class imbalance. This is addressed during training via `scale_pos_weight ≈ 11.3`.

### Data splits

| Split | Size | Purpose |
|---|---|---|
| Train | 184,506 (60%) | Model fitting |
| Validation | 61,502 (20%) | Early stopping, HPO evaluation |
| Test | 61,503 (20%) | Final held-out evaluation |

All splits are stratified on the target to preserve the class ratio.

### Preprocessing decisions

- **DAYS_EMPLOYED sentinel**: The value `365243` encodes "never employed". It is replaced with NaN and an additional binary flag column `days_employed_anom` is added.
- **High-missingness columns**: Columns with >60% missing values are dropped at fit time and excluded from inference.
- **Column alignment**: The preprocessor stores the column order seen at fit time. At inference, input DataFrames are realigned via `reindex()` so partial feature sets (missing auxiliary table features) do not cause errors.

---

## Evaluation

### Metrics

| Split | ROC-AUC | Gini | Average Precision |
|---|---|---|---|
| Validation | 0.7743 | — | — |
| **Test** | **0.7795** | **0.5591** | **0.2763** |

**ROC-AUC** is the primary metric because:
- It is threshold-independent, which matters when the decision threshold is set operationally
- It handles class imbalance more robustly than accuracy
- It is the standard metric for the Home Credit Kaggle competition (public benchmark: ~0.79)

**Gini coefficient** = 2 × AUC − 1 = 0.559. Values above 0.5 are generally considered acceptable for credit scoring.

**Average Precision** (area under the precision-recall curve) = 0.276. This is low relative to a balanced dataset but expected given the 8.1% positive rate.

### Hyperparameter optimisation results

Optuna ran 30 trials of Bayesian optimisation (TPE sampler) over:

| Parameter | Search range |
|---|---|
| `max_depth` | 3 – 8 |
| `learning_rate` | 0.01 – 0.3 (log scale) |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |
| `min_child_weight` | 1 – 10 |
| `reg_alpha` | 1e-8 – 10.0 (log scale) |
| `reg_lambda` | 1e-8 – 10.0 (log scale) |

### Top global features by mean |SHAP|

| Rank | Feature | Interpretation |
|---|---|---|
| 1 | `ext_source_prod23` | Product of two external credit bureau scores |
| 2 | `ext_source_2` | External credit bureau score 2 |
| 3 | `ext_source_3` | External credit bureau score 3 |
| 4 | `days_birth` | Applicant age (older = lower risk) |
| 5 | `credit_term` | Loan amount ÷ annuity (longer term = higher risk) |
| 6 | `goods_credit_ratio` | Goods price ÷ credit amount |
| 7 | `code_gender` | Applicant gender |
| 8 | `annuity_income_ratio` | Debt burden relative to income |

External credit scores dominate, which is consistent with the credit risk literature. The EXT_SOURCE interaction feature (`ext_source_prod23`) being the single most important feature suggests these scores carry complementary information.

---

## Limitations

### Known limitations

- **External scores are often missing at inference.** EXT_SOURCE_2 and EXT_SOURCE_3 are the two most important features but are not always available for new applicants. When absent, the model falls back to median imputation, which reduces predictive confidence.

- **Geographic specificity.** The dataset is from a Central/Eastern European lender. Behavioural and macroeconomic patterns may not generalise to other geographies.

- **Temporal drift.** The dataset has a fixed observation window. Model performance will degrade over time as borrower behaviour and macroeconomic conditions change. Periodic retraining is required.

- **Benchmark gap.** The test ROC-AUC of 0.7795 falls slightly short of the ~0.79 Kaggle public benchmark. The remaining gap is likely attributable to missing bureau_balance monthly aggregations and the absence of hand-crafted interaction features used by top competition entries.

- **Binary target.** The model predicts a binary default flag (3+ days late in first 3 instalments). It does not predict loss given default (LGD) or exposure at default (EAD), which are needed for full expected loss estimation.

### What this model cannot do

- Predict default probability for applicants significantly outside the training distribution
- Provide causal explanations — SHAP values reflect model logic, not causal mechanisms
- Guarantee fairness across protected groups without dedicated fairness analysis

---

## Ethical Considerations

### Fairness

`code_gender` appears as a top-8 global feature. In many jurisdictions, using gender in credit decisioning is restricted or prohibited under fair lending law. Before any production use, a fairness audit should be conducted covering:

- Demographic parity across gender, age, and any available protected class
- Equalised odds analysis
- Individual fairness under similar financial profiles

### Explainability

Every prediction from the `/score` API includes SHAP values for the top-10 contributing features. This supports the "right to explanation" requirement under GDPR Article 22 for automated decision-making, though legal review is still required for any real deployment.

### Data privacy

The training data contains sensitive financial information. The dataset is de-identified per Kaggle's terms. No personally identifiable information (PII) is stored, logged, or returned by the API.

---

## References

- Home Credit Default Risk dataset — https://www.kaggle.com/competitions/home-credit-default-risk
- Mitchell et al. (2019) — Model Cards for Model Reporting — https://arxiv.org/abs/1810.03993
- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System — https://arxiv.org/abs/1603.02754
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP) — https://arxiv.org/abs/1705.07874
- Akiba et al. (2019) — Optuna: A Next-generation Hyperparameter Optimization Framework — https://arxiv.org/abs/1907.10902
