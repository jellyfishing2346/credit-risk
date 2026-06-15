# Contributing Guide

Thank you for taking the time to contribute. This document covers everything you need to get set up, the conventions used throughout the codebase, and how to submit changes.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Conventions](#project-conventions)
- [Branching & Commits](#branching--commits)
- [Adding Features](#adding-features)
- [Writing Tests](#writing-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Bugs](#reporting-bugs)

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/your-username/credit-risk.git
cd credit-risk
```

### 2. Install dependencies

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
make install-dev
```

This installs all base, API, dashboard, and dev dependencies into an isolated `.venv`.

### 3. Verify the setup

```bash
make test
```

All 41 tests should pass before you make any changes.

### 4. Download the dataset (optional)

Only needed if you're working on data, features, or model changes. Requires a Kaggle account with the [Home Credit competition rules](https://www.kaggle.com/competitions/home-credit-default-risk) accepted.

```bash
make download
```

---

## Project Conventions

### Code style

All code is formatted and linted with [Ruff](https://docs.astral.sh/ruff/).

```bash
make format   # auto-fix formatting
make lint     # check for issues
```

Line length is 100 characters. Imports are sorted automatically. Do not disable Ruff rules inline without a comment explaining why.

### Type annotations

All public functions must have type annotations. Use `from __future__ import annotations` at the top of every file so annotations are evaluated lazily.

```python
# Good
def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[...]:
    ...

# Bad
def split_data(X, y):
    ...
```

### Comments

Only add a comment when the **why** is non-obvious — a hidden constraint, a workaround for a specific bug, or a subtle invariant. Do not comment what the code does; well-named identifiers already do that.

```python
# Good — explains a non-obvious constraint
# DAYS_EMPLOYED = 365243 is a sentinel for "never employed", not a real value
X["days_employed"] = X["days_employed"].replace(365243, np.nan)

# Bad — restates the code
# Replace 365243 with NaN
X["days_employed"] = X["days_employed"].replace(365243, np.nan)
```

### Imports

- Standard library first, then third-party, then internal — separated by blank lines
- Lazy-import heavy dependencies (mlflow, cloudpickle) inside the functions that need them so the module can be imported in environments where those packages are absent (e.g. the production container has no mlflow)

```python
# Good — mlflow only imported when actually needed
def load_artifacts(run_id=None):
    import mlflow
    import mlflow.xgboost
    ...
```

### No backwards-compatibility shims

If you rename or remove something, change all call sites. Do not add re-exports, aliases, or `# removed` comments for deleted code.

---

## Branching & Commits

### Branch names

```
feature/bureau-aggregations
fix/shap-column-alignment
docs/api-reference
```

### Commit messages

Use the imperative mood. One line for simple changes; add a blank line and a body for anything non-trivial.

```
Add bureau balance aggregation features

Adds dpd_mean, dpd_max, and months_balance_min from the bureau_balance
table. These were identified as the most impactful missing aggregations
in the Phase 2 ROC-AUC gap analysis.
```

Keep the subject line under 72 characters. Do not reference issue numbers in the subject — put them in the body.

---

## Adding Features

### New engineered features

All feature engineering lives in [src/credit_risk/features/engineer.py](src/credit_risk/features/engineer.py). Each source table has its own `build_*_features()` function.

Follow this pattern:

```python
def build_bureau_balance_features(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bureau_balance per SK_ID_BUREAU.
    Returns a DataFrame indexed on sk_id_bureau.
    """
    agg = bureau_balance.groupby("sk_id_bureau").agg(
        bb_dpd_mean=("dpd", "mean"),
        bb_dpd_max=("dpd", "max"),
    )
    return agg
```

Then join it inside `build_full_feature_matrix()`. After adding features, always:

1. Run `make train` to verify the feature matrix builds without error
2. Check that ROC-AUC does not regress
3. Add a test in `tests/test_pipeline.py` if the feature involves a new transformer step

### New API fields

Request fields are defined in [src/credit_risk/api/schemas.py](src/credit_risk/api/schemas.py). The `ScoreRequest` model uses `ConfigDict(extra="allow")` so any field passed in the JSON body is forwarded to the model — you do not need to add every feature explicitly. Only add a field to `ScoreRequest` if you want explicit typing or validation.

### New endpoints

Add routes to [src/credit_risk/api/main.py](src/credit_risk/api/main.py). Every new endpoint needs:
- A Pydantic response model in `schemas.py`
- At least two tests in `tests/api/test_api.py` (happy path + error case)

---

## Writing Tests

Tests live in `tests/` and mirror the `src/credit_risk/` structure.

```
tests/
├── test_pipeline.py
├── models/test_train.py
├── explain/test_shap_explain.py
└── api/test_api.py
```

### Rules

- **No real data in tests.** Use small synthetic DataFrames. The full dataset takes minutes to load.
- **No MLflow in tests.** Mock `load_artifacts` or use the `ModelStore` mock pattern already in `test_api.py`.
- **Test behaviour, not implementation.** Assert on outputs, not internal state.
- Each test should be independent — no shared mutable state between tests.

### Running tests

```bash
make test                          # full suite with coverage
uv run pytest tests/test_pipeline.py -v   # single file
uv run pytest -k "sentinel" -v            # filter by name
```

Coverage must not drop below the current level. Check with:

```bash
uv run pytest --cov=credit_risk --cov-report=term-missing
```

---

## Submitting a Pull Request

1. **Ensure tests pass** — `make test` must be green
2. **Lint is clean** — `make lint` must produce no errors
3. **One concern per PR** — do not mix feature work with refactoring
4. **Fill in the PR description** — explain what changed and why, not just what

PR title format:
```
Add bureau_balance DPD aggregations
Fix SHAP column alignment at inference time
Update Cloud Run deploy target for new region
```

A maintainer will review within a few days. Feedback will be left as inline comments on the diff.

---

## Reporting Bugs

Open an issue with:

1. **What you expected** to happen
2. **What actually happened** — include the full error traceback
3. **How to reproduce** — the exact command or code that triggers it
4. **Environment** — Python version (`python --version`), OS, and whether you are running locally or in Docker

If the bug is in the live API, include the `model_run_id` from the response — it tells us exactly which model version was running.
