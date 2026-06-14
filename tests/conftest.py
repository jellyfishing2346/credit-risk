import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_app_df() -> pd.DataFrame:
    """
    Minimal applications DataFrame that exercises all pipeline branches:
    - DAYS_EMPLOYED sentinel values (every 10th row)
    - a column with >60% missing (should be dropped)
    - numeric columns with some NaN
    - binary and low-cardinality categorical columns
    """
    n = 200
    rng = np.random.default_rng(42)

    days_employed = rng.integers(-3000, -100, n).astype(float)
    days_employed[::10] = 365243  # sentinel: never employed

    ext_source_3 = rng.uniform(0.0, 1.0, n)
    ext_source_3[rng.random(n) < 0.3] = np.nan  # ~30% missing — kept

    high_missing = rng.random(n)
    high_missing[rng.random(n) < 0.7] = np.nan  # ~70% missing — dropped

    return pd.DataFrame({
        "sk_id_curr": np.arange(n),
        "target": rng.integers(0, 2, n),
        "days_birth": rng.integers(-20000, -8000, n),
        "days_employed": days_employed,
        "amt_income_total": rng.uniform(50_000, 500_000, n),
        "amt_credit": rng.uniform(100_000, 1_000_000, n),
        "ext_source_2": rng.uniform(0.0, 1.0, n),
        "ext_source_3": ext_source_3,
        "code_gender": rng.choice(["M", "F"], n),
        "flag_own_car": rng.choice(["Y", "N"], n),
        "flag_own_realty": rng.choice(["Y", "N"], n),
        "name_income_type": rng.choice(
            ["Working", "Commercial associate", "Pensioner", "State servant"], n
        ),
        "name_education_type": rng.choice(
            ["Secondary / secondary special", "Higher education", "Incomplete higher"], n
        ),
        "high_missing_col": high_missing,
    })
