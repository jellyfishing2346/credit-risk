"""
Leakage detection for the Home Credit feature matrix.

Two checks:
  1. Pattern-based — column names that suggest post-origination information
     (e.g. "past_due", "charge_off"). These fields would only exist after the
     loan has been serviced and therefore cannot be known at application time.

  2. Correlation-based — numeric columns with suspiciously high correlation to
     the target variable.  |corr| > 0.95 is treated as critical; values above
     a configurable warn_threshold are reported for review.

Usage:
    from credit_risk.data.leakage import report_leakage
    import pandas as pd
    report_leakage(pd.read_csv("data/raw/application_train.csv"))
"""
from __future__ import annotations

import pandas as pd

# Substrings that indicate post-origination information.
# These are generic patterns; extend the list as new tables are joined in.
_POST_ORIGINATION_PATTERNS = [
    "past_due",
    "dpd",           # days past due
    "delinquency",
    "charge_off",
    "recovery",
    "write_off",
    "default_flag",
    "npl",           # non-performing loan
    "restructur",
]

LeakageReport = dict[str, list[str]]


def check_leakage(
    df: pd.DataFrame,
    target_col: str = "target",
    warn_threshold: float = 0.3,
    critical_threshold: float = 0.95,
) -> LeakageReport:
    """
    Return a dict with three keys:
    - 'pattern_flags'    : columns whose names match known post-origination patterns
    - 'high_correlation' : columns with |corr(col, target)| > warn_threshold  (sorted desc)
    - 'near_perfect'     : columns with |corr(col, target)| > critical_threshold
    """
    col_lower = {c.lower(): c for c in df.columns if c != target_col}

    pattern_flags = [
        original
        for lower, original in col_lower.items()
        if any(p in lower for p in _POST_ORIGINATION_PATTERNS)
    ]

    high_correlation: list[str] = []
    near_perfect: list[str] = []

    if target_col in df.columns:
        numeric = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
        corr = numeric.corrwith(df[target_col]).abs().dropna().sort_values(ascending=False)

        high_correlation = corr[corr > warn_threshold].index.tolist()
        near_perfect = corr[corr > critical_threshold].index.tolist()

    return {
        "pattern_flags": pattern_flags,
        "high_correlation": high_correlation,
        "near_perfect": near_perfect,
    }


def report_leakage(
    df: pd.DataFrame,
    target_col: str = "target",
    warn_threshold: float = 0.3,
    critical_threshold: float = 0.95,
) -> None:
    """Print a human-readable leakage report to stdout."""
    results = check_leakage(df, target_col, warn_threshold, critical_threshold)

    print("=" * 50)
    print("Leakage Check Report")
    print("=" * 50)

    if results["near_perfect"]:
        print(f"\n[CRITICAL] Near-perfect predictors (|corr| > {critical_threshold}):")
        for col in results["near_perfect"]:
            print(f"  - {col}")
    else:
        print(f"\n[OK] No near-perfect predictors (|corr| > {critical_threshold})")

    if results["pattern_flags"]:
        print("\n[WARNING] Columns matching post-origination name patterns:")
        for col in results["pattern_flags"]:
            print(f"  - {col}")
    else:
        print("\n[OK] No post-origination name patterns detected")

    print(f"\n[INFO] Top correlated features (|corr| > {warn_threshold}):")
    if results["high_correlation"]:
        for col in results["high_correlation"][:15]:
            print(f"  - {col}")
    else:
        print("  none above threshold")

    print("=" * 50)
