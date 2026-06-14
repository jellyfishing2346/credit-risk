"""
Feature engineering from Home Credit auxiliary tables.

Each function aggregates one CSV to one row per SK_ID_CURR.
Call build_full_feature_matrix() to get the merged result ready for the pipeline.

Note: joining bureau + previous_application typically lifts ROC-AUC from ~0.75
(application table alone) to ~0.79 (full feature set).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[3] / "data" / "raw"


def _load(filename: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / filename, low_memory=False)
    df.columns = df.columns.str.lower()
    return df


def build_bureau_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate bureau.csv — one row per applicant, indexed by sk_id_curr."""
    bureau = _load("bureau.csv", data_dir)

    agg = bureau.groupby("sk_id_curr").agg(
        bureau_count=("sk_id_bureau", "count"),
        bureau_active_count=("credit_active", lambda x: (x == "Active").sum()),
        bureau_closed_count=("credit_active", lambda x: (x == "Closed").sum()),
        bureau_days_credit_max=("days_credit", "max"),
        bureau_days_credit_mean=("days_credit", "mean"),
        bureau_days_enddate_max=("days_credit_enddate", "max"),
        bureau_days_overdue_max=("credit_day_overdue", "max"),
        bureau_amt_credit_sum=("amt_credit_sum", "sum"),
        bureau_amt_credit_mean=("amt_credit_sum", "mean"),
        bureau_amt_debt_sum=("amt_credit_sum_debt", "sum"),
        bureau_amt_overdue_max=("amt_credit_sum_overdue", "max"),
    )
    agg["bureau_active_ratio"] = agg["bureau_active_count"] / agg["bureau_count"]
    return agg


def build_prev_app_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate previous_application.csv — one row per applicant, indexed by sk_id_curr."""
    prev = _load("previous_application.csv", data_dir)

    agg = prev.groupby("sk_id_curr").agg(
        prev_count=("sk_id_prev", "count"),
        prev_approved_count=("name_contract_status", lambda x: (x == "Approved").sum()),
        prev_refused_count=("name_contract_status", lambda x: (x == "Refused").sum()),
        prev_amt_credit_mean=("amt_credit", "mean"),
        prev_amt_credit_max=("amt_credit", "max"),
        prev_days_decision_max=("days_decision", "max"),
        prev_days_decision_mean=("days_decision", "mean"),
        prev_cnt_payment_mean=("cnt_payment", "mean"),
    )
    agg["prev_approval_rate"] = agg["prev_approved_count"] / agg["prev_count"]
    return agg


def build_installments_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate installments_payments.csv — payment behaviour per applicant."""
    inst = _load("installments_payments.csv", data_dir)

    inst["payment_diff"] = inst["amt_payment"] - inst["amt_instalment"]
    inst["days_late"] = (inst["days_entry_payment"] - inst["days_instalment"]).clip(lower=0)

    agg = inst.groupby("sk_id_curr").agg(
        inst_count=("num_instalment_number", "count"),
        inst_days_late_max=("days_late", "max"),
        inst_days_late_mean=("days_late", "mean"),
        inst_days_late_sum=("days_late", "sum"),
        inst_payment_diff_mean=("payment_diff", "mean"),
        inst_payment_diff_min=("payment_diff", "min"),
        inst_amt_payment_sum=("amt_payment", "sum"),
        inst_amt_instalment_sum=("amt_instalment", "sum"),
    )
    agg["inst_payment_ratio"] = agg["inst_amt_payment_sum"] / (
        agg["inst_amt_instalment_sum"] + 1e-9
    )
    return agg


def build_pos_cash_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate POS_CASH_balance.csv — POS loan status history per applicant."""
    pos = _load("POS_CASH_balance.csv", data_dir)

    agg = pos.groupby("sk_id_curr").agg(
        pos_count=("months_balance", "count"),
        pos_months_balance_min=("months_balance", "min"),
        pos_sk_dpd_max=("sk_dpd", "max"),
        pos_sk_dpd_mean=("sk_dpd", "mean"),
        pos_sk_dpd_def_max=("sk_dpd_def", "max"),
        pos_completed_count=("name_contract_status", lambda x: (x == "Completed").sum()),
        pos_active_count=("name_contract_status", lambda x: (x == "Active").sum()),
    )
    agg["pos_completed_ratio"] = agg["pos_completed_count"] / agg["pos_count"]
    return agg


def build_credit_card_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Aggregate credit_card_balance.csv — credit card utilisation per applicant."""
    cc = _load("credit_card_balance.csv", data_dir)

    cc["utilisation"] = cc["amt_balance"] / (cc["amt_credit_limit_actual"] + 1e-9)

    agg = cc.groupby("sk_id_curr").agg(
        cc_count=("months_balance", "count"),
        cc_utilisation_max=("utilisation", "max"),
        cc_utilisation_mean=("utilisation", "mean"),
        cc_amt_balance_max=("amt_balance", "max"),
        cc_amt_balance_mean=("amt_balance", "mean"),
        cc_amt_drawings_mean=("amt_drawings_current", "mean"),
        cc_sk_dpd_max=("sk_dpd", "max"),
        cc_sk_dpd_mean=("sk_dpd", "mean"),
    )
    return agg


def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add handcrafted domain features to the applications-level DataFrame.

    EXT_SOURCE interactions are the single highest-impact addition in Home Credit
    competitions — the three external credit scores have multiplicative signal that
    tree splits can't recover from raw columns alone.

    Domain ratios encode credit-to-income stress, which is the core underwriting signal.
    """
    X = X.copy()
    cols = X.columns.str.lower()

    # EXT_SOURCE interactions
    e1 = X.get("ext_source_1", X.get("EXT_SOURCE_1"))
    e2 = X.get("ext_source_2", X.get("EXT_SOURCE_2"))
    e3 = X.get("ext_source_3", X.get("EXT_SOURCE_3"))

    if e2 is not None and e3 is not None:
        X["ext_source_prod23"] = e2 * e3
    if e1 is not None and e2 is not None and e3 is not None:
        X["ext_source_prod123"] = e1 * e2 * e3
        X["ext_source_mean"] = (e1 + e2 + e3) / 3
        X["ext_source_std"] = X[["ext_source_1", "ext_source_2", "ext_source_3"]].std(axis=1)

    # Credit stress ratios
    income = X.get("amt_income_total", X.get("AMT_INCOME_TOTAL"))
    credit = X.get("amt_credit", X.get("AMT_CREDIT"))
    annuity = X.get("amt_annuity", X.get("AMT_ANNUITY"))
    goods = X.get("amt_goods_price", X.get("AMT_GOODS_PRICE"))
    days_birth = X.get("days_birth", X.get("DAYS_BIRTH"))
    days_employed = X.get("days_employed", X.get("DAYS_EMPLOYED"))

    if income is not None and credit is not None:
        X["credit_income_ratio"] = credit / (income + 1)
    if income is not None and annuity is not None:
        X["annuity_income_ratio"] = annuity / (income + 1)
    if annuity is not None and credit is not None:
        X["credit_term"] = annuity / (credit + 1)
    if goods is not None and credit is not None:
        X["goods_credit_ratio"] = goods / (credit + 1)
    if days_birth is not None and days_employed is not None:
        X["employed_to_age_ratio"] = days_employed / (days_birth + 1)

    return X


def build_full_feature_matrix(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load application_train.csv, left-join all auxiliary table aggregates,
    add domain features, and return (X, y) ready for CreditRiskPreprocessor.
    """
    print("Loading application_train.csv...")
    app = _load("application_train.csv", data_dir)
    y = app.pop("target")

    sources = [
        ("bureau", build_bureau_features),
        ("previous application", build_prev_app_features),
        ("installments payments", build_installments_features),
        ("POS cash balance", build_pos_cash_features),
        ("credit card balance", build_credit_card_features),
    ]

    X = app
    for name, fn in sources:
        print(f"Building {name} features...")
        X = X.join(fn(data_dir), on="sk_id_curr", how="left")

    print("Adding domain features (EXT_SOURCE interactions, credit ratios)...")
    X = add_domain_features(X)

    print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]:,} columns")
    return X, y
