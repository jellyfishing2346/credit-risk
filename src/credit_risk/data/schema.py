"""
SQLAlchemy Core table definitions for the Home Credit Default Risk dataset.

Column names are lowercased to follow PostgreSQL convention; they match the CSV headers
after `df.columns.str.lower()` is applied during ingestion.
"""
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
)

metadata = MetaData()

# ---------------------------------------------------------------------------
# Main fact table: one row per loan application
# Source: application_train.csv (target present) + application_test.csv (target NULL)
# ---------------------------------------------------------------------------
applications = Table(
    "applications",
    metadata,
    Column("sk_id_curr", Integer, primary_key=True),
    Column("target", SmallInteger, nullable=True),  # NULL for test-set rows
    # Contract
    Column("name_contract_type", String(50)),
    # Applicant demographics
    Column("code_gender", String(1)),
    Column("flag_own_car", String(1)),
    Column("flag_own_realty", String(1)),
    Column("cnt_children", Integer),
    Column("amt_income_total", Numeric(15, 2)),
    Column("amt_credit", Numeric(15, 2)),
    Column("amt_annuity", Numeric(15, 2)),
    Column("amt_goods_price", Numeric(15, 2)),
    Column("name_type_suite", String(100)),
    Column("name_income_type", String(100)),
    Column("name_education_type", String(100)),
    Column("name_family_status", String(100)),
    Column("name_housing_type", String(100)),
    Column("region_population_relative", Float),
    # Time offsets (days before application date — negative values are normal)
    Column("days_birth", Integer),
    Column("days_employed", Integer),  # 365243 = never employed sentinel → replaced in pipeline
    Column("days_registration", Float),
    Column("days_id_publish", Integer),
    Column("own_car_age", Float),
    # Contact / reachability flags
    Column("flag_mobil", SmallInteger),
    Column("flag_emp_phone", SmallInteger),
    Column("flag_work_phone", SmallInteger),
    Column("flag_cont_mobile", SmallInteger),
    Column("flag_phone", SmallInteger),
    Column("flag_email", SmallInteger),
    # Employment
    Column("occupation_type", String(100)),
    Column("cnt_fam_members", Float),
    Column("region_rating_client", SmallInteger),
    Column("region_rating_client_w_city", SmallInteger),
    Column("weekday_appr_process_start", String(20)),
    Column("hour_appr_process_start", SmallInteger),
    Column("organization_type", String(100)),
    # Address / region mismatch flags
    Column("reg_region_not_live_region", SmallInteger),
    Column("reg_region_not_work_region", SmallInteger),
    Column("live_region_not_work_region", SmallInteger),
    Column("reg_city_not_live_city", SmallInteger),
    Column("reg_city_not_work_city", SmallInteger),
    Column("live_city_not_work_city", SmallInteger),
    # Normalised external credit scores (from credit bureaus / partners)
    Column("ext_source_1", Float),
    Column("ext_source_2", Float),
    Column("ext_source_3", Float),
    # Building / apartment statistics — 14 attributes × 3 aggregations (AVG / MODE / MEDI)
    Column("apartments_avg", Float),
    Column("basementarea_avg", Float),
    Column("years_beginexpluatation_avg", Float),
    Column("years_build_avg", Float),
    Column("commonarea_avg", Float),
    Column("elevators_avg", Float),
    Column("entrances_avg", Float),
    Column("floorsmax_avg", Float),
    Column("floorsmin_avg", Float),
    Column("landarea_avg", Float),
    Column("livingapartments_avg", Float),
    Column("livingarea_avg", Float),
    Column("nonlivingapartments_avg", Float),
    Column("nonlivingarea_avg", Float),
    Column("apartments_mode", Float),
    Column("basementarea_mode", Float),
    Column("years_beginexpluatation_mode", Float),
    Column("years_build_mode", Float),
    Column("commonarea_mode", Float),
    Column("elevators_mode", Float),
    Column("entrances_mode", Float),
    Column("floorsmax_mode", Float),
    Column("floorsmin_mode", Float),
    Column("landarea_mode", Float),
    Column("livingapartments_mode", Float),
    Column("livingarea_mode", Float),
    Column("nonlivingapartments_mode", Float),
    Column("nonlivingarea_mode", Float),
    Column("apartments_medi", Float),
    Column("basementarea_medi", Float),
    Column("years_beginexpluatation_medi", Float),
    Column("years_build_medi", Float),
    Column("commonarea_medi", Float),
    Column("elevators_medi", Float),
    Column("entrances_medi", Float),
    Column("floorsmax_medi", Float),
    Column("floorsmin_medi", Float),
    Column("landarea_medi", Float),
    Column("livingapartments_medi", Float),
    Column("livingarea_medi", Float),
    Column("nonlivingapartments_medi", Float),
    Column("nonlivingarea_medi", Float),
    # Categorical building descriptors
    Column("fondkapremont_mode", String(100)),
    Column("housetype_mode", String(100)),
    Column("totalarea_mode", Float),
    Column("wallsmaterial_mode", String(100)),
    Column("emergencystate_mode", String(10)),
    # Social circle defaults (30-day / 60-day observation windows)
    Column("obs_30_cnt_social_circle", Float),
    Column("def_30_cnt_social_circle", Float),
    Column("obs_60_cnt_social_circle", Float),
    Column("def_60_cnt_social_circle", Float),
    Column("days_last_phone_change", Float),
    # Submitted document flags (flag_document_2 … flag_document_21)
    *[Column(f"flag_document_{i}", SmallInteger) for i in range(2, 22)],
    # Credit bureau enquiry counts in different time windows
    Column("amt_req_credit_bureau_hour", Float),
    Column("amt_req_credit_bureau_day", Float),
    Column("amt_req_credit_bureau_week", Float),
    Column("amt_req_credit_bureau_mon", Float),
    Column("amt_req_credit_bureau_qrt", Float),
    Column("amt_req_credit_bureau_year", Float),
)

# ---------------------------------------------------------------------------
# Credit bureau history (one application → many bureau records)
# ---------------------------------------------------------------------------
bureau = Table(
    "bureau",
    metadata,
    Column("sk_id_bureau", Integer, primary_key=True),
    Column("sk_id_curr", Integer, ForeignKey("applications.sk_id_curr"), index=True),
    Column("credit_active", String(50)),
    Column("credit_currency", String(20)),
    Column("days_credit", Integer),
    Column("credit_day_overdue", Integer),
    Column("days_credit_enddate", Float),
    Column("days_enddate_fact", Float),
    Column("amt_credit_max_overdue", Float),
    Column("cnt_credit_prolong", Integer),
    Column("amt_credit_sum", Float),
    Column("amt_credit_sum_debt", Float),
    Column("amt_credit_sum_limit", Float),
    Column("amt_credit_sum_overdue", Float),
    Column("credit_type", String(100)),
    Column("days_credit_update", Integer),
    Column("amt_annuity", Float),
)

bureau_balance = Table(
    "bureau_balance",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sk_id_bureau", Integer, ForeignKey("bureau.sk_id_bureau"), index=True),
    Column("months_balance", Integer),
    Column("status", String(10)),
)

# ---------------------------------------------------------------------------
# Previous loan applications at Home Credit
# ---------------------------------------------------------------------------
previous_applications = Table(
    "previous_applications",
    metadata,
    Column("sk_id_prev", Integer, primary_key=True),
    Column("sk_id_curr", Integer, ForeignKey("applications.sk_id_curr"), index=True),
    Column("name_contract_type", String(50)),
    Column("amt_annuity", Float),
    Column("amt_application", Float),
    Column("amt_credit", Float),
    Column("amt_down_payment", Float),
    Column("amt_goods_price", Float),
    Column("weekday_appr_process_start", String(20)),
    Column("hour_appr_process_start", SmallInteger),
    Column("flag_last_appl_per_contract", String(5)),
    Column("nflag_last_appl_in_day", SmallInteger),
    Column("rate_down_payment", Float),
    Column("rate_interest_primary", Float),
    Column("rate_interest_privileged", Float),
    Column("name_cash_loan_purpose", String(100)),
    Column("name_contract_status", String(50)),
    Column("days_decision", Integer),
    Column("name_payment_type", String(100)),
    Column("code_reject_reason", String(50)),
    Column("name_type_suite", String(100)),
    Column("name_client_type", String(50)),
    Column("name_goods_category", String(100)),
    Column("name_portfolio", String(50)),
    Column("name_product_type", String(50)),
    Column("channel_type", String(100)),
    Column("sellerplace_area", Float),
    Column("name_seller_industry", String(100)),
    Column("cnt_payment", Float),
    Column("name_yield_group", String(50)),
    Column("product_combination", String(100)),
    Column("days_first_drawing", Float),
    Column("days_first_due", Float),
    Column("days_last_due_1st_version", Float),
    Column("days_last_due", Float),
    Column("days_termination", Float),
    Column("nflag_insured_on_approval", Float),
)

# ---------------------------------------------------------------------------
# Point-of-sale cash balance monthly snapshots
# ---------------------------------------------------------------------------
pos_cash_balance = Table(
    "pos_cash_balance",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sk_id_prev", Integer, ForeignKey("previous_applications.sk_id_prev"), index=True),
    Column("sk_id_curr", Integer, ForeignKey("applications.sk_id_curr"), index=True),
    Column("months_balance", Integer),
    Column("cnt_instalment", Float),
    Column("cnt_instalment_future", Float),
    Column("name_contract_status", String(50)),
    Column("sk_dpd", Integer),
    Column("sk_dpd_def", Integer),
)

# ---------------------------------------------------------------------------
# Instalment payment history
# ---------------------------------------------------------------------------
installments_payments = Table(
    "installments_payments",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sk_id_prev", Integer, ForeignKey("previous_applications.sk_id_prev"), index=True),
    Column("sk_id_curr", Integer, ForeignKey("applications.sk_id_curr"), index=True),
    Column("num_instalment_version", Float),
    Column("num_instalment_number", Integer),
    Column("days_instalment", Float),
    Column("days_entry_payment", Float),
    Column("amt_instalment", Float),
    Column("amt_payment", Float),
)

# ---------------------------------------------------------------------------
# Credit card balance monthly snapshots
# ---------------------------------------------------------------------------
credit_card_balance = Table(
    "credit_card_balance",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sk_id_prev", Integer, ForeignKey("previous_applications.sk_id_prev"), index=True),
    Column("sk_id_curr", Integer, ForeignKey("applications.sk_id_curr"), index=True),
    Column("months_balance", Integer),
    Column("amt_balance", Float),
    Column("amt_credit_limit_actual", Float),
    Column("amt_drawings_atm_current", Float),
    Column("amt_drawings_current", Float),
    Column("amt_drawings_other_current", Float),
    Column("amt_drawings_pos_current", Float),
    Column("amt_inst_min_regularity", Float),
    Column("amt_payment_current", Float),
    Column("amt_payment_total_current", Float),
    Column("amt_receivable_principal", Float),
    Column("amt_recivable", Float),
    Column("amt_total_receivable", Float),
    Column("cnt_drawings_atm_current", Float),
    Column("cnt_drawings_current", Integer),
    Column("cnt_drawings_other_current", Float),
    Column("cnt_drawings_pos_current", Float),
    Column("cnt_instalment_mature_cum", Float),
    Column("name_contract_status", String(50)),
    Column("sk_dpd", Integer),
    Column("sk_dpd_def", Integer),
)
