import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = "https://credit-risk-scorer-218846370015.europe-west1.run.app"

st.set_page_config(page_title="Credit Risk Scorer", page_icon="🏦", layout="wide")
st.title("🏦 Credit Risk Scoring Engine")
st.caption("Powered by XGBoost + TreeSHAP · Home Credit Default Risk dataset")

st.divider()

left, right = st.columns([1, 1.5], gap="large")

with left:
    st.subheader("Loan Application")

    application_id = st.text_input("Application ID", value="APP-001")

    st.markdown("**Financials**")
    amt_credit = st.number_input("Loan Amount ($)", min_value=0, value=250_000, step=5_000)
    amt_income_total = st.number_input("Annual Income ($)", min_value=0, value=90_000, step=5_000)
    amt_annuity = st.number_input("Annual Annuity ($)", min_value=0, value=20_000, step=1_000)
    amt_goods_price = st.number_input("Goods Price ($)", min_value=0, value=230_000, step=5_000)

    st.markdown("**Applicant**")
    age_years = st.slider("Age", min_value=18, max_value=70, value=35)
    gender = st.selectbox("Gender", ["M", "F"])
    contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])

    st.markdown("**External Credit Scores**")
    ext_source_2 = st.slider("External Score 2", 0.0, 1.0, 0.5, 0.01)
    ext_source_3 = st.slider("External Score 3", 0.0, 1.0, 0.5, 0.01)

    submitted = st.button("Score Application →", type="primary", use_container_width=True)

with right:
    if not submitted:
        st.info("Fill in the application details on the left and click **Score Application**.")
    else:
        payload = {
            "application_id": application_id,
            "amt_credit": float(amt_credit),
            "amt_income_total": float(amt_income_total),
            "amt_annuity": float(amt_annuity),
            "amt_goods_price": float(amt_goods_price),
            "days_birth": -(age_years * 365),
            "ext_source_2": ext_source_2,
            "ext_source_3": ext_source_3,
            "code_gender": gender,
            "name_contract_type": contract_type,
        }

        with st.spinner("Scoring..."):
            try:
                resp = requests.post(f"{API_URL}/score", json=payload, timeout=30)
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(f"API error: {e}")
                st.stop()

        data = resp.json()
        prob = data["default_probability"]
        risk_band = data["risk_band"]
        drivers = data["shap_explanation"]["top_drivers"]

        # ── Risk band banner ──────────────────────────────────────────────────
        band_color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[risk_band]
        st.markdown(f"### Risk Band: :{band_color}[**{risk_band}**]")

        # ── Probability gauge ─────────────────────────────────────────────────
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36}},
            title={"text": "Default Probability", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": band_color},
                "steps": [
                    {"range": [0, 5], "color": "#d4edda"},
                    {"range": [5, 15], "color": "#fff3cd"},
                    {"range": [15, 100], "color": "#f8d7da"},
                ],
            },
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── SHAP waterfall ────────────────────────────────────────────────────
        df = pd.DataFrame(drivers)
        df["label"] = (
            df["feature"]
            .str.replace(r"^(num__|cat__|bin__)", "", regex=True)
            .str.replace("_", " ")
        )
        df = df.sort_values("shap_value")

        fig_shap = go.Figure(go.Bar(
            x=df["shap_value"],
            y=df["label"],
            orientation="h",
            marker_color=["#dc3545" if v > 0 else "#0d6efd" for v in df["shap_value"]],
            text=[f"{v:+.3f}" for v in df["shap_value"]],
            textposition="auto",
        ))
        fig_shap.update_layout(
            title="Top Risk Drivers (SHAP)",
            xaxis_title="← Decreases risk  |  Increases risk →",
            height=380,
            margin=dict(t=40, b=20, l=10, r=80),
            xaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=1),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.caption(
            f"Model run `{data['model_run_id'][:8]}…` · "
            f"Latency {data['latency_ms']} ms"
        )
