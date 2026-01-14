import streamlit as st
import pandas as pd

from src.models.anomaly_detector import detect_anomalies
from src.models.impact_scorer import score_impacts
from src.db.db_utils import read_prescriptions
from src.models.backtesting import backtest



st.set_page_config(
    page_title="Prescription Uptake Early Warning System",
    layout="wide"
)

st.title("Prescription Uptake Early-Warning System")
st.caption(
    "Detects abnormal prescription changes, quantifies business impact, "
    "and prioritizes actions."
)



st.sidebar.header("Filters")

products = ["Drug_A", "Drug_B"]
regions = ["North", "South", "East", "West"]

product = st.sidebar.selectbox("Select Product", products)
region = st.sidebar.selectbox("Select Region", regions)

PRICE_LOOKUP = {
    "Drug_A": 450,
    "Drug_B": 300
}

metrics, _ = backtest(
    product=product,
    region=region,
    price_per_unit=PRICE_LOOKUP[product]
)

st.subheader("Model Validation (Backtesting)")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="Precision",
    value=f"{metrics['precision']:.2f}"
)

col2.metric(
    label="Recall",
    value=f"{metrics['recall']:.2f}"
)

col3.metric(
    label="False Positives",
    value=str(metrics["false_positives"])
)

latency = metrics["avg_detection_latency_weeks"]
col4.metric(
    label="Avg Detection Latency (weeks)",
    value="N/A" if latency is None else f"{latency:.1f}"
)


base_df = read_prescriptions(product=product, region=region)
base_df["date"] = pd.to_datetime(base_df["date"])

if base_df.empty:
    st.warning("No data available for selection.")
    st.stop()


anomaly_df = detect_anomalies(product, region)


impact_df = score_impacts(
    anomaly_df,
    price_per_unit=PRICE_LOOKUP[product]
)


st.subheader("1️⃣ Prescription Trend & Detected Anomalies")

chart_df = anomaly_df.copy()
chart_df = chart_df.set_index("date")

st.line_chart(
    chart_df[["actual_units", "expected_units"]],
    height=350
)

anomaly_points = chart_df[chart_df["is_anomaly"]]

if not anomaly_points.empty:
    st.markdown("**⚠️ Detected Anomalies**")
    st.dataframe(
        anomaly_points[
            ["actual_units", "expected_units", "pct_deviation", "z_score"]
        ].round(2),
        use_container_width=True
    )
else:
    st.success("No anomalies detected for this selection.")


st.subheader("2️⃣ Ranked Business Impact")

if impact_df.empty:
    st.success("No high-impact events identified.")
else:
    st.dataframe(
        impact_df.round(2),
        use_container_width=True
    )


st.subheader("3️⃣ Executive Summary")

if not impact_df.empty:
    top_event = impact_df.iloc[0]

    summary = f"""
**Key Finding:**  
Prescription uptake for **{product}** in **{region}** shows a  
**{abs(top_event.avg_pct_deviation)*100:.1f}% deviation** from expected levels
over **{top_event.duration_weeks} weeks**.

**Business Impact:**  
Estimated revenue impact of **₹{top_event.total_revenue_impact:,.0f}**.

**Severity:** {top_event.severity}  
**Likely Cause:** {top_event.likely_cause}

**Recommended Action:**  
Immediate investigation and targeted field intervention.
"""

    st.markdown(summary)
else:
    st.markdown("No executive actions required at this time.")


st.caption(
    "Prototype built for analytics-driven decision support. "
    "Data is synthetic and used for demonstration purposes."
)
