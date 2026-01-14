import pandas as pd
import numpy as np



# Configuration

SEVERITY_THRESHOLDS = {
    "low": 0.15,
    "medium": 0.30,
    "high": 0.45
}



# Cause inference (rule-based)

def infer_likely_cause(pct_deviation_series: pd.Series) -> str:
    """
    Infers likely business cause based on anomaly shape.
    Transparent & explainable by design.
    """

    if pct_deviation_series.mean() < -0.35:
        return "Supply Issue / Recall"

    if pct_deviation_series.mean() < -0.20 and len(pct_deviation_series) >= 6:
        return "Competitor Entry"

    if pct_deviation_series.mean() > 0.20:
        return "Promotion / Campaign"

    return "Unclassified"



# Impact scoring logic

def score_impacts(
    anomaly_df: pd.DataFrame,
    price_per_unit: float
) -> pd.DataFrame:
    """
    Converts anomaly signals into ranked business impacts.
    """

    # Keep only anomaly points
    anomalies = anomaly_df[anomaly_df["is_anomaly"]].copy()

    if anomalies.empty:
        return pd.DataFrame()

     # Revenue impact 
    anomalies["revenue_impact"] = (
        anomalies["residual"].abs() * price_per_unit
    )

    # Group consecutive anomalies into events 
    anomalies["event_id"] = (
        anomalies["date"].diff().dt.days.ne(7)
    ).cumsum()

    events = []

    for _, group in anomalies.groupby("event_id"):
        start_date = group["date"].min()
        end_date = group["date"].max()
        duration_weeks = len(group)

        total_revenue_impact = group["revenue_impact"].sum()
        avg_pct_dev = group["pct_deviation"].mean()

        # Severity 
        abs_dev = abs(avg_pct_dev)
        if abs_dev >= SEVERITY_THRESHOLDS["high"]:
            severity = "High"
        elif abs_dev >= SEVERITY_THRESHOLDS["medium"]:
            severity = "Medium"
        else:
            severity = "Low"

       #  Likely cause 
        cause = infer_likely_cause(group["pct_deviation"])

        events.append({
            "start_date": start_date,
            "end_date": end_date,
            "duration_weeks": duration_weeks,
            "avg_pct_deviation": avg_pct_dev,
            "total_revenue_impact": total_revenue_impact,
            "severity": severity,
            "likely_cause": cause
        })

    impact_df = pd.DataFrame(events)

   # Rank by business priority 
    impact_df = impact_df.sort_values(
        by=["severity", "total_revenue_impact"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return impact_df



# Local test

if __name__ == "__main__":
    # Dummy test input (for structure validation)
    data = {
        "date": pd.date_range("2023-01-01", periods=6, freq="W"),
        "actual_units": [800, 780, 760, 740, 730, 720],
        "expected_units": [1000] * 6,
        "residual": [-200, -220, -240, -260, -270, -280],
        "pct_deviation": [-0.20, -0.22, -0.24, -0.26, -0.27, -0.28],
        "z_score": [-4, -4.1, -4.2, -4.3, -4.4, -4.5],
        "is_anomaly": [True] * 6
    }

    test_df = pd.DataFrame(data)

    result = score_impacts(test_df, price_per_unit=450)

    print(" Impact scoring test output")
    print(result)
