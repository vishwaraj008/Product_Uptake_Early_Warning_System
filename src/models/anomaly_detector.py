import pandas as pd
import numpy as np
from prophet import Prophet

from src.db.db_utils import read_prescriptions


# Configuration

Z_SCORE_THRESHOLD = 3.5


# Core anomaly detection logic

def detect_anomalies(product: str, region: str) -> pd.DataFrame:
    """
    Pulls data from MySQL for a given product & region,
    fits a seasonal baseline, and flags anomalies.
    """

    # Read from SQL 
    df = read_prescriptions(product=product, region=region)

    if df.empty:
        raise ValueError(f"No data found for {product} - {region}")

    #  CRITICAL FIX: normalize date dtype 
    df["date"] = pd.to_datetime(df["date"])

    #  Prepare data for Prophet 
    ts = (
        df[["date", "units"]]
        .rename(columns={"date": "ds", "units": "y"})
        .copy()
    )

    ts = ts.sort_values("ds")

    #  Fit baseline model 
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )

    model.fit(ts)

    forecast = model.predict(ts)

    #  CRITICAL FIX: ensure forecast ds is datetime 
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    # Merge forecast with actuals 
    result = ts.merge(
        forecast[["ds", "yhat"]],
        on="ds",
        how="left"
    )

    #  Residual-based anomaly score 
    result["residual"] = result["y"] - result["yhat"]

    median = np.median(result["residual"])
    mad = np.median(np.abs(result["residual"] - median)) + 1e-6

    result["z_score"] = (result["residual"] - median) / mad
    result["is_anomaly"] = result["z_score"].abs() > Z_SCORE_THRESHOLD

    #  Business context 

    result["pct_deviation"] = result["residual"] / result["yhat"]

    # Final formatting 
    result = result.rename(
        columns={
            "ds": "date",
            "y": "actual_units",
            "yhat": "expected_units"
        }
    )

    return result[
        [
            "date",
            "actual_units",
            "expected_units",
            "residual",
            "pct_deviation",
            "z_score",
            "is_anomaly"
        ]
    ]



# Local test

if __name__ == "__main__":
    product = "Drug_A"
    region = "North"

    df = detect_anomalies(product, region)

    print(f" Anomaly detection completed for {product} - {region}")
    print(df[df["is_anomaly"]].head())
