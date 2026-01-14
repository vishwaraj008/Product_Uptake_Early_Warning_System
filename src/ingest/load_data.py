import pandas as pd
import os

from src.db.db_utils import write_prescriptions


# Expected schema

REQUIRED_COLUMNS = {
    "date",
    "product",
    "region",
    "units",
    "price_per_unit",
    "revenue",
    "event_type"
}

DATA_PATH = "data/prescriptions.csv"


def load_prescriptions(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    #  Schema validation 
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    #  Type conversions 
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = ["units", "price_per_unit", "revenue"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    #  Data quality checks 
    if df["date"].isnull().any():
        raise ValueError("Invalid or null dates found")

    if (df["units"] < 0).any():
        raise ValueError("Negative units found")

    if (df["price_per_unit"] <= 0).any():
        raise ValueError("Invalid price_per_unit values")

    if (df["revenue"] < 0).any():
        raise ValueError("Negative revenue values")

    #  Sorting (important for time-series) 
    df = df.sort_values(["product", "region", "date"]).reset_index(drop=True)

    return df


# Entry point

if __name__ == "__main__":
    print("📥 Loading prescription data...")

    df = load_prescriptions(DATA_PATH)

    print(f" Data validated: {len(df)} rows")

    write_prescriptions(df)

    print(" Data written to MySQL table: prescriptions")
