import pandas as pd
import numpy as np
from datetime import timedelta
import random
import os


# Configuration

START_DATE = "2021-01-04"   # Monday
WEEKS = 160                # ~3 years

PRODUCTS = ["Drug_A", "Drug_B"]
REGIONS = ["North", "South", "East", "West"]

BASE_VOLUME = {
    "Drug_A": 1200,
    "Drug_B": 800
}

PRICE_PER_UNIT = {
    "Drug_A": 450.0,   # INR
    "Drug_B": 300.0
}

EVENT_TYPES = ["supply_issue", "competitor_entry", "promotion"]

OUTPUT_PATH = "data/prescriptions.csv"

np.random.seed(42)
random.seed(42)


# Baseline generator

def generate_baseline_series(base, weeks):
    t = np.arange(weeks)

    trend = 1 + 0.015 * t / 52
    seasonality = 1 + 0.25 * np.sin(2 * np.pi * t / 52)
    noise = np.random.normal(0, 0.08, weeks)

    series = base * trend * seasonality * (1 + noise)

    return np.maximum(series, 0)  # keep FLOAT


# Event injector

def inject_event(series, event_type, start_idx):
    series = series.copy()
    duration = 0

    if event_type == "supply_issue":
        duration = random.randint(3, 6)
        impact = random.uniform(0.35, 0.55)
        series[start_idx:start_idx + duration] *= (1 - impact)

    elif event_type == "competitor_entry":
        duration = random.randint(8, 14)
        decay = np.linspace(0.1, 0.4, duration)
        for i in range(duration):
            series[start_idx + i] *= (1 - decay[i])

    elif event_type == "promotion":
        duration = random.randint(2, 4)
        lift = random.uniform(0.25, 0.4)
        series[start_idx:start_idx + duration] *= (1 + lift)

    return series, duration


# Dataset generator

def generate_dataset():
    start_date = pd.to_datetime(START_DATE)
    rows = []

    for product in PRODUCTS:
        for region in REGIONS:
            baseline = generate_baseline_series(
                BASE_VOLUME[product], WEEKS
            )

            # Inject 1–2 events
            event_weeks = random.sample(range(20, WEEKS - 20), random.randint(1, 2))
            event_map = {}

            for ew in event_weeks:
                etype = random.choice(EVENT_TYPES)
                baseline, duration = inject_event(baseline, etype, ew)
                for d in range(duration):
                    event_map[ew + d] = etype

            # Build rows
            for i in range(WEEKS):
                date = start_date + timedelta(weeks=i)
                units = int(round(baseline[i]))
                price = PRICE_PER_UNIT[product]
                revenue = units * price

                rows.append({
                    "date": date.date(),
                    "product": product,
                    "region": region,
                    "units": units,
                    "price_per_unit": price,
                    "revenue": revenue,
                    "event_type": event_map.get(i, "none")
                })

    return pd.DataFrame(rows)


# Entry point

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)

    print(" Synthetic prescription data generated")
    print(f" Path: {OUTPUT_PATH}")
    print(f" Rows: {len(df)}")
