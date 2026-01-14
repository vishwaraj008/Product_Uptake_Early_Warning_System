import pandas as pd
import numpy as np

from src.db.db_utils import read_prescriptions
from src.models.anomaly_detector import detect_anomalies

# -------------------------------
# Helpers
# -------------------------------
def _weeks_between(d1, d2):
    return int(abs((d2 - d1).days) / 7)


def _group_events(df: pd.DataFrame, col: str):
    """
    Groups consecutive non-'none' labels into events.
    """
    events = []
    current = None

    for _, r in df.iterrows():
        label = r[col]
        date = r["date"]

        if label != "none":
            if current is None:
                current = {
                    "label": label,
                    "start": date,
                    "end": date
                }
            else:
                # continue event if same label and consecutive week
                if label == current["label"] and (date - current["end"]).days <= 7:
                    current["end"] = date
                else:
                    events.append(current)
                    current = {"label": label, "start": date, "end": date}
        else:
            if current is not None:
                events.append(current)
                current = None

    if current is not None:
        events.append(current)

    return events


# -------------------------------
# Backtesting core
# -------------------------------
def backtest(product: str, region: str, price_per_unit: float):
    """
    Computes detection metrics using synthetic ground truth.
    """

    # ---- Pull ground truth ----
    gt = read_prescriptions(product=product, region=region)
    gt["date"] = pd.to_datetime(gt["date"])
    gt = gt.sort_values("date")

    gt_events = _group_events(gt, col="event_type")

    # ---- Run detector ----
    anoms = detect_anomalies(product, region)
    anoms = anoms.sort_values("date")

    # Build anomaly events (consecutive anomaly weeks)
    anoms["flag"] = anoms["is_anomaly"].astype(int)
    anom_events = _group_events(
        anoms.assign(event_type=anoms["is_anomaly"].map(lambda x: "anomaly" if x else "none")),
        col="event_type"
    )

    # ---- Match detections to ground truth ----
    matches = []
    used_anoms = set()

    for gt_e in gt_events:
        # find earliest anomaly overlapping or after gt start
        candidates = []
        for i, a in enumerate(anom_events):
            if i in used_anoms:
                continue
            # overlap or starts after
            if a["end"] >= gt_e["start"]:
                candidates.append((i, a))

        if candidates:
            i, a = min(candidates, key=lambda x: abs((x[1]["start"] - gt_e["start"]).days))
            used_anoms.add(i)
            latency = _weeks_between(gt_e["start"], a["start"])
            matches.append({
                "gt_label": gt_e["label"],
                "gt_start": gt_e["start"],
                "detected_at": a["start"],
                "latency_weeks": latency
            })

    # ---- Metrics ----
    true_positives = len(matches)
    total_alerts = len(anom_events)
    total_gt = len(gt_events)

    precision = true_positives / total_alerts if total_alerts else 0.0
    recall = true_positives / total_gt if total_gt else 0.0
    false_positives = total_alerts - true_positives
    avg_latency = (
        np.mean([m["latency_weeks"] for m in matches])
        if matches else None
    )

    metrics = {
        "product": product,
        "region": region,
        "ground_truth_events": total_gt,
        "detected_events": total_alerts,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "avg_detection_latency_weeks": round(avg_latency, 2) if avg_latency is not None else None
    }

    return metrics, pd.DataFrame(matches)


# -------------------------------
# Local run
# -------------------------------
if __name__ == "__main__":
    product = "Drug_A"
    region = "North"
    price = 450

    metrics, matches = backtest(product, region, price)

    print(" Backtesting metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if not matches.empty:
        print("\nMatched events:")
        print(matches)
