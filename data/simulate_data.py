import pandas as pd
import numpy as np
import os

np.random.seed(42)

# Activity types an elderly person does daily
ACTIVITIES = [
    "wake_up",
    "kitchen",      # breakfast
    "living_room",  # TV / sitting
    "kitchen",      # lunch
    "bedroom",      # nap
    "living_room",  # evening
    "kitchen",      # dinner
    "bedroom"       # sleep
]

# Base times for each activity (24hr format as float)
# e.g. 8.0 = 8:00 AM, 13.5 = 1:30 PM
BASE_TIMES = [6.5, 8.0, 10.0, 13.0, 15.0, 17.0, 19.0, 21.5]

def generate_normal_day(date, noise=0.5):
    """Generate one normal day with small random variation."""
    records = []
    for i, activity in enumerate(ACTIVITIES):
        # Occasionally skip an activity naturally (not anomalous, just human)
        if np.random.random() < 0.03:
            continue
        actual_time = BASE_TIMES[i] + np.random.uniform(-noise, noise)
        hour = int(actual_time)
        minute = int((actual_time - hour) * 60)
        timestamp = pd.Timestamp(date) + pd.Timedelta(hours=hour, minutes=minute)
        records.append({
            "timestamp": timestamp,
            "activity": activity,
            "day_type": "normal"
        })
    return records

def generate_anomalous_day(date):
    """
    Generate a clearly anomalous day.
    - Very few activities (3-4 max)
    - Long inactive gaps
    - Missing key activities like kitchen visits
    - Unusual timing
    """
    records = []
    # Only pick 3 random activities from the full list
    selected_indices = np.random.choice(len(ACTIVITIES), size=3, replace=False)
    selected_indices = sorted(selected_indices)

    for i in selected_indices:
        activity = ACTIVITIES[i]
        # Add large random shift to timing
        actual_time = BASE_TIMES[i] + np.random.uniform(-3, 3)
        actual_time = max(0, min(23, actual_time))
        hour = int(actual_time)
        minute = int((actual_time - hour) * 60)
        timestamp = pd.Timestamp(date) + pd.Timedelta(hours=hour, minutes=minute)
        records.append({
            "timestamp": timestamp,
            "activity": activity,
            "day_type": "anomalous"
        })
    return records

def generate_dataset(days=30, anomaly_days=3):
    all_records = []
    dates = pd.date_range(start="2026-05-01", periods=days, freq="D")
    
    # Pick last few days as anomalous
    anomaly_indices = list(range(days - anomaly_days, days))
    
    for i, date in enumerate(dates):
        if i in anomaly_indices:
            records = generate_anomalous_day(date)
        else:
            records = generate_normal_day(date)
        all_records.extend(records)
    
    df = pd.DataFrame(all_records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_dataset(days=30, anomaly_days=3)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/activity_log.csv", index=False)
    print(f"Generated {len(df)} activity records")
    print(df.tail(20))
    