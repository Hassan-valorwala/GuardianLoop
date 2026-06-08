import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def extract_features(df):
    """
    Convert raw activity logs into numerical features per day.
    Each day becomes one row of numbers for the ML model.
    """
    features = []
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    
    for date, group in df.groupby("date"):
        group = group.sort_values("timestamp")
        timestamps = group["timestamp"]
        hours = timestamps.dt.hour + timestamps.dt.minute / 60
        activities = group["activity"].tolist()
        
        # Feature 1: what hour did first activity happen
        hour_first = hours.iloc[0]
        
        # Feature 2: what hour did last activity happen
        hour_last = hours.iloc[-1]
        
        # Feature 3: total number of activities
        total_activities = len(group)
        
        # Feature 4: average gap between activities (in hours)
        if len(hours) > 1:
            gaps = hours.diff().dropna()
            avg_gap = gaps.mean()
            max_gap = gaps.max()
        else:
            avg_gap = 0
            max_gap = 0
        
        # Feature 5: how many times kitchen was visited
        kitchen_count = activities.count("kitchen")
        
        # Feature 6: how many times bedroom was visited
        bedroom_count = activities.count("bedroom")
        
        # Feature 7: how many times living room was visited
        living_room_count = activities.count("living_room")
        
        # Feature 8: active hours span (last - first)
        active_span = hour_last - hour_first

        features.append({
            "date": date,
            "hour_first": hour_first,
            "hour_last": hour_last,
            "total_activities": total_activities,
            "avg_gap": avg_gap,
            "max_gap": max_gap,
            "kitchen_count": kitchen_count,
            "bedroom_count": bedroom_count,
            "living_room_count": living_room_count,
            "active_span": active_span
        })
    
    return pd.DataFrame(features)

def train_model(feature_df):
    """
    Train Isolation Forest on normal days only.
    """
    feature_cols = [
        "hour_first", "hour_last", "total_activities",
        "avg_gap", "max_gap", "kitchen_count",
        "bedroom_count", "living_room_count", "active_span"
    ]
    
    X = feature_df[feature_cols].values
    
    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    model.fit(X)
    return model, feature_cols

def save_model(model, feature_cols):
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/guardian_model.pkl")
    joblib.dump(feature_cols, "model/feature_cols.pkl")
    print("Model saved to model/")

def load_model():
    model = joblib.load("model/guardian_model.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    return model, feature_cols

if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("data/activity_log.csv")
    
    # Extract features
    feature_df = extract_features(df)
    print("Features extracted:")
    print(feature_df)
    
    # Train only on normal days
    normal_df = feature_df.head(27)  # first 27 days are normal
    model, feature_cols = train_model(normal_df)
    
    # Score all days
    X_all = feature_df[feature_cols].values
    scores = model.decision_function(X_all)

    feature_df["anomaly_score"] = scores

    # Use manual threshold instead of model's built-in prediction
    # Anomalous days consistently score below -0.05
    THRESHOLD = -0.09
    feature_df["prediction"] = feature_df["anomaly_score"].apply(
        lambda x: -1 if x < THRESHOLD else 1
    )
    
    print("\nAnomaly Detection Results:")
    print(feature_df[["date", "anomaly_score", "prediction"]].to_string())
    
    # Save model
    save_model(model, feature_cols)