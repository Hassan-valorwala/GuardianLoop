from flask import Flask, render_template, jsonify
import pandas as pd
from model import extract_features
import joblib
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

app = Flask(__name__)

THRESHOLD = -0.09

def load_model():
    model = joblib.load("model/guardian_model.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    return model, feature_cols

def get_agent_log():
    log_path = "data/agent_log.txt"
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as f:
        lines = f.readlines()
    return [line.strip() for line in lines[-20:]]  # last 20 entries

def get_activity_data():
    df = pd.read_csv("data/activity_log.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status")
def status():
    try:
        df = get_activity_data()
        feature_df = extract_features(df)
        model, feature_cols = load_model()

        X = feature_df[feature_cols].values
        scores = model.decision_function(X)
        feature_df["anomaly_score"] = scores
        feature_df["prediction"] = feature_df["anomaly_score"].apply(
            lambda x: -1 if x < THRESHOLD else 1
        )

        # Last 7 days for dashboard
        last7 = feature_df.tail(7).copy()
        last7["date"] = last7["date"].astype(str)

        days = []
        for _, row in last7.iterrows():
            days.append({
                "date": row["date"],
                "score": round(row["anomaly_score"], 4),
                "status": "anomalous" if row["prediction"] == -1 else "normal",
                "total_activities": int(row["total_activities"])
            })

        # Latest day summary
        latest = feature_df.iloc[-1]
        current_status = "anomalous" if latest["anomaly_score"] < THRESHOLD else "normal"

        return jsonify({
            "current_status": current_status,
            "current_score": round(float(latest["anomaly_score"]), 4),
            "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "days": days
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/log")
def log():
    entries = get_agent_log()
    return jsonify({"log": entries})

@app.route("/api/run-agent")
def run_agent_now():
    try:
        from agent import run_agent
        run_agent()
        return jsonify({"message": "Agent cycle complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)