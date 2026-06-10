from flask import Flask, render_template, jsonify
import pandas as pd
from model import extract_features
import joblib
import os
from dotenv import load_dotenv
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

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
        from agent import check_only
        check_only()
        return jsonify({"message": "Agent cycle complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/demo/normal")
def demo_normal():
    try:
        import subprocess
        # Set today as normal
        with open("data/demo_mode.txt", "w") as f:
            f.write("normal")
        from data.simulate_data import generate_normal_day, generate_anomalous_day
        import pandas as pd
        df = pd.read_csv("data/activity_log.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Remove any existing June 10/11 rows
        df = df[~df["timestamp"].dt.strftime("%Y-%m-%d").isin(["2026-06-10", "2026-06-11"])]
        today = pd.Timestamp("2026-06-10")
        records = generate_normal_day(today)
        today_df = pd.DataFrame(records)
        today_df["day_type"] = "normal"
        df = pd.concat([df, today_df], ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv("data/activity_log.csv", index=False)
        # Retrain
        from model import extract_features, train_model, save_model
        feature_df = extract_features(df)
        normal_df = feature_df.head(27)
        model, feature_cols = train_model(normal_df)
        save_model(model, feature_cols)
        return jsonify({"message": "Demo set to NORMAL"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/demo/anomaly")
def demo_anomaly():
    try:
        from data.simulate_data import generate_anomalous_day
        import pandas as pd
        df = pd.read_csv("data/activity_log.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Remove any existing June 10/11 rows
        df = df[~df["timestamp"].dt.strftime("%Y-%m-%d").isin(["2026-06-10", "2026-06-11"])]
        today = pd.Timestamp("2026-06-11")
        records = generate_anomalous_day(today)
        today_df = pd.DataFrame(records)
        today_df["day_type"] = "anomalous"
        df = pd.concat([df, today_df], ignore_index=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv("data/activity_log.csv", index=False)
        # Retrain
        from model import extract_features, train_model, save_model
        feature_df = extract_features(df)
        normal_df = feature_df.head(27)
        model, feature_cols = train_model(normal_df)
        save_model(model, feature_cols)
        return jsonify({"message": "Demo set to ANOMALY"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500    
    
# ── Scheduler ─────────────────────────────────────────────
def scheduled_agent():
    from agent import run_agent
    run_agent()

scheduler = BackgroundScheduler()
scheduler.add_job(scheduled_agent, 'interval', hours=1)
scheduler.start()    

if __name__ == "__main__":
    app.run(debug=True)