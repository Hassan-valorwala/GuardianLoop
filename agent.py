import pandas as pd
import numpy as np
import joblib
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from model import extract_features
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────
THRESHOLD = -0.09
ALERT_EMAIL = os.getenv("ALERT_EMAIL")
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD") 
SUBJECT = "🚨 GuardianLoop Alert — Unusual Activity Detected"

# ── Load model ────────────────────────────────────────────
def load_model():
    model = joblib.load("model/guardian_model.pkl")
    feature_cols = joblib.load("model/feature_cols.pkl")
    return model, feature_cols

# ── Log to file ───────────────────────────────────────────
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    os.makedirs("data", exist_ok=True)
    with open("data/agent_log.txt", "a") as f:
        f.write(entry + "\n")
        
def already_alerted(date):
    alert_log = "data/alert_history.txt"
    if not os.path.exists(alert_log):
        return False
    with open(alert_log, "r") as f:
        alerted_dates = f.read().splitlines()
    return str(date) in alerted_dates

def mark_alerted(date):
    alert_log = "data/alert_history.txt"
    with open(alert_log, "a") as f:
        f.write(str(date) + "\n")        

# ── Send email alert ──────────────────────────────────────
def send_alert(score, date):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = ALERT_EMAIL
        msg["Subject"] = SUBJECT

        body = f"""
GuardianLoop Anomaly Alert

Date: {date}
Anomaly Score: {score:.4f}
Status: UNUSUAL ACTIVITY DETECTED

The autonomous agent has detected a significant deviation
from the normal daily routine.

Please check on your family member immediately.

— GuardianLoop Autonomous Care Agent
        """
        msg.attach(MIMEText(body, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, ALERT_EMAIL, msg.as_string())
        server.quit()

        log_event(f"EMAIL ALERT SENT to {ALERT_EMAIL}")
        return True

    except Exception as e:
        log_event(f"EMAIL FAILED: {str(e)}")
        return False

# ── Core agent logic ──────────────────────────────────────
def run_agent():
    log_event("Agent cycle started")

    try:
        # Load data
        df = pd.read_csv("data/activity_log.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Get today's date (last date in our simulated data)
        latest_date = df["timestamp"].dt.date.max()
        today_df = df[df["timestamp"].dt.date == latest_date]

        if today_df.empty:
            log_event("No activity data found for today")
            return

        log_event(f"Checking activity for {latest_date} — {len(today_df)} events recorded")

        # Extract features
        feature_df = extract_features(today_df)
        model, feature_cols = load_model()

        X = feature_df[feature_cols].values
        score = model.decision_function(X)[0]
        
        log_event(f"Anomaly score: {score:.4f}")

        # Decision
        if score < THRESHOLD:
            log_event(f"ANOMALY DETECTED — score {score:.4f} below threshold {THRESHOLD}")
            if not already_alerted(latest_date):
                sent = send_alert(score, latest_date)
                if sent:
                    mark_alerted(latest_date)
            else:
                log_event("Alert already sent for today — skipping email")
        else:
                log_event(f"Status NORMAL — routine appears healthy")

    except Exception as e:
        log_event(f"Agent error: {str(e)}")
        
def check_only():
    """
    Check activity and return status — no email sending.
    Used by the dashboard Check Now button.
    """
    log_event("Manual check triggered")
    try:
        df = pd.read_csv("data/activity_log.csv")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        latest_date = df["timestamp"].dt.date.max()
        today_df = df[df["timestamp"].dt.date == latest_date]

        if today_df.empty:
            log_event("No activity data found for today")
            return

        log_event(f"Checking activity for {latest_date} — {len(today_df)} events recorded")

        feature_df = extract_features(today_df)
        model, feature_cols = load_model()
        X = feature_df[feature_cols].values
        score = model.decision_function(X)[0]

        log_event(f"Anomaly score: {score:.4f}")

        if score < THRESHOLD:
            log_event("ANOMALY DETECTED — routine looks unusual")
        else:
            log_event("Status NORMAL — routine appears healthy")

    except Exception as e:
        log_event(f"Agent error: {str(e)}")        

# ── Run once immediately ──────────────────────────────────
if __name__ == "__main__":
    log_event("GuardianLoop Agent starting...")
    run_agent()
    log_event("Agent cycle complete")
    
    