---
title: GuardianLoop
emoji: 🔒
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
---
# GuardianLoop

Autonomous behavioral anomaly detection for elderly people living alone.

GuardianLoop learns what a normal day looks like for a person — when they wake up, visit the kitchen, rest, move through their home — and autonomously detects when something is wrong. No human triggers the check. The agent runs on its own, decides on its own, and alerts family when a pattern deviates significantly from the norm.

---

## The Problem

Millions of elderly people live alone. Their families cannot watch them around the clock. The danger is rarely dramatic — it is subtle. A person who always makes tea at 8am did not today. Nobody noticed until evening.

Existing solutions are either expensive hardware or simple rule-based timers. A timer is not intelligence. GuardianLoop learns the individual's actual routine and flags deviations from *their* normal — not a generic threshold.

---

## How It Works

```
Simulated activity data
        ↓
Feature engineering (activity logs → numerical daily profile)
        ↓
Isolation Forest (learns normal behavioral patterns)
        ↓
Autonomous agent loop (perceive → score → decide → act)
        ↓
Dashboard update + email alert to family
```

**The agent loop runs automatically every hour.** It does not wait for a human to press a button. If today's activity pattern scores below the learned threshold, it sends one email alert to the designated family contact — and only one, no matter how many times it checks that day.

---

## Features

- **Behavioral learning** — Isolation Forest trained on 27 days of normal activity to learn individual routine
- **Feature engineering** — raw activity logs converted to 9 numerical features per day (first activity hour, last activity hour, total events, average gap, max inactive gap, kitchen visits, bedroom visits, living room visits, active span)
- **Autonomous agent** — runs on a scheduler every hour, no human intervention required
- **Smart alerting** — email sent once per anomalous day, duplicate suppression via alert history
- **Live dashboard** — real-time status, 7-day activity strip, human-readable agent log
- **Demo controls** — simulate normal or anomalous day live without touching the terminal

---

## Tech Stack

| Layer | Tool |
|---|---|
| Data simulation | Python, Pandas, NumPy |
| Anomaly detection | Scikit-learn (Isolation Forest) |
| Autonomous loop | APScheduler |
| Backend | Flask |
| Alerting | Gmail SMTP + python-dotenv |
| Frontend | HTML, CSS, JavaScript |

---

## Project Structure

```
GuardianLoop/
├── app.py                  # Flask backend, API routes, scheduler
├── agent.py                # Autonomous agent logic, email alerts
├── model.py                # Feature engineering, Isolation Forest
├── data/
│   ├── simulate_data.py    # Activity data simulator
│   ├── activity_log.csv    # Generated activity dataset
│   └── agent_log.txt       # Agent decision log
├── model/
│   ├── guardian_model.pkl  # Trained Isolation Forest
│   └── feature_cols.pkl    # Feature column names
├── templates/
│   └── index.html          # Dashboard UI
├── static/
│   └── style.css           # Styles
├── .env                    # Email credentials (not committed)
├── .gitignore
└── requirements.txt
```

---

## Setup & Run

**1. Clone the repository**

```bash
git clone https://github.com/Hassan-valorwala/GuardianLoop.git
cd GuardianLoop
```

**2. Create virtual environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure email alerts**

Create a `.env` file in the project root:

```
SENDER_EMAIL=your_gmail@gmail.com
SENDER_PASSWORD=your_16_char_app_password
ALERT_EMAIL=family_member@gmail.com
```

To generate a Gmail App Password: Google Account → Security → 2-Step Verification → App Passwords.

**5. Generate data and train model**

```bash
python data/simulate_data.py
python model.py
```

**6. Run the application**

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Data Approach

GuardianLoop uses simulated activity data for the prototype. Each day is represented as a sequence of timestamped activity events — wake up, kitchen visits, living room presence, bedroom rest — with realistic timing variation drawn from a base daily routine.

The simulator generates 30 days of normal behavior and injects anomalous days with significantly reduced activity counts and irregular timing. In a production deployment, this data would come from passive sensors: motion detectors, smart plugs, wearables, or smartphone activity — no cameras required.

---

## What Makes an Anomaly

The agent does not use a simple rule like "no movement for 4 hours." It scores each day against a learned behavioral model.

A day is flagged when its numerical profile — fewer events than usual, missing kitchen visits, compressed active window, large inactive gaps — deviates enough from the trained baseline that the Isolation Forest scores it below the detection threshold of `-0.09`.

This means the system adapts to the individual. A person who naturally sleeps late will not be flagged for a late first activity. A person who visits the kitchen three times a day will be flagged if they only visit once.

---

## Theme

**Agentic & Autonomous Systems** — FAR AWAY International Hackathon 2026

GuardianLoop is not a dashboard that waits for input. It is an agent that perceives, decides, and acts in a continuous loop — autonomously, without human intervention at each step.

---

## Built By

Hassan Valorwala and Team — AIML student, FAR AWAY Hackathon 2026