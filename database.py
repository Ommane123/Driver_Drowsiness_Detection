import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drowsiness_logs.db")

def init_db():
    """Initializes the database and creates the events table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            duration REAL,
            max_risk_score REAL,
            avg_ear REAL,
            avg_mar REAL,
            avg_head_yaw REAL,
            avg_head_pitch REAL
        );
    """)
    conn.commit()
    conn.close()

def log_event(event_type, duration, max_risk_score, avg_ear, avg_mar, avg_head_yaw, avg_head_pitch):
    """Logs a new event to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO events (event_type, duration, max_risk_score, avg_ear, avg_mar, avg_head_yaw, avg_head_pitch)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (event_type, duration, max_risk_score, avg_ear, avg_mar, avg_head_yaw, avg_head_pitch))
    conn.commit()
    conn.close()
    print(f"[DB LOG] Saved event: {event_type} (Duration: {duration:.2f}s, Max Risk: {max_risk_score:.1f})")

def fetch_recent_events(limit=50):
    """Fetches recent events sorted by timestamp in descending order."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, timestamp, event_type, duration, max_risk_score FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# Automatically initialize database when database.py is imported or run
init_db()
