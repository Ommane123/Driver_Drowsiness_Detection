import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drowsiness_logs.db")

def init_db():
    """Initializes the database and creates the events and driver_photos tables if they do not exist."""
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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS driver_photos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            photo BLOB,
            distractions INTEGER DEFAULT 0
        );
    """)
    try:
        cursor.execute("ALTER TABLE driver_photos ADD COLUMN distractions INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
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

def save_driver_photo(photo_bytes):
    """Saves a driver's photo in the database, keeps only the 5 most recent, and returns the inserted row ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check current count of photos
    cursor.execute("SELECT id FROM driver_photos ORDER BY timestamp ASC")
    rows = cursor.fetchall()
    
    # If there are 5 or more, delete the oldest ones to make space for the new one (so total is at most 5)
    if len(rows) >= 5:
        to_delete = len(rows) - 4
        for i in range(to_delete):
            cursor.execute("DELETE FROM driver_photos WHERE id = ?", (rows[i][0],))
            
    # Insert new photo
    cursor.execute("""
        INSERT INTO driver_photos (photo, distractions)
        VALUES (?, 0)
    """, (sqlite3.Binary(photo_bytes),))
    inserted_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    print(f"[DB LOG] Saved driver photo. ID: {inserted_id}. Total count is capped at 5.")
    return inserted_id

def update_driver_distractions(row_id, count):
    """Updates the distractions counter for a specific driver photo row."""
    if row_id is None:
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE driver_photos
        SET distractions = ?
        WHERE id = ?
    """, (count, row_id))
    conn.commit()
    conn.close()
    print(f"[DB LOG] Updated distractions count to {count} for photo ID {row_id}")

def fetch_last_driver_photo_and_distractions():
    """Fetches the latest driver photo and distractions count from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT photo, distractions FROM driver_photos ORDER BY timestamp DESC LIMIT 1")
        row = cursor.fetchone()
        result = (row[0], row[1]) if row else (None, 0)
    except sqlite3.OperationalError:
        result = (None, 0)
    conn.close()
    return result

# Automatically initialize database when database.py is imported or run
init_db()
