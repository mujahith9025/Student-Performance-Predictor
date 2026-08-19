import os
import sys
import json
import sqlite3
from datetime import datetime
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "student_records.db")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def get_connection():
    """Returns a SQLite connection with timeout and foreign key support."""
    conn = sqlite3.connect(DB_PATH, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes database tables and indexes if they do not exist."""
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # 1. Predictions History Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                student_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                model_used TEXT NOT NULL,
                hours_studied REAL NOT NULL,
                previous_score REAL NOT NULL,
                sleep_hours REAL NOT NULL,
                sample_papers INTEGER NOT NULL,
                extracurricular TEXT NOT NULL,
                predicted_score REAL NOT NULL,
                lower_bound REAL NOT NULL,
                upper_bound REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                letter_grade TEXT NOT NULL,
                standing_desc TEXT,
                habit_balance_score REAL,
                study_to_sleep_ratio REAL,
                practice_density REAL,
                study_effort_score REAL,
                view_mode TEXT,
                teacher_notes TEXT,
                risk_flags TEXT,
                shap_summary TEXT
            )
        """)
        
        # 2. Target Goal Roadmaps Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS goal_targets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                student_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                current_pred REAL NOT NULL,
                target_score REAL NOT NULL,
                score_gap REAL NOT NULL,
                recommended_pathway TEXT NOT NULL,
                required_hours REAL NOT NULL,
                required_papers INTEGER NOT NULL,
                required_sleep REAL NOT NULL,
                weekly_hours REAL NOT NULL
            )
        """)
        
        # Indexes for fast search & filtering
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_student ON predictions(student_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_subject ON predictions(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goal_student ON goal_targets(student_name)")
        
        conn.commit()
    finally:
        conn.close()

def save_prediction_record(
    student_name: str,
    subject: str,
    model_used: str,
    hours_studied: float,
    previous_score: float,
    sleep_hours: float,
    sample_papers: int,
    extracurricular: str,
    predicted_score: float,
    lower_bound: float,
    upper_bound: float,
    confidence_level: str,
    letter_grade: str,
    standing_desc: str = "",
    habit_balance_score: float = 75.0,
    study_to_sleep_ratio: float = None,
    practice_density: float = None,
    study_effort_score: float = None,
    view_mode: str = "Student Mode",
    teacher_notes: str = "",
    risk_flags: list = None,
    shap_summary: dict = None
) -> int:
    """Saves a student prediction record into the database."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        risk_flags_json = json.dumps(risk_flags if risk_flags else [])
        shap_json = json.dumps(shap_summary if shap_summary else {})
        
        if study_to_sleep_ratio is None and sleep_hours > 0:
            study_to_sleep_ratio = round(hours_studied / sleep_hours, 2)
        if practice_density is None:
            practice_density = round(sample_papers / (hours_studied + 1.0), 2)
        if study_effort_score is None:
            study_effort_score = round((0.6 * hours_studied) + (0.4 * sample_papers), 2)
            
        cursor.execute("""
            INSERT INTO predictions (
                timestamp, student_name, subject, model_used,
                hours_studied, previous_score, sleep_hours, sample_papers, extracurricular,
                predicted_score, lower_bound, upper_bound, confidence_level,
                letter_grade, standing_desc, habit_balance_score,
                study_to_sleep_ratio, practice_density, study_effort_score,
                view_mode, teacher_notes, risk_flags, shap_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str, student_name.strip() or "Student", subject, model_used,
            float(hours_studied), float(previous_score), float(sleep_hours), int(sample_papers), str(extracurricular),
            float(predicted_score), float(lower_bound), float(upper_bound), str(confidence_level),
            str(letter_grade), str(standing_desc), float(habit_balance_score),
            float(study_to_sleep_ratio), float(practice_density), float(study_effort_score),
            str(view_mode), str(teacher_notes or ""), risk_flags_json, shap_json
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def save_goal_record(
    student_name: str,
    subject: str,
    current_pred: float,
    target_score: float,
    score_gap: float,
    recommended_pathway: str,
    required_hours: float,
    required_papers: int,
    required_sleep: float,
    weekly_hours: float
) -> int:
    """Saves a target score roadmap into the database."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO goal_targets (
                timestamp, student_name, subject, current_pred, target_score, score_gap,
                recommended_pathway, required_hours, required_papers, required_sleep, weekly_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_str, student_name.strip() or "Student", subject,
            float(current_pred), float(target_score), float(score_gap),
            str(recommended_pathway), float(required_hours), int(required_papers),
            float(required_sleep), float(weekly_hours)
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def fetch_prediction_history(student_name: str = None, subject: str = None, limit: int = 200) -> pd.DataFrame:
    """Fetches historical predictions as a Pandas DataFrame with optional filters."""
    init_db()
    conn = get_connection()
    try:
        query = "SELECT * FROM predictions"
        params = []
        conditions = []
        
        if student_name and student_name.strip() and student_name != "All Students":
            conditions.append("student_name LIKE ?")
            params.append(f"%{student_name.strip()}%")
            
        if subject and subject.strip() and subject != "All Subjects":
            conditions.append("subject = ?")
            params.append(subject.strip())
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

def fetch_goal_history(student_name: str = None, limit: int = 50) -> pd.DataFrame:
    """Fetches saved target goal roadmaps as a Pandas DataFrame."""
    init_db()
    conn = get_connection()
    try:
        query = "SELECT * FROM goal_targets"
        params = []
        if student_name and student_name.strip() and student_name != "All Students":
            query += " WHERE student_name LIKE ?"
            params.append(f"%{student_name.strip()}%")
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

def fetch_student_progress_timeline(student_name: str) -> pd.DataFrame:
    """Fetches chronological prediction progress for a single student for time-series charts."""
    init_db()
    conn = get_connection()
    try:
        query = """
            SELECT id, timestamp, subject, hours_studied, previous_score,
                   predicted_score, lower_bound, upper_bound, letter_grade, habit_balance_score
            FROM predictions
            WHERE student_name LIKE ?
            ORDER BY id ASC
        """
        return pd.read_sql_query(query, conn, params=[f"%{student_name.strip()}%"])
    finally:
        conn.close()

def delete_prediction_record(record_id: int) -> bool:
    """Deletes a single prediction record by ID."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()

def clear_all_records() -> bool:
    """Clears all records from predictions and goal_targets tables."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        cursor.execute("DELETE FROM goal_targets")
        conn.commit()
        return True
    finally:
        conn.close()

def get_database_stats() -> dict:
    """Returns high-level statistics about stored records."""
    init_db()
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), AVG(predicted_score), COUNT(DISTINCT student_name) FROM predictions")
        row = cursor.fetchone()
        
        total_preds = row[0] if row else 0
        avg_score = round(row[1], 1) if (row and row[1] is not None) else 0.0
        unique_students = row[2] if row else 0
        
        cursor.execute("SELECT COUNT(*) FROM goal_targets")
        total_goals = cursor.fetchone()[0]
        
        return {
            "total_predictions": total_preds,
            "avg_predicted_score": avg_score,
            "unique_students": unique_students,
            "total_saved_goals": total_goals,
            "db_path": DB_PATH
        }
    finally:
        conn.close()

# Auto-initialize on import
init_db()
