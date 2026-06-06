import sqlite3
import os
from pathlib import Path
from passlib.context import CryptContext
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "auth.db"

# Ensure data dir exists
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Create Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            county TEXT,
            hashed_password TEXT NOT NULL,
            is_verified INTEGER DEFAULT 0,
            verification_code TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Try to add the new columns if the table already existed from a previous session
    try:
        c.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
        c.execute("ALTER TABLE users ADD COLUMN verification_code TEXT")
    except sqlite3.OperationalError:
        # Columns probably already exist
        pass
        
    # Create Activity Logs table
    c.execute('''
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS answer_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            rating TEXT NOT NULL CHECK(rating IN ('up', 'down')),
            sources_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

def create_user(name: str, email: str, county: str, password: str, verification_code: str):
    conn = get_db_connection()
    c = conn.cursor()
    hashed_pw = get_password_hash(password)
    try:
        c.execute(
            "INSERT INTO users (name, email, county, hashed_password, verification_code, is_verified) VALUES (?, ?, ?, ?, ?, 0)",
            (name, email, county, hashed_pw, verification_code)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(email: str, code: str) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE email = ? AND verification_code = ? AND is_verified = 0", (email, code))
    user = c.fetchone()
    if user:
        c.execute("UPDATE users SET is_verified = 1 WHERE id = ?", (user["id"],))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def update_verification_code(email: str, new_code: str) -> bool:
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("UPDATE users SET verification_code = ? WHERE email = ? AND is_verified = 0", (new_code, email))
    success = c.rowcount > 0
    conn.commit()
    conn.close()
    return success

def get_user_by_email(email: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user

def log_activity(user_id: int, action: str, details: str = None):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO activity_logs (user_id, action, details) VALUES (?, ?, ?)",
        (user_id, action, details)
    )
    conn.commit()
    conn.close()

def log_feedback(user_id: int, question: str, answer: str, rating: str, sources_json: str = None):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO answer_feedback (user_id, question, answer, rating, sources_json) VALUES (?, ?, ?, ?, ?)",
        (user_id, question, answer, rating, sources_json)
    )
    conn.commit()
    conn.close()

# Initialize database on module import
init_db()
