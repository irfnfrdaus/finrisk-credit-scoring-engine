"""
FinRisk-Engine: SQLite Audit & Compliance Database
Maintains an immutable audit log of all automated credit scoring decisions,
essential for regulatory compliance (Central Bank / Bank Negara Malaysia & Basel guidelines).
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credit_audit.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS loan_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        applicant_name TEXT NOT NULL,
        person_age INTEGER NOT NULL,
        person_income REAL NOT NULL,
        loan_amnt REAL NOT NULL,
        loan_intent TEXT NOT NULL,
        credit_score INTEGER NOT NULL,
        dti_ratio REAL NOT NULL,
        probability_of_default REAL NOT NULL,
        risk_tier TEXT NOT NULL,
        decision TEXT NOT NULL,
        notes TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_evaluation(
    applicant_name: str, 
    applicant_input: Dict[str, Any], 
    eval_output: Dict[str, Any], 
    notes: str = ""
) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
    INSERT INTO loan_evaluations (
        timestamp, applicant_name, person_age, person_income, loan_amnt,
        loan_intent, credit_score, dti_ratio, probability_of_default,
        risk_tier, decision, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        applicant_name,
        int(applicant_input.get('person_age', 30)),
        float(applicant_input.get('person_income', 0.0)),
        float(applicant_input.get('loan_amnt', 0.0)),
        str(applicant_input.get('loan_intent', 'PERSONAL')),
        int(applicant_input.get('credit_score', 650)),
        float(eval_output.get('dti_ratio', 0.0)),
        float(eval_output.get('probability_of_default', 0.0)),
        str(eval_output.get('risk_tier', 'Unknown')),
        str(eval_output.get('decision', 'PENDING')),
        notes
    ))
    
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def get_recent_evaluations(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM loan_evaluations ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    results = [dict(r) for r in rows]
    conn.close()
    return results

def get_portfolio_stats() -> Dict[str, Any]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM loan_evaluations")
    total_count = cursor.fetchone()[0]
    
    if total_count == 0:
        conn.close()
        return {
            'total_applications': 0,
            'approved_count': 0,
            'review_count': 0,
            'rejected_count': 0,
            'avg_pd': 0.0,
            'total_loan_volume': 0.0
        }
        
    cursor.execute("SELECT decision, COUNT(*) FROM loan_evaluations GROUP BY decision")
    decision_counts = dict(cursor.fetchall())
    
    cursor.execute("SELECT AVG(probability_of_default), SUM(loan_amnt) FROM loan_evaluations")
    avg_pd, total_vol = cursor.fetchone()
    
    conn.close()
    return {
        'total_applications': total_count,
        'approved_count': decision_counts.get('APPROVED', 0),
        'review_count': decision_counts.get('MANUAL_REVIEW', 0),
        'rejected_count': decision_counts.get('REJECTED', 0),
        'avg_pd': round(avg_pd or 0.0, 4),
        'total_loan_volume': round(total_vol or 0.0, 2)
    }

if __name__ == '__main__':
    init_db()
    print("Audit database initialized successfully at:", DB_PATH)
