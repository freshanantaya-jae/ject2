import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quality_inspection.db')

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            status TEXT NOT NULL,
            scratch_detected INTEGER NOT NULL,
            hole_count INTEGER NOT NULL,
            hole_sizes TEXT NOT NULL,         -- JSON array of float sizes (mm)
            classification TEXT NOT NULL,     -- JSON array of classes (e.g. ["M3", "M4"])
            fail_reason TEXT                  -- Reason for failure (if any)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized at: {DB_PATH}")

def insert_record(status, scratch_detected, hole_count, hole_sizes, classification, fail_reason=None):
    """Inserts a new quality inspection record.
    
    Args:
        status (str): 'PASS' or 'FAIL'
        scratch_detected (bool): True if scratch was found, False otherwise
        hole_count (int): Number of detected holes
        hole_sizes (list): List of measured diameters (float)
        classification (list): List of classified standards (e.g. ['M3', 'M4', 'Invalid'])
        fail_reason (str, optional): Explanation for failures.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO inspections (status, scratch_detected, hole_count, hole_sizes, classification, fail_reason)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        status, 
        1 if scratch_detected else 0, 
        hole_count, 
        json.dumps(hole_sizes), 
        json.dumps(classification), 
        fail_reason
    ))
    conn.commit()
    conn.close()

def get_recent_records(limit=10):
    """Retrieves the most recent inspection records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM inspections ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    
    records = []
    for r in rows:
        records.append({
            'id': r['id'],
            'timestamp': r['timestamp'],
            'status': r['status'],
            'scratch_detected': bool(r['scratch_detected']),
            'hole_count': r['hole_count'],
            'hole_sizes': json.loads(r['hole_sizes']),
            'classification': json.loads(r['classification']),
            'fail_reason': r['fail_reason']
        })
    conn.close()
    return records

def get_summary_stats():
    """Calculates statistics of pass/fail inspections."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM inspections')
    total = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE status = 'PASS'")
    passed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE status = 'FAIL'")
    failed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM inspections WHERE scratch_detected = 1")
    scratched = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'scratch_detected': scratched,
        'pass_rate': (passed / total * 100) if total > 0 else 0.0
    }

if __name__ == '__main__':
    # Test initialization
    init_db()
    
    # Test insertion
    insert_record(
        status='PASS',
        scratch_detected=False,
        hole_count=3,
        hole_sizes=[3.1, 4.0, 5.9],
        classification=['M3', 'M4', 'M6']
    )
    
    insert_record(
        status='FAIL',
        scratch_detected=True,
        hole_count=2,
        hole_sizes=[3.2, 8.5],
        classification=['M3', 'Invalid'],
        fail_reason='Scratch detected & Invalid hole size (8.5mm)'
    )
    
    print("Stats:", get_summary_stats())
    print("Recent:", get_recent_records(2))
