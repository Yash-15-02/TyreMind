"""Record technician actions against a Step 5 approved order."""
import sqlite3
from contextlib import closing
from pathlib import Path

from tyremind.phase4.history import sync_history


def record_service(identity, expected_hash, technician, findings, performed, maintenance,
                   database=Path('data/tyremind.db')):
    technician, findings, maintenance = technician.strip(), findings.strip(), maintenance.strip()
    if not technician or not findings or type(performed) is not bool:
        raise ValueError('Enter a technician name, inspection findings and maintenance outcome.')
    if performed != bool(maintenance):
        raise ValueError('Describe the authorized maintenance only when work was performed.')
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='approved_work_orders'").fetchone():
            raise ValueError('This action requires a Step 5 approved order.')
        approved = conn.execute("""SELECT 1 FROM approved_work_orders a JOIN fleet_manager_reviews r
            USING (investigation_id,payload_sha256) WHERE investigation_id=? AND payload_sha256=?
            AND r.decision='APPROVED'""", (identity, expected_hash)).fetchone()
        if not approved:
            raise ValueError('This action requires the matching Step 5 approved order.')
        conn.execute("""CREATE TABLE IF NOT EXISTS technician_service_records (
            investigation_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, technician TEXT NOT NULL,
            findings TEXT NOT NULL, maintenance_performed INTEGER NOT NULL, maintenance TEXT NOT NULL,
            recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        values = (expected_hash, technician, findings, int(performed), maintenance)
        previous = conn.execute("""SELECT payload_sha256,technician,findings,maintenance_performed,maintenance
            FROM technician_service_records WHERE investigation_id=?""", (identity,)).fetchone()
        if previous is None:
            conn.execute("""INSERT INTO technician_service_records (investigation_id,payload_sha256,
                technician,findings,maintenance_performed,maintenance) VALUES (?,?,?,?,?,?)""", (identity, *values))
        elif previous != values:
            raise ValueError('This order already has a technician record. Refresh to view it.')


def record_completion(identity, expected_hash, actual_findings, status, recorded_by,
                      previous_id=None, database=Path('data/tyremind.db')):
    actual_findings, recorded_by = actual_findings.strip(), recorded_by.strip()
    if not actual_findings or not recorded_by or status not in ('COMPLETED', 'FOLLOW_UP_REQUIRED'):
        raise ValueError('Enter actual findings, your name and a completion status.')
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='technician_service_records'").fetchone()
        if not exists or not conn.execute("""SELECT 1 FROM technician_service_records
                WHERE investigation_id=? AND payload_sha256=?""", (identity, expected_hash)).fetchone():
            raise ValueError('Save the matching Step 7 technician record first.')
        conn.execute("""CREATE TABLE IF NOT EXISTS technician_completion_history (
            id INTEGER PRIMARY KEY, investigation_id TEXT NOT NULL, actual_findings TEXT NOT NULL,
            status TEXT NOT NULL, recorded_by TEXT NOT NULL, recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        latest = conn.execute("""SELECT id,actual_findings,status,recorded_by FROM technician_completion_history
            WHERE investigation_id=? ORDER BY id DESC LIMIT 1""", (identity,)).fetchone()
        values = (actual_findings, status, recorded_by)
        if latest and latest[1:] == values:
            return
        if latest and latest[2] == 'COMPLETED':
            raise ValueError('This work order already has a completed outcome.')
        if (latest[0] if latest else None) != previous_id:
            raise ValueError('The outcome changed. Refresh before submitting.')
        conn.execute("""INSERT INTO technician_completion_history
            (investigation_id,actual_findings,status,recorded_by) VALUES (?,?,?,?)""", (identity, *values))
        sync_history(conn, identity)
