"""Record technician actions against a Step 5 approved order."""
import sqlite3
from contextlib import closing
from pathlib import Path


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
