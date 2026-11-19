"""Persist Step 6 rejections and Step 8 technician outcomes in maintenance history."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def sync_history(conn, identity=None):
    """Copy eligible source events inside the caller's transaction."""
    conn.execute("""CREATE TABLE IF NOT EXISTS maintenance_history (
        source_key TEXT PRIMARY KEY, investigation_id TEXT NOT NULL, tyre_id TEXT NOT NULL,
        event_id TEXT NOT NULL, status TEXT NOT NULL, recorded_by TEXT NOT NULL,
        recorded_at TEXT NOT NULL, rejection_reason TEXT, actual_findings TEXT,
        maintenance_performed INTEGER, maintenance TEXT)""")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    params = (identity, identity)
    if {'fleet_manager_reviews', 'phase4_inbox'} <= tables:
        conn.execute("""INSERT INTO maintenance_history
            (source_key,investigation_id,tyre_id,event_id,status,recorded_by,recorded_at,rejection_reason)
            SELECT 'review:'||r.investigation_id,r.investigation_id,json_extract(p.payload_json,'$.tyre_id'),
                   json_extract(p.payload_json,'$.event_id'),r.decision,r.reviewer,r.reviewed_at,r.note
            FROM fleet_manager_reviews r JOIN phase4_inbox p USING(investigation_id,payload_sha256)
            WHERE r.decision='REJECTED' AND (? IS NULL OR r.investigation_id=?)
            ON CONFLICT(source_key) DO NOTHING""", params)
    if {'technician_completion_history', 'technician_service_records', 'approved_work_orders'} <= tables:
        conn.execute("""INSERT INTO maintenance_history
            (source_key,investigation_id,tyre_id,event_id,status,recorded_by,recorded_at,
             actual_findings,maintenance_performed,maintenance)
            SELECT 'completion:'||c.id,c.investigation_id,a.tyre_id,a.event_id,c.status,c.recorded_by,
                   c.recorded_at,c.actual_findings,s.maintenance_performed,s.maintenance
            FROM technician_completion_history c JOIN technician_service_records s USING(investigation_id)
            JOIN approved_work_orders a USING(investigation_id,payload_sha256)
            WHERE (? IS NULL OR c.investigation_id=?) ON CONFLICT(source_key) DO NOTHING""", params)


def maintenance_history(database=Path('data/tyremind.db'), tyre_id=None):
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN IMMEDIATE')
        sync_history(conn)
        return [dict(row) for row in conn.execute("""SELECT * FROM maintenance_history
            WHERE (? IS NULL OR tyre_id=?) ORDER BY recorded_at,rowid""", (tyre_id, tyre_id))]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--tyre-id')
    print(json.dumps(maintenance_history(**vars(parser.parse_args())), indent=2))
