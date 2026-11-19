"""Record the manager's decision against the exact published draft."""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from tyremind.phase4.history import sync_history

DECISIONS = {'APPROVED': 'Approve draft', 'CHANGES_REQUESTED': 'Request changes',
             'REJECTED': 'Reject draft'}


def record_review(identity, expected_hash, reviewer, decision, note='', database=Path('data/tyremind.db')):
    reviewer, note = reviewer.strip(), note.strip()
    if not reviewer or decision not in DECISIONS:
        raise ValueError('Enter your name and select a decision.')
    if decision != 'APPROVED' and not note:
        raise ValueError('Explain the changes needed or the reason for rejection.')
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=rw', uri=True)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT message_type,payload_json,payload_sha256 FROM phase4_inbox WHERE investigation_id=?',
                           (identity,)).fetchone()
        if not row or hashlib.sha256(row[1].encode()).hexdigest() != row[2] or row[2] != expected_hash:
            raise ValueError('The published draft changed. Refresh before reviewing.')
        packet = json.loads(row[1])
        if (row[0] != 'INVESTIGATION_READY_FOR_REVIEW' or packet['investigation_id'] != identity
                or packet['step'] != 11 or packet['status'] != 'AWAITING_HUMAN_REVIEW'
                or packet['validation']['stage'] != 'AWAITING_HUMAN_REVIEW' or packet['validation']['findings']
                or packet.get('work_order', {}).get('status') != 'DRAFT'):
            raise ValueError('Only a validated, published draft can receive a manager decision.')
        conn.execute("""CREATE TABLE IF NOT EXISTS fleet_manager_reviews (
            investigation_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, reviewer TEXT NOT NULL,
            decision TEXT NOT NULL, note TEXT NOT NULL, reviewed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        previous = conn.execute('SELECT payload_sha256,reviewer,decision,note FROM fleet_manager_reviews WHERE investigation_id=?',
                                (identity,)).fetchone()
        values = (expected_hash, reviewer, decision, note)
        if previous is None:
            conn.execute('INSERT INTO fleet_manager_reviews (investigation_id,payload_sha256,reviewer,decision,note) VALUES (?,?,?,?,?)',
                         (identity, *values))
        elif previous != values:
            raise ValueError('A decision has already been recorded. Refresh to view it.')

        if decision == 'APPROVED':
            conn.execute("""CREATE TABLE IF NOT EXISTS approved_work_orders (
                investigation_id TEXT PRIMARY KEY, tyre_id TEXT NOT NULL, event_id TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL, instructions TEXT NOT NULL, approved_by TEXT NOT NULL,
                approved_at TEXT NOT NULL)""")
            conn.execute("""INSERT INTO approved_work_orders
                SELECT investigation_id, ?, ?, payload_sha256, ?, reviewer, reviewed_at
                FROM fleet_manager_reviews WHERE investigation_id=?
                ON CONFLICT(investigation_id) DO NOTHING""",
                (packet['tyre_id'], packet['event_id'], packet['work_order']['instructions'], identity))
        if decision == 'REJECTED':
            sync_history(conn, identity)
