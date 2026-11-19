"""Read the latest assessment per tyre and published investigation outcomes."""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3


def read_results(database=Path('data/tyremind.db'), tyre_id=None):
    uri = Path(database).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN')
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'risk_assessments' not in tables:
            raise ValueError('Phase 2 monitoring assessments are missing. Run the monitoring worker first.')
        rows = connection.execute("""WITH ranked AS (
            SELECT a.*,row_number() OVER (PARTITION BY tyre_id ORDER BY feature_as_of DESC,
                assessed_at DESC, rowid DESC) AS rn FROM risk_assessments a
            WHERE (? IS NULL OR tyre_id=?))
            SELECT r.assessment_id,r.tyre_id,r.truck_id,r.timestamp,r.feature_as_of,r.risk_level,
                r.data_status,r.rules_json,r.data_issues_json,r.evidence_json,r.prediction_id,d.next_action
            FROM ranked r LEFT JOIN monitoring_decisions d USING(assessment_id)
            WHERE rn=1 ORDER BY r.tyre_id""", (tyre_id, tyre_id))
        monitoring = []
        for row in rows:
            assessment = dict(row)
            for field in ('rules', 'data_issues', 'evidence'):
                assessment[field] = json.loads(assessment.pop(field + '_json'))
            monitoring.append(assessment)
        investigations = []
        if 'phase4_inbox' in tables:
            for row in connection.execute('SELECT * FROM phase4_inbox ORDER BY published_at,investigation_id'):
                payload = row['payload_json']
                if hashlib.sha256(payload.encode()).hexdigest() != row['payload_sha256']:
                    raise ValueError(f"Invalid Phase 4 payload checksum for {row['investigation_id']}")
                outcome = json.loads(payload)
                if outcome['investigation_id'] != row['investigation_id']:
                    raise ValueError('Phase 4 message identity mismatch.')
                if tyre_id is None or outcome['tyre_id'] == tyre_id:
                    investigations.append(dict(message_type=row['message_type'], inbox_status=row['status'],
                                               published_at=row['published_at'], outcome=outcome))
    return dict(monitoring_count=len(monitoring), investigation_count=len(investigations),
                monitoring=monitoring, investigations=investigations)


def read_tyre_history(tyre_id, cutoff, database=Path('data/tyremind.db')):
    """Return observed history through the selected assessment's evidence cutoff."""
    queries = {
        'telemetry': """SELECT timestamp, pressure_kpa, temperature_c, load_ratio FROM telemetry
                        WHERE tyre_id=? AND timestamp<=? ORDER BY timestamp""",
        'inspections': """SELECT timestamp, tread_depth_mm FROM inspections
                          WHERE tyre_id=? AND timestamp<=? ORDER BY timestamp""",
        'risk': """SELECT feature_as_of AS timestamp, risk_level, data_status FROM risk_assessments
                   WHERE tyre_id=? AND feature_as_of<=? ORDER BY feature_as_of, assessed_at, rowid"""
    }
    with closing(sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute('BEGIN')
        return {name: [dict(row) for row in connection.execute(query, (tyre_id, cutoff))]
                for name, query in queries.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--tyre-id')
    args = parser.parse_args()
    print(json.dumps(read_results(args.database, args.tyre_id), indent=2))
