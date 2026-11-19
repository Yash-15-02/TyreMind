"""Load operational evidence for the investigation selected by the supervisor."""
import argparse
import json
from pathlib import Path

import pandas as pd

from tyremind.agents.supervisor import SupervisorCheckpoint, digest, export_json, start_investigation
from tyremind.monitoring.events import transaction


def load_tyre_history(connection, tyre_id, cutoff):
    history = {}
    for table, tie_breaker in [('telemetry', 'truck_id'), ('inspections', 'inspection_id'),
                               ('maintenance_events', 'event_id')]:
        # Table and column names are fixed here. All event values are bound parameters.
        rows = connection.execute(
            f'SELECT * FROM {table} WHERE tyre_id=? AND julianday(timestamp)<=julianday(?) '
            f'ORDER BY julianday(timestamp), {tie_breaker}', (tyre_id, cutoff))
        history[table] = [dict(row) for row in rows if pd.Timestamp(row['timestamp']) <= pd.Timestamp(cutoff)]
    return history


def load_ml_results(connection, investigation):
    assessment = investigation.source_handoff['assessment']
    if assessment['prediction_id'] is None:
        return None
    row = connection.execute('SELECT * FROM predictions WHERE prediction_id=? AND tyre_id=? AND truck_id=?',
                             (assessment['prediction_id'], investigation.tyre_id, investigation.truck_id)).fetchone()
    if (row is None or pd.Timestamp(row['feature_as_of']) != pd.Timestamp(assessment['feature_as_of'])
            or pd.Timestamp(row['timestamp']) != pd.Timestamp(assessment['timestamp'])):
        raise ValueError('Referenced prediction is missing or differs from the investigation snapshot.')
    result = dict(row)
    result['input_features'] = json.loads(result.pop('input_features_json'))
    return result


def load_evidence(database=Path('data/tyremind.db'), consumer_id='local-phase3',
                  phase3_output=Path('data/phase3')):
    root = Path(phase3_output)
    report = start_investigation(database, root/'supervisor', consumer_id, root)
    if report['action'] == 'NO_WORK':
        return {'action': 'NO_WORK'}
    identity = report['investigation_id']
    with transaction(database) as connection:
        checkpoint = connection.execute("""SELECT s.* FROM phase3_supervisor_states s
            JOIN phase3_receipts r USING (investigation_id) JOIN investigations j USING (investigation_id)
            WHERE s.investigation_id=? AND r.consumer_id=? AND j.status='RUNNING'
            AND j.claim_token=r.claim_token AND s.receipt_sha256=r.state_sha256""", (identity, consumer_id)).fetchone()
        if checkpoint is None or digest(checkpoint['state_json']) != checkpoint['state_sha256']:
            raise ValueError('Supervisor checkpoint or investigation ownership changed.')
        state = SupervisorCheckpoint.model_validate_json(checkpoint['state_json'])
        case = state.investigation
        cutoff = case.source_handoff['assessment']['feature_as_of']
        connection.execute("""CREATE TABLE IF NOT EXISTS phase3_evidence (
            investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
            supervisor_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL)""")
        saved = connection.execute('SELECT * FROM phase3_evidence WHERE investigation_id=?', (identity,)).fetchone()
        if saved:
            if saved['supervisor_sha256'] != checkpoint['state_sha256'] or digest(saved['payload_json']) != saved['payload_sha256']:
                raise ValueError('Saved evidence no longer matches its checkpoint.')
            evidence = json.loads(saved['payload_json'])
            action = 'RESUMED'
        else:
            evidence = dict(investigation_id=identity, tyre_id=case.tyre_id, truck_id=case.truck_id,
                            assessment_id=case.source_handoff['assessment']['assessment_id'],
                            cutoff=cutoff, next_task=state.next_task,
                            history=load_tyre_history(connection, case.tyre_id, cutoff),
                            prediction=load_ml_results(connection, case))
            payload = json.dumps(evidence, sort_keys=True, allow_nan=False)
            connection.execute('INSERT INTO phase3_evidence VALUES (?,?,?,?)',
                               (identity, checkpoint['state_sha256'], payload, digest(payload)))
            action = 'LOADED'
    path = root/'evidence'/f'{identity}.json'
    export_json(path, evidence)
    return dict(action=action, tyre_id=evidence['tyre_id'], cutoff=evidence['cutoff'],
                row_counts={name: len(rows) for name, rows in evidence['history'].items()},
                prediction_status=(evidence['prediction'] or {}).get('status', 'no_prediction_reference'),
                next_task=evidence['next_task'], evidence_path=str(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--consumer-id', default='local-phase3')
    args = parser.parse_args()
    print(json.dumps(load_evidence(args.database, args.consumer_id), indent=2))
