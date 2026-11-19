"""Run the Health Agent on the supervisor's frozen SQLite evidence."""
import argparse
import json
from pathlib import Path

from tyremind.agents.evidence import load_evidence
from tyremind.agents.llm import request_text
from tyremind.agents.supervisor import digest, export_json
from tyremind.monitoring.events import transaction


def assess_health(database=Path('data/tyremind.db'), consumer_id='local-phase3', root=Path('data/phase3')):
    report = load_evidence(database, consumer_id, root)
    if report['action'] == 'NO_WORK':
        return report
    if report['next_task'] != 'HEALTH_ASSESSMENT':
        return {'action': 'AWAITING_DATA_REVIEW', 'tyre_id': report['tyre_id']}
    identity = Path(report['evidence_path']).stem
    with transaction(database) as connection:
        row = connection.execute("""SELECT e.* FROM phase3_evidence e JOIN investigations j USING(investigation_id)
            JOIN phase3_receipts r USING(investigation_id) WHERE investigation_id=? AND j.status='RUNNING'
            AND r.consumer_id=? AND r.claim_token=j.claim_token""", (identity, consumer_id)).fetchone()
        if row is None or digest(row['payload_json']) != row['payload_sha256']:
            raise ValueError('Investigation ownership or evidence checksum changed.')
        evidence = json.loads(row['payload_json'])
        connection.execute("""CREATE TABLE IF NOT EXISTS phase3_health (
            investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
            evidence_sha256 TEXT NOT NULL, assessment TEXT NOT NULL)""")
        saved = connection.execute('SELECT * FROM phase3_health WHERE investigation_id=?', (identity,)).fetchone()
        if saved:
            if saved['evidence_sha256'] != row['payload_sha256']:
                raise ValueError('Stored assessment references different evidence.')
            assessment = saved['assessment']
        else:
            history = evidence['history']
            inputs = dict(cutoff=evidence['cutoff'], recent_readings=history['telemetry'][-7:],
                          recent_inspections=history['inspections'][-2:],
                          recent_maintenance=history['maintenance_events'][-3:], prediction=evidence['prediction'])
            prompt = ('Explain the measured tyre condition and trends across the dated readings. '
                      'Separate observed tread from the forecast at 5000 km. Explain missing data and '
                      'prediction uncertainty. An anomaly score is not a probability. Do not invent '
                      'confidence intervals, certify safety, or assert a root cause. Use only this evidence:\n'
                      + json.dumps(inputs, allow_nan=False))
            assessment = request_text('health_assessment', prompt)
            connection.execute('INSERT INTO phase3_health VALUES (?,?,?)',
                               (identity, row['payload_sha256'], assessment))
    result = dict(action='RESUMED' if saved else 'ASSESSED', investigation_id=identity,
                  tyre_id=evidence['tyre_id'], cutoff=evidence['cutoff'], assessment=assessment)
    export_json(root/'health'/f'{identity}.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--consumer-id', default='local-phase3')
    print(json.dumps(assess_health(**vars(parser.parse_args())), indent=2))
