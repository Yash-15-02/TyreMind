"""Publish immutable finalized Phase 3 outcomes to the local Phase 4 inbox."""
import argparse
import json
from pathlib import Path

from tyremind.agents.supervisor import digest, export_json
from tyremind.monitoring.events import now, transaction


def enqueue(connection, packet):
    validation = packet['validation']
    identity = packet['investigation_id']
    if validation['investigation_id'] != identity or validation['tyre_id'] != packet['tyre_id']:
        raise ValueError('Validation and outcome identities differ.')
    if packet['step'] == 11 and packet['status'] == 'AWAITING_HUMAN_REVIEW' and not validation['findings']:
        if packet['work_order']['status'] != 'DRAFT' or not packet['report'] or not packet['alert']:
            raise ValueError('Incomplete review packet.')
        kind = 'INVESTIGATION_READY_FOR_REVIEW'
    elif packet['step'] == 12 and packet['status'] == 'VALIDATION_FAILED' and validation['findings']:
        if 'work_order' in packet:
            raise ValueError('Failed validation cannot deliver a work order.')
        kind = 'INVESTIGATION_VALIDATION_FAILED'
    else:
        raise ValueError('Unexpected Phase 3 outcome.')
    payload = json.dumps(packet, sort_keys=True, allow_nan=False)
    connection.execute("""CREATE TABLE IF NOT EXISTS phase4_inbox (
        investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
        message_type TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'READY',
        payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL, published_at TEXT NOT NULL)""")
    added = connection.execute("""INSERT INTO phase4_inbox
        (investigation_id,message_type,payload_json,payload_sha256,published_at) VALUES (?,?,?,?,?)
        ON CONFLICT(investigation_id) DO NOTHING""", (identity, kind, payload, digest(payload), now())).rowcount
    saved = connection.execute('SELECT * FROM phase4_inbox WHERE investigation_id=?', (identity,)).fetchone()
    if saved['payload_json'] != payload or saved['payload_sha256'] != digest(payload):
        raise ValueError('Existing inbox message differs from saved outcome.')
    return dict(action='PUBLISHED' if added else 'ALREADY_PUBLISHED', message_type=kind,
                status=saved['status'], investigation_id=identity)


def publish(identity, database=Path('data/tyremind.db'), output=Path('data/phase4/inbox')):
    with transaction(database) as connection:
        job = connection.execute('SELECT status,result_json FROM investigations WHERE investigation_id=?',
                                 (identity,)).fetchone()
        if job is None or job['status'] not in ('FAILED','COMPLETED'):
            raise ValueError('Finalize this investigation before publishing.')
        packet = json.loads(job['result_json'] or '{}')
        if (packet.get('workflow') != 'phase3_validation_delivery' or packet.get('investigation_id') != identity
                or (job['status'] == 'COMPLETED') != (packet.get('step') == 11)):
            raise ValueError('Investigation does not have a matching finalized Phase 3 outcome.')
        report = enqueue(connection, packet)
    path = output/f'{identity}.json'
    export_json(path, dict(schema_version=1, message_type=report['message_type'], outcome=packet))
    return dict(**report, output=str(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('identity', help='Finalized Phase 3 investigation ID')
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--output', type=Path, default=Path('data/phase4/inbox'))
    print(json.dumps(publish(**vars(parser.parse_args())), indent=2))
