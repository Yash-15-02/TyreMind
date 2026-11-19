"""Route a saved Step 10 validation to delivery or recorded failure."""
import argparse
import json
from pathlib import Path

from tyremind.agents.supervisor import digest, export_json
from tyremind.monitoring.events import finish_in_transaction, transaction


def finalize(identity, database=Path('data/tyremind.db'), consumer_id='local-phase3', root=Path('data/phase3')):
    with transaction(database) as connection:
        job = connection.execute("""SELECT j.*,r.consumer_id,r.claim_token AS owner_token
            FROM investigations j JOIN phase3_receipts r USING(investigation_id)
            WHERE investigation_id=? AND r.consumer_id=?""", (identity, consumer_id)).fetchone()
        if job is None:
            raise ValueError('No receipt matches this investigation and consumer.')
        if job['status'] in ('FAILED', 'COMPLETED'):
            packet = json.loads(job['result_json'] or '{}')
            if packet.get('workflow') != 'phase3_validation_delivery':
                raise ValueError('Investigation already finished outside validation delivery.')
        else:
            row = connection.execute("""SELECT v.result_json,h.assessment,c.analysis,m.draft,e.payload_json,e.payload_sha256
                FROM phase3_validations v JOIN phase3_health h USING(investigation_id)
                JOIN phase3_root_causes c USING(investigation_id) JOIN phase3_maintenance_plans m USING(investigation_id)
                JOIN phase3_evidence e USING(investigation_id) WHERE investigation_id=?""", (identity,)).fetchone()
            if row is None:
                raise ValueError('Run Step 10 validation for this investigation first.')
            validation = json.loads(row['result_json'])
            sections = {k: row[k] for k in ('assessment','analysis','draft')}
            fingerprint = digest(json.dumps([sections,row['payload_sha256'],validation['references']],sort_keys=True))
            if (validation['investigation_id'] != identity or fingerprint != validation['source_sha256']
                    or digest(row['payload_json']) != row['payload_sha256']):
                raise ValueError('Validation inputs changed; rerun Step 10.')
            issues = validation['findings']
            expected = 'REVISION_REQUIRED' if issues else 'AWAITING_HUMAN_REVIEW'
            if validation['stage'] != expected or not isinstance(issues, list):
                raise ValueError('Inconsistent validation decision; rerun Step 10.')
            packet = dict(workflow='phase3_validation_delivery', investigation_id=identity,
                          event_id=job['event_id'], tyre_id=validation['tyre_id'], validation=validation)
            error = None
            if issues:
                error = f'Validation requires revision: {len(issues)} findings.'
                packet.update(step=12, status='VALIDATION_FAILED', error=error, automatic_retries=0)
            else:
                packet.update(step=11, status='AWAITING_HUMAN_REVIEW',
                              report=dict(assessment=row['assessment'], analysis=row['analysis'],
                                          references=validation['references'], evidence_notes=validation['evidence_notes']),
                              alert=dict(event_id=job['event_id'], priority=job['priority'], cutoff=validation['cutoff']),
                              work_order=dict(status='DRAFT', instructions=row['draft']))
            finished = finish_in_transaction(connection, identity, job['owner_token'], packet, error)
            connection.execute('UPDATE phase3_receipts SET stage=? WHERE investigation_id=?',
                               (finished['status'], identity))
    folder = root/'delivery'/identity
    export_json(folder/'outcome.json', packet)
    if packet['step'] == 11:
        for key in ('report','alert','work_order'):
            export_json(folder/f'{key}.json', packet[key])
    return dict(step=packet['step'], status=packet['status'], investigation_id=identity,
                findings=len(packet['validation']['findings']), output=str(folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('identity', help='investigation_id from Step 10')
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--consumer-id', default='local-phase3')
    print(json.dumps(finalize(**vars(parser.parse_args())), indent=2))
