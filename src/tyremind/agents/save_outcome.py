"""Commit the validated agent outcome to the investigation's SQLite result."""
import argparse
import json
from pathlib import Path

from tyremind.agents.validate_outcome import fingerprint, inspect
from tyremind.monitoring.events import finish_in_transaction, transaction


def save_outcome(identity, database=Path('data/tyremind.db'), consumer='local-phase3'):
    with transaction(database) as db:
        job = db.execute("""SELECT i.*, r.claim_token AS receipt_token
            FROM investigations i JOIN phase3_receipts r USING(investigation_id)
            WHERE i.investigation_id=? AND r.consumer_id=?""", (identity, consumer)).fetchone()
        if job is None:
            raise ValueError('No receipt for this investigation and consumer.')
        if job['status'] in ('FAILED', 'COMPLETED'):
            packet = json.loads(job['result_json'] or '{}')
            if packet.get('workflow') != 'agent_investigation':
                raise ValueError('This investigation was finalized by another workflow.')
            action = 'RESUMED'
        else:
            row = db.execute('SELECT state_json FROM phase3_agent_states WHERE investigation_id=?',
                             (identity,)).fetchone()
            state = json.loads(row['state_json']) if row else {}
            validation = state.pop('outcome_validation', None)
            if not validation or validation.get('snapshot') != fingerprint(state):
                raise ValueError('Run Step 8 on the current outcome before saving.')
            outcome = state['draft_outcome']
            issues = inspect(state, outcome)
            passed = validation['status'] == 'PASSED'
            if (validation['investigation_id'] != identity or state['investigation']['investigation_id'] != identity
                    or validation['status'] not in ('PASSED', 'FAILED')
                    or passed != (validation['issues'] == []) or (passed and issues)):
                raise ValueError('Validation does not match this outcome.')
            packet = dict(workflow='agent_investigation', investigation_id=identity,
                          event_id=job['event_id'], validation=validation)
            error = None
            if passed:
                packet.update(status='AWAITING_HUMAN_REVIEW', report=outcome,
                              alert=dict(event_id=job['event_id'], tyre_id=outcome['tyre_id'],
                                         priority=job['priority'], risk_level=outcome['risk_level']),
                              work_order=outcome['maintenance_draft'])
            else:
                error = 'Validation failed: ' + ' '.join(validation['issues'])
                packet.update(status='VALIDATION_FAILED', error=error)
            finished = finish_in_transaction(db, identity, job['receipt_token'], packet, error)
            db.execute('UPDATE phase3_receipts SET stage=? WHERE investigation_id=?',
                       (finished['status'], identity))
            action = 'SAVED'
    return dict(action=action, investigation_id=identity, status=packet['status'],
                kind=packet.get('report', {}).get('kind'), has_work_order=bool(packet.get('work_order')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('investigation_id', help='Investigation ID from Step 8')
    print(json.dumps(save_outcome(parser.parse_args().investigation_id), indent=2))
