"""Claim or resume one actionable event and persist Phase 3's initial state."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from tyremind.agents.state import InvestigationState,validate_handoff
from tyremind.monitoring.events import transaction,now
from tyremind.monitoring.handoff import (HANDOFF_DDL,claim_handoff_transaction,prepare_phase3_handoffs)

RECEIPT_DDL='''CREATE TABLE IF NOT EXISTS phase3_receipts (
 investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
 consumer_id TEXT NOT NULL,handoff_id TEXT NOT NULL,claim_token TEXT NOT NULL,
 stage TEXT NOT NULL CHECK(stage IN ('RECEIVED','COMPLETED','FAILED')),
 state_json TEXT NOT NULL,state_sha256 TEXT NOT NULL,received_at TEXT NOT NULL)'''


def receive_next(database=Path('data/tyremind.db'),output=Path('data/phase3/received'),
                 consumer_id='local-phase3',inbox_output=Path('data/phase3')):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',consumer_id):
        raise ValueError('Consumer ID must contain 1–64 letters, digits, underscores or hyphens.')
    # Refresh transport snapshots only; no new readings/events, model or LLM calls.
    prepare_phase3_handoffs(database,inbox_output)
    state=None
    action='NO_WORK'
    with transaction(database) as connection:
        connection.execute(HANDOFF_DDL)
        connection.execute(RECEIPT_DDL)
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS one_received_job_per_consumer ON phase3_receipts(consumer_id) WHERE stage='RECEIVED'")
        # Existing terminal jobs are authoritative; never fake completion of a received job.
        connection.execute('''UPDATE phase3_receipts SET stage=(SELECT status FROM investigations j
            WHERE j.investigation_id=phase3_receipts.investigation_id)
            WHERE stage='RECEIVED' AND investigation_id IN
            (SELECT investigation_id FROM investigations WHERE status IN ('COMPLETED','FAILED'))''')
        receipt=connection.execute("SELECT * FROM phase3_receipts WHERE consumer_id=? AND stage='RECEIVED'",(consumer_id,)).fetchone()
        if receipt:
            job=connection.execute('SELECT * FROM investigations WHERE investigation_id=?',(receipt['investigation_id'],)).fetchone()
            if job['status']!='RUNNING' or job['claim_token']!=receipt['claim_token']:
                raise ValueError('Received-state ownership mismatch; inspect the investigation before resuming.')
            if hashlib.sha256(receipt['state_json'].encode()).hexdigest()!=receipt['state_sha256']:
                raise ValueError('Stored investigation state checksum mismatch.')
            state=InvestigationState.model_validate_json(receipt['state_json'])
            action='RESUMED'
        else:
            candidate=connection.execute('''SELECT h.* FROM phase3_handoffs h
                JOIN investigations j ON j.investigation_id=h.investigation_id JOIN risk_events e ON e.event_id=j.event_id
                WHERE h.status='READY' AND j.status='PENDING' AND e.status='OPEN'
                ORDER BY CASE j.priority WHEN 'Critical' THEN 0 ELSE 1 END,j.created_at,j.investigation_id LIMIT 1''').fetchone()
            if candidate:
                raw=json.loads(candidate['payload_json'])
                message=validate_handoff(raw)
                claimed=claim_handoff_transaction(connection,candidate['investigation_id'],candidate['handoff_id'])
                received_at=now()
                evidence=raw['evidence']
                state=InvestigationState(investigation_id=message.investigation.investigation_id,
                    event_id=message.event.event_id,tyre_id=message.event.tyre_id,truck_id=message.event.truck_id,
                    handoff_id=message.handoff_id,target_revision=message.investigation.target_revision,
                    priority=message.investigation.priority,risk_level=message.assessment.risk_level,
                    received_at=received_at,source_handoff=raw,
                    telemetry={key:value for key,value in evidence.items() if key not in ('prediction','feature_values')},
                    ml_prediction=evidence.get('prediction'))
                serialized=json.dumps(state.model_dump(mode='json'),sort_keys=True,allow_nan=False)
                connection.execute('INSERT INTO phase3_receipts VALUES (?,?,?,?,\'RECEIVED\',?,?,?)',
                    (state.investigation_id,consumer_id,state.handoff_id,claimed['claim_token'],serialized,
                     hashlib.sha256(serialized.encode()).hexdigest(),received_at))
                action='RECEIVED'
        pending=connection.execute("SELECT COUNT(*) FROM investigations WHERE status='PENDING'").fetchone()[0]
    report={'action':action,'consumer_id':consumer_id,'stage':state.stage if state else None,
            'investigation_id':state.investigation_id if state else None,'event_id':state.event_id if state else None,
            'tyre_id':state.tyre_id if state else None,'priority':state.priority if state else None,
            'pending_investigations':pending,'agents_executed':False,'llm_calls':0,
            'note':'Job ownership is RUNNING; reception is checkpointed at RECEIVED, awaiting the next Phase 3 step.' if state else 'No pending actionable event is available.'}
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if state:
        target=output/f'{state.investigation_id}.json'
        temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state.model_dump(mode='json'),indent=2,allow_nan=False)+'\n',encoding='utf-8')
        temporary.replace(target)
        report['state_path']=str(target)
    temporary=output/'receive_report.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(output/'receive_report.json')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('data/tyremind.db'))
    parser.add_argument('--output',type=Path,default=Path('data/phase3/received'))
    parser.add_argument('--inbox-output',type=Path,default=Path('data/phase3'))
    parser.add_argument('--consumer-id',default='local-phase3')
    args=parser.parse_args()
    print(json.dumps(receive_next(args.database,args.output,args.consumer_id,args.inbox_output),indent=2))


if __name__=='__main__':
    main()
