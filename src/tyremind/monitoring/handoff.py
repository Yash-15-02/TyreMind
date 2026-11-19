"""Publish a durable local Phase 3 inbox without executing investigations."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path

from tyremind.monitoring.events import transaction,now

SCHEMA_VERSION='1.0'
HANDOFF_DDL='''CREATE TABLE IF NOT EXISTS phase3_handoffs (
 investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
 handoff_id TEXT UNIQUE NOT NULL,revision INTEGER NOT NULL,payload_json TEXT NOT NULL,
 payload_sha256 TEXT NOT NULL,status TEXT NOT NULL CHECK(status IN ('READY','CLAIMED','COMPLETED','FAILED','CANCELLED')),
 prepared_at TEXT NOT NULL,updated_at TEXT NOT NULL)'''


def payload_for(job,event,assessment):
    if assessment['risk_level'] not in ('High','Critical') or assessment['data_status']=='stale':
        raise ValueError('Investigation snapshot must reference actionable evidence.')
    if assessment['tyre_id']!=event['tyre_id'] or assessment['truck_id']!=event['truck_id']:
        raise ValueError('Investigation snapshot identities do not match its event.')
    payload={'schema_version':SCHEMA_VERSION,'message_type':'INVESTIGATE_TYRE',
             'investigation':{key:job[key] for key in ('investigation_id','event_id','target_revision','priority','requested_assessment_id')},
             'event':{key:event[key] for key in ('event_id','tyre_id','truck_id','status','current_level','highest_level','first_seen','last_seen','revision')},
             'assessment':{key:assessment[key] for key in ('assessment_id','timestamp','feature_as_of','risk_level','data_status','policy_version','policy_sha256','prediction_id')},
             'triggered_rules':json.loads(assessment['rules_json']),
             'data_issues':json.loads(assessment['data_issues_json']),
             'evidence':json.loads(assessment['evidence_json'])}
    identity=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return {'handoff_id':identity,**payload}


def prepare_phase3_handoffs(database=Path('data/tyremind.db'),output=Path('data/phase3')):
    created,updated,unchanged=0,0,0
    with transaction(database) as connection:
        connection.execute(HANDOFF_DDL)
        # Reconcile transport state with authoritative job/event state.
        connection.execute('''UPDATE phase3_handoffs SET status=(SELECT CASE
            WHEN j.status='RUNNING' THEN 'CLAIMED'
            WHEN j.status='COMPLETED' THEN 'COMPLETED'
            WHEN j.status='FAILED' THEN 'FAILED'
            WHEN e.status!='OPEN' THEN 'CANCELLED' ELSE 'READY' END
            FROM investigations j JOIN risk_events e ON e.event_id=j.event_id
            WHERE j.investigation_id=phase3_handoffs.investigation_id)''')
        jobs=connection.execute('''SELECT j.* FROM investigations j JOIN risk_events e ON e.event_id=j.event_id
            WHERE j.status='PENDING' AND e.status='OPEN'
            ORDER BY CASE j.priority WHEN 'Critical' THEN 0 ELSE 1 END,j.created_at,j.investigation_id''').fetchall()
        for job in jobs:
            event=connection.execute('SELECT * FROM risk_events WHERE event_id=?',(job['event_id'],)).fetchone()
            assessment=connection.execute('SELECT * FROM risk_assessments WHERE assessment_id=?',(job['requested_assessment_id'],)).fetchone()
            if job['target_revision']!=event['revision']:
                raise ValueError('Pending job is behind its event revision; run event routing first.')
            payload=payload_for(job,event,assessment)
            serialized=json.dumps(payload,sort_keys=True,allow_nan=False)
            digest=hashlib.sha256(serialized.encode()).hexdigest()
            existing=connection.execute('SELECT payload_sha256 FROM phase3_handoffs WHERE investigation_id=?',(job['investigation_id'],)).fetchone()
            if existing and existing['payload_sha256']==digest:
                unchanged+=1
                continue
            timestamp=now()
            connection.execute('''INSERT INTO phase3_handoffs VALUES (?,?,?,?,?,'READY',?,?)
                ON CONFLICT(investigation_id) DO UPDATE SET handoff_id=excluded.handoff_id,
                revision=excluded.revision,payload_json=excluded.payload_json,payload_sha256=excluded.payload_sha256,
                status='READY',updated_at=excluded.updated_at''',
                (job['investigation_id'],payload['handoff_id'],job['target_revision'],serialized,digest,timestamp,timestamp))
            created+=int(existing is None)
            updated+=int(existing is not None)
        envelopes=[json.loads(row['payload_json']) for row in connection.execute('''SELECT h.payload_json FROM phase3_handoffs h
            JOIN investigations j ON j.investigation_id=h.investigation_id JOIN risk_events e ON e.event_id=j.event_id
            WHERE h.status='READY' AND j.status='PENDING' AND e.status='OPEN'
            ORDER BY CASE j.priority WHEN 'Critical' THEN 0 ELSE 1 END,j.created_at,j.investigation_id''')]
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    inbox={'schema_version':SCHEMA_VERSION,'destination':'phase3_langgraph_consumer',
           'ready_count':len(envelopes),'messages':envelopes,'agents_executed':False}
    temporary=output/'inbox.json.tmp'
    temporary.write_text(json.dumps(inbox,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(output/'inbox.json')
    report={'ready_handoffs':len(envelopes),'created_handoffs':created,'updated_handoffs':updated,
            'unchanged_handoffs':unchanged,'agents_executed':False,'jobs_claimed':0,
            'inbox_path':str(output/'inbox.json'),
            'inbox_sha256':hashlib.sha256((output/'inbox.json').read_bytes()).hexdigest()}
    temporary=output/'handoff_report.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(output/'handoff_report.json')
    return report


def claim_handoff_transaction(connection,investigation_id,handoff_id):
    """Claim with an existing transaction, allowing atomic consumer checkpointing."""
    handoff=connection.execute('SELECT * FROM phase3_handoffs WHERE investigation_id=?',(investigation_id,)).fetchone()
    job=connection.execute('SELECT * FROM investigations WHERE investigation_id=?',(investigation_id,)).fetchone()
    if handoff is None or job is None:
        raise ValueError('Handoff does not exist; prepare the Phase 3 inbox first.')
    if handoff['handoff_id']!=handoff_id or handoff['status']!='READY' or job['status']!='PENDING':
        raise ValueError('Handoff is stale or already claimed.')
    if hashlib.sha256(handoff['payload_json'].encode()).hexdigest()!=handoff['payload_sha256']:
        raise ValueError('Handoff payload checksum mismatch.')
    payload=json.loads(handoff['payload_json'])
    event=connection.execute('SELECT * FROM risk_events WHERE event_id=?',(job['event_id'],)).fetchone()
    if event['status']!='OPEN' or job['target_revision']!=handoff['revision'] or job['requested_assessment_id']!=payload['investigation']['requested_assessment_id']:
        raise ValueError('Job snapshot changed; refresh the Phase 3 handoff before claiming.')
    expected_event={key:event[key] for key in payload['event']}
    if expected_event!=payload['event']:
        raise ValueError('Event snapshot changed; refresh the Phase 3 handoff before claiming.')
    token=uuid.uuid4().hex
    connection.execute("UPDATE investigations SET status='RUNNING',started_at=?,claim_token=? WHERE investigation_id=?",
                       (now(),token,investigation_id))
    connection.execute("UPDATE phase3_handoffs SET status='CLAIMED',updated_at=? WHERE investigation_id=?",(now(),investigation_id))
    return {**payload,'claim_token':token}


def claim_phase3_handoff(database,investigation_id,handoff_id):
    """Claim one validated snapshot; retained for existing callers."""
    with transaction(database) as connection:
        connection.execute(HANDOFF_DDL)
        return claim_handoff_transaction(connection,investigation_id,handoff_id)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('data/tyremind.db'))
    parser.add_argument('--output',type=Path,default=Path('data/phase3'))
    args=parser.parse_args()
    print(json.dumps(prepare_phase3_handoffs(args.database,args.output),indent=2))


if __name__=='__main__':
    main()
