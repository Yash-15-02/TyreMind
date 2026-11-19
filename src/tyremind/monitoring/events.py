"""Deduplicate tyre risk cases and queue investigations without executing agents."""
import argparse
import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

SEVERITY={'Normal':0,'Medium':1,'High':2,'Critical':3}
DDL=[
'''CREATE TABLE IF NOT EXISTS risk_events (
 event_id TEXT PRIMARY KEY,tyre_id TEXT NOT NULL REFERENCES tyres(tyre_id),
 truck_id TEXT NOT NULL REFERENCES trucks(truck_id),status TEXT NOT NULL CHECK(status IN ('OPEN','RESOLVED')),
 current_level TEXT,highest_level TEXT NOT NULL CHECK(highest_level IN ('High','Critical')),
 first_seen TEXT NOT NULL,last_seen TEXT NOT NULL,revision INTEGER NOT NULL,
 known_rules_json TEXT NOT NULL,latest_assessment_id TEXT NOT NULL REFERENCES risk_assessments(assessment_id),
 latest_actionable_assessment_id TEXT NOT NULL REFERENCES risk_assessments(assessment_id),
 recovery_observed INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,
 resolved_by TEXT,resolution_reason TEXT,resolved_at TEXT)''',
"CREATE UNIQUE INDEX IF NOT EXISTS one_open_event_per_tyre ON risk_events(tyre_id) WHERE status='OPEN'",
'''CREATE TABLE IF NOT EXISTS event_routing (
 assessment_id TEXT PRIMARY KEY REFERENCES risk_assessments(assessment_id),event_id TEXT REFERENCES risk_events(event_id),
 action TEXT NOT NULL,routed_at TEXT NOT NULL)''',
'''CREATE TABLE IF NOT EXISTS event_assessments (
 assessment_id TEXT PRIMARY KEY REFERENCES risk_assessments(assessment_id),event_id TEXT NOT NULL REFERENCES risk_events(event_id),
 action TEXT NOT NULL,new_rules_json TEXT NOT NULL,escalated INTEGER NOT NULL,recovery_observed INTEGER NOT NULL,
 recorded_at TEXT NOT NULL)''',
'''CREATE TABLE IF NOT EXISTS investigations (
 investigation_id TEXT PRIMARY KEY,event_id TEXT NOT NULL REFERENCES risk_events(event_id),
 target_revision INTEGER NOT NULL,requested_assessment_id TEXT NOT NULL REFERENCES risk_assessments(assessment_id),
 priority TEXT NOT NULL CHECK(priority IN ('High','Critical')),
 status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETED','FAILED')),
 created_at TEXT NOT NULL,started_at TEXT,finished_at TEXT,claim_token TEXT,error TEXT,result_json TEXT,
 retry_of TEXT UNIQUE REFERENCES investigations(investigation_id))''',
"CREATE UNIQUE INDEX IF NOT EXISTS one_active_investigation_per_event ON investigations(event_id) WHERE status IN ('PENDING','RUNNING')",
]


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def transaction(database):
    database=Path(database)
    if not database.is_file():
        raise FileNotFoundError('Operational database missing.')
    with closing(sqlite3.connect(database,timeout=30)) as connection:
        connection.row_factory=sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('BEGIN IMMEDIATE')
        try:
            for statement in DDL:
                connection.execute(statement)
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def queue_job(connection,event_id,retry_of=None):
    event=connection.execute('SELECT * FROM risk_events WHERE event_id=?',(event_id,)).fetchone()
    if event['status']!='OPEN':
        return None,'resolved'
    active=connection.execute("SELECT * FROM investigations WHERE event_id=? AND status IN ('PENDING','RUNNING')",(event_id,)).fetchone()
    if active:
        if active['status']=='PENDING':
            connection.execute('UPDATE investigations SET target_revision=?,requested_assessment_id=?,priority=? WHERE investigation_id=?',
                (event['revision'],event['latest_actionable_assessment_id'],event['highest_level'],active['investigation_id']))
        return active['investigation_id'],'coalesced'
    identity=uuid.uuid4().hex
    connection.execute('''INSERT INTO investigations
        (investigation_id,event_id,target_revision,requested_assessment_id,priority,status,created_at,retry_of)
        VALUES (?,?,?,?,?,'PENDING',?,?)''',
        (identity,event_id,event['revision'],event['latest_actionable_assessment_id'],event['highest_level'],now(),retry_of))
    return identity,'created'


def process_risk_events(database=Path('data/tyremind.db'),output=Path('data/monitoring/events')):
    counts=dict(processed_assessments=0,events_created=0,events_updated=0,
                investigations_queued=0,investigations_coalesced=0,recovery_observations=0)
    with transaction(database) as connection:
        assessments=connection.execute('''SELECT r.* FROM risk_assessments r
            LEFT JOIN event_routing h ON h.assessment_id=r.assessment_id WHERE h.assessment_id IS NULL
            ORDER BY r.feature_as_of,r.assessed_at,r.assessment_id''').fetchall()
        for assessment in assessments:
            counts['processed_assessments']+=1
            identity=assessment['assessment_id']
            level=assessment['risk_level']
            actionable=level in ('High','Critical') and assessment['data_status']!='stale'
            rules={rule['rule_id'] for rule in json.loads(assessment['rules_json'])}
            event=connection.execute("SELECT * FROM risk_events WHERE tyre_id=? AND status='OPEN'",(assessment['tyre_id'],)).fetchone()
            timestamp=assessment['feature_as_of']
            new_rules,escalated,recovery=[],False,False
            material=False
            late=False
            if event is None and actionable:
                event_id=uuid.uuid4().hex
                connection.execute('''INSERT INTO risk_events
                    (event_id,tyre_id,truck_id,status,current_level,highest_level,first_seen,last_seen,revision,
                     known_rules_json,latest_assessment_id,latest_actionable_assessment_id,created_at,updated_at)
                    VALUES (?,?,?,'OPEN',?,?,?,?,1,?,?,?,?,?)''',
                    (event_id,assessment['tyre_id'],assessment['truck_id'],level,level,timestamp,timestamp,
                     json.dumps(sorted(rules)),identity,identity,now(),now()))
                action='created'
                material=True
                new_rules=sorted(rules)
                counts['events_created']+=1
            elif event is None:
                event_id=None
                action='continue_monitoring' if level in ('Normal','Medium') else 'data_review'
            else:
                event_id=event['event_id']
                late=pd.Timestamp(timestamp)<pd.Timestamp(event['last_seen'])
                if late:
                    action='late_evidence'
                    known=set(json.loads(event['known_rules_json']))
                    new_rules=sorted(rules-known) if actionable else []
                    escalated=actionable and SEVERITY[level]>SEVERITY[event['highest_level']]
                    material=actionable and (bool(new_rules) or escalated)
                    highest=level if escalated else event['highest_level']
                    first_seen=min(event['first_seen'],timestamp) if actionable else event['first_seen']
                    connection.execute('UPDATE risk_events SET first_seen=?,highest_level=?,revision=?,known_rules_json=?,updated_at=? WHERE event_id=?',
                        (first_seen,highest,event['revision']+int(material),json.dumps(sorted(known|set(new_rules))),now(),event_id))
                    counts['events_updated']+=1
                else:
                    known=set(json.loads(event['known_rules_json']))
                    new_rules=sorted(rules-known) if actionable else []
                    escalated=actionable and SEVERITY[level]>SEVERITY.get(event['current_level'],0)
                    material=actionable and (bool(new_rules) or escalated)
                    recovery=level in ('Normal','Medium') and assessment['data_status']=='complete'
                    action=('escalated' if escalated else 'new_concern' if new_rules else
                            'recovery_observed' if recovery else 'data_review' if level is None else 'updated')
                    current=level if level is not None else event['current_level']
                    highest=max((event['highest_level'],level),key=lambda value:SEVERITY.get(value,-1))
                    connection.execute('''UPDATE risk_events SET current_level=?,highest_level=?,last_seen=?,revision=?,
                        known_rules_json=?,latest_assessment_id=?,latest_actionable_assessment_id=?,
                        recovery_observed=?,updated_at=? WHERE event_id=?''',
                        (current,highest,timestamp,event['revision']+int(material),json.dumps(sorted(known|set(new_rules))),
                         identity,identity if actionable else event['latest_actionable_assessment_id'],
                         1 if recovery else 0 if actionable else event['recovery_observed'],now(),event_id))
                    counts['events_updated']+=1
                    counts['recovery_observations']+=int(recovery)
            connection.execute('INSERT INTO event_routing VALUES (?,?,?,?)',(identity,event_id,action,now()))
            if event_id:
                connection.execute('INSERT INTO event_assessments VALUES (?,?,?,?,?,?,?)',
                    (identity,event_id,action,json.dumps(new_rules),int(escalated),int(recovery),now()))
            active=connection.execute("SELECT investigation_id FROM investigations WHERE event_id=? AND status IN ('PENDING','RUNNING')",(event_id,)).fetchone() if event_id else None
            if material or (actionable and not late and active is not None):
                _,queue_action=queue_job(connection,event_id)
                counts['investigations_queued']+=int(queue_action=='created')
                counts['investigations_coalesced']+=int(queue_action=='coalesced')
        open_events=pd.read_sql_query("SELECT * FROM risk_events WHERE status='OPEN' ORDER BY last_seen DESC,event_id",connection)
        jobs=pd.read_sql_query('SELECT * FROM investigations ORDER BY created_at,investigation_id',connection)
        counts['open_events']=len(open_events)
        counts['pending_investigations']=int(jobs.status.eq('PENDING').sum())
        counts['running_investigations']=int(jobs.status.eq('RUNNING').sum())
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for filename,frame in [('open_events.csv',open_events),('investigation_queue.csv',jobs.drop(columns=['claim_token']))]:
        temporary=output/(filename+'.tmp')
        frame.to_csv(temporary,index=False)
        temporary.replace(output/filename)
    report={**counts,'agents_executed':False,'output':str(output)}
    temporary=output/'event_report.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(output/'event_report.json')
    return report


def claim_investigation(database=Path('data/tyremind.db')):
    """Phase 3 consumer hook: atomically claim one job, highest priority first."""
    with transaction(database) as connection:
        job=connection.execute('''SELECT j.* FROM investigations j JOIN risk_events e ON j.event_id=e.event_id
            WHERE j.status='PENDING' AND e.status='OPEN'
            ORDER BY CASE j.priority WHEN 'Critical' THEN 0 ELSE 1 END,j.created_at,j.investigation_id LIMIT 1''').fetchone()
        if job is None:
            return None
        token=uuid.uuid4().hex
        connection.execute("UPDATE investigations SET status='RUNNING',started_at=?,claim_token=? WHERE investigation_id=?",
                           (now(),token,job['investigation_id']))
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='phase3_handoffs'").fetchone():
            connection.execute("UPDATE phase3_handoffs SET status='CLAIMED',updated_at=? WHERE investigation_id=?",(now(),job['investigation_id']))
        return dict(connection.execute('SELECT * FROM investigations WHERE investigation_id=?',(job['investigation_id'],)).fetchone())


def finish_investigation(database,investigation_id,claim_token,result=None,error=None):
    """Persist a claimed result/error and queue one follow-up for newer concerns."""
    with transaction(database) as connection:
        return finish_in_transaction(connection,investigation_id,claim_token,result,error)


def finish_in_transaction(connection,investigation_id,claim_token,result=None,error=None):
    job=connection.execute('SELECT * FROM investigations WHERE investigation_id=?',(investigation_id,)).fetchone()
    if job is None or job['status']!='RUNNING' or job['claim_token']!=claim_token:
        raise ValueError('Investigation is not running under this claim token.')
    status='FAILED' if error is not None else 'COMPLETED'
    payload=json.dumps(result,allow_nan=False) if result is not None else None
    connection.execute('UPDATE investigations SET status=?,finished_at=?,error=?,result_json=? WHERE investigation_id=?',
                       (status,now(),str(error) if error is not None else None,payload,investigation_id))
    if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='phase3_handoffs'").fetchone():
        connection.execute('UPDATE phase3_handoffs SET status=?,updated_at=? WHERE investigation_id=?',(status,now(),investigation_id))
    event=connection.execute('SELECT * FROM risk_events WHERE event_id=?',(job['event_id'],)).fetchone()
    followup=None
    if event['status']=='OPEN' and event['revision']>job['target_revision']:
        followup,_=queue_job(connection,event['event_id'])
    return {'status':status,'followup_investigation_id':followup}


def retry_failed_investigation(database,investigation_id):
    """Explicit retry hook; repeated calls do not create duplicate retry jobs."""
    with transaction(database) as connection:
        job=connection.execute('SELECT * FROM investigations WHERE investigation_id=?',(investigation_id,)).fetchone()
        if job is None or job['status']!='FAILED':
            raise ValueError('Only a failed investigation can be explicitly retried.')
        previous=connection.execute('SELECT investigation_id FROM investigations WHERE retry_of=?',(investigation_id,)).fetchone()
        if previous:
            return previous['investigation_id']
        completed=connection.execute("SELECT investigation_id FROM investigations WHERE event_id=? AND target_revision>=? AND status='COMPLETED' ORDER BY finished_at DESC LIMIT 1",
                                     (job['event_id'],job['target_revision'])).fetchone()
        if completed:
            return completed['investigation_id']
        return queue_job(connection,job['event_id'],retry_of=investigation_id)[0]


def resolve_event(database,event_id,reviewed_by,reason):
    """Explicit human-review hook; no automatic resolution on recovery."""
    if not reviewed_by.strip() or not reason.strip():
        raise ValueError('Reviewer and resolution reason are required.')
    with transaction(database) as connection:
        event=connection.execute('SELECT * FROM risk_events WHERE event_id=?',(event_id,)).fetchone()
        if event is None or event['status']!='OPEN':
            raise ValueError('Event is not open.')
        if connection.execute("SELECT 1 FROM investigations WHERE event_id=? AND status IN ('PENDING','RUNNING')",(event_id,)).fetchone():
            raise ValueError('Finish active investigations before resolving the event.')
        connection.execute("UPDATE risk_events SET status='RESOLVED',resolved_by=?,resolution_reason=?,resolved_at=?,updated_at=? WHERE event_id=?",
                           (reviewed_by,reason,now(),now(),event_id))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('data/tyremind.db'))
    parser.add_argument('--output',type=Path,default=Path('data/monitoring/events'))
    args=parser.parse_args()
    print(json.dumps(process_risk_events(args.database,args.output),indent=2))


if __name__=='__main__':
    main()
