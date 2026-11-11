"""Risk-case deduplication, queue ownership, escalation and lifecycle tests."""
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from tyremind.monitoring.events import (process_risk_events,claim_investigation,finish_investigation,
    retry_failed_investigation,resolve_event,transaction)


class EventTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.database=self.root/'fleet.db'
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute('CREATE TABLE trucks (truck_id TEXT PRIMARY KEY)')
                connection.execute('CREATE TABLE tyres (tyre_id TEXT PRIMARY KEY,truck_id TEXT REFERENCES trucks(truck_id))')
                connection.execute('INSERT INTO trucks VALUES (\'T1\')')
                connection.executemany('INSERT INTO tyres VALUES (?,\'T1\')',[('Y1',),('Y2',)])
                connection.execute('''CREATE TABLE risk_assessments (
                    assessment_id TEXT PRIMARY KEY,tyre_id TEXT,truck_id TEXT,risk_level TEXT,
                    data_status TEXT,rules_json TEXT,feature_as_of TEXT,assessed_at TEXT)''')

    def add(self,identity,level='High',day=1,tyre='Y1',rules=None,data_status='complete'):
        if rules is None:
            rules=['pressure_deviation'] if level in ('High','Critical') else []
        stamp=f'2026-01-{day:02d}T00:00:00+00:00'
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute('INSERT INTO risk_assessments VALUES (?,?,?,?,?,?,?,?)',
                    (identity,tyre,'T1',level,data_status,json.dumps([{'rule_id':rule} for rule in rules]),stamp,stamp))

    def process(self):
        return process_risk_events(self.database,self.root/'events')

    def query(self,sql):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.row_factory=sqlite3.Row
            return [dict(row) for row in connection.execute(sql)]

    def test_normal_medium_unknown_do_not_create_events(self):
        for index,level in enumerate(['Normal','Medium',None],1):
            self.add(str(index),level,index,data_status='insufficient_data' if level is None else 'complete')
        report=self.process()
        self.assertEqual(report['events_created'],0)
        self.assertEqual(report['investigations_queued'],0)
        self.assertFalse(report['agents_executed'])

    def test_replay_and_repeated_high_update_one_pending_case(self):
        self.add('first')
        self.assertEqual(self.process()['events_created'],1)
        self.assertEqual(self.process()['processed_assessments'],0)
        self.add('next',day=2)
        self.process()
        events=self.query('SELECT * FROM risk_events')
        jobs=self.query('SELECT * FROM investigations')
        self.assertEqual(len(events),1)
        self.assertEqual(len(jobs),1)
        self.assertEqual(events[0]['revision'],1)
        self.assertEqual(jobs[0]['requested_assessment_id'],'next')
        self.assertEqual(len(self.query('SELECT * FROM event_assessments')),2)

    def test_escalation_coalesces_pending_job(self):
        self.add('high')
        self.process()
        self.add('critical','Critical',2)
        result=self.process()
        self.assertEqual(result['investigations_queued'],0)
        job=self.query('SELECT * FROM investigations')[0]
        self.assertEqual(job['priority'],'Critical')
        self.assertEqual(job['target_revision'],2)
        self.assertEqual(self.query('SELECT highest_level FROM risk_events')[0]['highest_level'],'Critical')

    def test_new_concern_while_running_queues_one_followup(self):
        self.add('first')
        self.process()
        job=claim_investigation(self.database)
        self.add('new',day=2,rules=['pressure_deviation','temperature_elevated'])
        self.process()
        self.assertEqual(len(self.query('SELECT * FROM investigations')),1)
        result=finish_investigation(self.database,job['investigation_id'],job['claim_token'],result={'test':'done'})
        self.assertIsNotNone(result['followup_investigation_id'])
        followup=claim_investigation(self.database)
        self.assertEqual(followup['target_revision'],2)
        self.assertEqual(followup['requested_assessment_id'],'new')
        finish_investigation(self.database,followup['investigation_id'],followup['claim_token'],result={})
        self.assertIsNone(claim_investigation(self.database))

    def test_completed_report_is_not_requeued_for_unchanged_reading(self):
        self.add('first')
        self.process()
        job=claim_investigation(self.database)
        finish_investigation(self.database,job['investigation_id'],job['claim_token'],result={})
        self.add('same',day=2)
        self.assertEqual(self.process()['investigations_queued'],0)
        self.assertEqual(len(self.query('SELECT * FROM investigations')),1)

    def test_recovery_requires_review_and_unknown_does_not_close(self):
        self.add('critical','Critical')
        self.process()
        self.add('recovered','Normal',2)
        self.process()
        self.add('missing',None,3,data_status='stale')
        self.process()
        event=self.query('SELECT * FROM risk_events')[0]
        self.assertEqual(event['status'],'OPEN')
        self.assertEqual(event['current_level'],'Normal')
        self.assertEqual(event['highest_level'],'Critical')
        self.assertEqual(event['recovery_observed'],1)

    def test_human_resolution_and_new_episode(self):
        self.add('first')
        self.process()
        event=self.query('SELECT * FROM risk_events')[0]
        with self.assertRaisesRegex(ValueError,'Finish active'):
            resolve_event(self.database,event['event_id'],'reviewer','inspection done')
        job=claim_investigation(self.database)
        finish_investigation(self.database,job['investigation_id'],job['claim_token'],result={})
        resolve_event(self.database,event['event_id'],'reviewer','inspection completed and reviewed')
        self.add('new-episode',day=2)
        self.process()
        events=self.query('SELECT * FROM risk_events')
        self.assertEqual(len(events),2)
        self.assertEqual(sum(row['status']=='OPEN' for row in events),1)
        with closing(sqlite3.connect(self.database)) as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute("UPDATE risk_events SET status='OPEN' WHERE event_id=?",(event['event_id'],))
            connection.rollback()

    def test_failed_job_retry_is_explicit_and_idempotent(self):
        self.add('first')
        self.process()
        job=claim_investigation(self.database)
        finish_investigation(self.database,job['investigation_id'],job['claim_token'],error='test failure')
        self.add('same',day=2)
        self.assertEqual(self.process()['investigations_queued'],0)
        retry=retry_failed_investigation(self.database,job['investigation_id'])
        self.assertEqual(retry,retry_failed_investigation(self.database,job['investigation_id']))
        self.assertEqual(len(self.query("SELECT * FROM investigations WHERE status='PENDING'")),1)

    def test_wrong_claim_token_cannot_finish(self):
        self.add('first')
        self.process()
        job=claim_investigation(self.database)
        with self.assertRaisesRegex(ValueError,'claim token'):
            finish_investigation(self.database,job['investigation_id'],'wrong',result={})
        self.assertEqual(self.query('SELECT status FROM investigations')[0]['status'],'RUNNING')

    def test_concurrent_claims_do_not_claim_same_job(self):
        self.add('first')
        self.process()
        with ThreadPoolExecutor(max_workers=2) as executor:
            jobs=list(executor.map(lambda _:claim_investigation(self.database),range(2)))
        self.assertEqual(sum(job is not None for job in jobs),1)

    def test_late_evidence_does_not_rewind_current_state(self):
        self.add('current','Critical',10)
        self.process()
        self.add('late','High',1)
        self.process()
        event=self.query('SELECT * FROM risk_events')[0]
        self.assertEqual(event['current_level'],'Critical')
        self.assertEqual(event['last_seen'],'2026-01-10T00:00:00+00:00')
        self.assertEqual(event['first_seen'],'2026-01-01T00:00:00+00:00')
        self.assertEqual(self.query("SELECT action FROM event_assessments WHERE assessment_id='late'")[0]['action'],'late_evidence')

    def test_queue_failure_rolls_back_event_and_association(self):
        self.add('first')
        with transaction(self.database) as connection:
            connection.execute("CREATE TRIGGER fail_queue BEFORE INSERT ON investigations BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError,'injected failure'):
            self.process()
        self.assertEqual(self.query('SELECT * FROM risk_events'),[])
        self.assertEqual(self.query('SELECT * FROM event_routing'),[])

    def test_critical_job_is_claimed_first(self):
        self.add('high','High',1,'Y1')
        self.add('critical','Critical',2,'Y2')
        self.process()
        job=claim_investigation(self.database)
        self.assertEqual(job['priority'],'Critical')


if __name__=='__main__':
    unittest.main()
