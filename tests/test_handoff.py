"""Durable Phase 3 delivery, snapshot freshness and claim isolation."""
import json
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

from tyremind.monitoring.events import process_risk_events,finish_investigation
from tyremind.monitoring.handoff import prepare_phase3_handoffs,claim_phase3_handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.database=self.root/'fleet.db'
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute('CREATE TABLE trucks (truck_id TEXT PRIMARY KEY)')
                connection.execute('CREATE TABLE tyres (tyre_id TEXT PRIMARY KEY,truck_id TEXT)')
                connection.execute("INSERT INTO trucks VALUES ('T1')")
                connection.execute("INSERT INTO tyres VALUES ('Y1','T1')")
                connection.execute('''CREATE TABLE risk_assessments (
                    assessment_id TEXT PRIMARY KEY,tyre_id TEXT,truck_id TEXT,risk_level TEXT,data_status TEXT,
                    rules_json TEXT,feature_as_of TEXT,assessed_at TEXT,timestamp TEXT,policy_version TEXT,
                    policy_sha256 TEXT,prediction_id TEXT,data_issues_json TEXT,evidence_json TEXT)''')
        self.add('first')
        self.route()

    def add(self,identity,level='High',day=1):
        stamp=f'2026-01-{day:02d}T00:00:00+00:00'
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute('INSERT INTO risk_assessments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (identity,'Y1','T1',level,'complete',json.dumps([dict(rule_id='pressure_deviation',level=level)]),
                     stamp,stamp,stamp,'demo-1.0','policy-test','prediction-test','[]',json.dumps(dict(pressure_kpa=620,reference_pressure_kpa=800))))

    def route(self):
        return process_risk_events(self.database,self.root/'events')

    def prepare(self):
        return prepare_phase3_handoffs(self.database,self.root/'phase3')

    def message(self):
        return json.loads((self.root/'phase3/inbox.json').read_text())['messages'][0]

    def query(self,sql):
        with closing(sqlite3.connect(self.database)) as connection:
            return connection.execute(sql).fetchall()

    def test_prepare_preserves_payload_and_does_not_start_agents(self):
        result=self.prepare()
        payload=self.message()
        self.assertEqual(result['created_handoffs'],1)
        self.assertEqual(result['jobs_claimed'],0)
        self.assertFalse(result['agents_executed'])
        self.assertEqual(payload['event']['tyre_id'],'Y1')
        self.assertEqual(payload['assessment']['assessment_id'],'first')
        self.assertEqual(payload['evidence']['pressure_kpa'],620)
        self.assertEqual(self.query('SELECT status FROM investigations'),[('PENDING',)])
        self.assertNotIn('claim_token',json.dumps(payload))

    def test_repeat_is_identical_and_coalesced_revision_updates_one_handoff(self):
        self.prepare()
        original=self.message()['handoff_id']
        repeat=self.prepare()
        self.assertEqual(repeat['created_handoffs'],0)
        self.assertEqual(repeat['unchanged_handoffs'],1)
        self.assertEqual(self.message()['handoff_id'],original)
        self.add('escalated','Critical',2)
        self.route()
        changed=self.prepare()
        self.assertEqual(changed['updated_handoffs'],1)
        self.assertEqual(len(self.query('SELECT * FROM phase3_handoffs')),1)
        self.assertEqual(self.message()['investigation']['target_revision'],2)
        self.assertNotEqual(self.message()['handoff_id'],original)

    def test_claim_is_atomic_and_completed_handoff_not_redelivered(self):
        self.prepare()
        message=self.message()
        def claim(_):
            try:
                return claim_phase3_handoff(self.database,message['investigation']['investigation_id'],message['handoff_id'])
            except ValueError:
                return None
        with ThreadPoolExecutor(max_workers=2) as executor:
            results=list(executor.map(claim,range(2)))
        self.assertEqual(sum(result is not None for result in results),1)
        claimed=next(result for result in results if result is not None)
        job_id=message['investigation']['investigation_id']
        self.assertEqual(self.prepare()['ready_handoffs'],0)
        finish_investigation(self.database,job_id,claimed['claim_token'],result={'test':'completed'})
        self.assertEqual(self.query('SELECT status FROM phase3_handoffs'),[('COMPLETED',)])
        self.assertEqual(self.prepare()['ready_handoffs'],0)

    def test_stale_envelope_rejected_after_escalation(self):
        self.prepare()
        old=self.message()
        self.add('escalated','Critical',2)
        self.route()
        with self.assertRaisesRegex(ValueError,'snapshot changed'):
            claim_phase3_handoff(self.database,old['investigation']['investigation_id'],old['handoff_id'])
        self.assertEqual(self.query('SELECT status FROM investigations'),[('PENDING',)])

    def test_failed_export_can_be_retried_without_duplicate_handoffs(self):
        from unittest.mock import patch
        original=Path.replace
        def fail(self,target):
            if self.name=='inbox.json.tmp':
                raise OSError('injected export failure')
            return original(self,target)
        with patch.object(Path,'replace',fail):
            with self.assertRaises(OSError):
                self.prepare()
        self.assertEqual(self.prepare()['unchanged_handoffs'],1)
        self.assertEqual(len(self.query('SELECT * FROM phase3_handoffs')),1)

    def test_normal_without_open_case_produces_empty_inbox(self):
        # Remove only fixture-created cases/jobs; use a separate normal assessment.
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute('DELETE FROM investigations')
                connection.execute('DELETE FROM event_assessments')
                connection.execute('DELETE FROM event_routing')
                connection.execute('DELETE FROM risk_events')
                connection.execute('DELETE FROM risk_assessments')
        self.add('normal','Normal',2)
        self.route()
        self.assertEqual(self.prepare()['ready_handoffs'],0)


if __name__=='__main__':
    unittest.main()
