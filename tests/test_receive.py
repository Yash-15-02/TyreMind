"""Phase 3 receipt ownership, validation and restart behavior."""
import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import test_handoff as fixtures
from tyremind.agents.receive import receive_next


class ReceiveTests(unittest.TestCase):
    setUp=fixtures.HandoffTests.setUp
    add=fixtures.HandoffTests.add
    route=fixtures.HandoffTests.route
    query=fixtures.HandoffTests.query
    prepare=fixtures.HandoffTests.prepare

    def receive(self):
        return receive_next(self.database,self.root/'received',inbox_output=self.root/'phase3')

    def test_receive_then_resume_same_job(self):
        first=self.receive()
        second=self.receive()
        self.assertEqual(first['action'],'RECEIVED')
        self.assertEqual(second['action'],'RESUMED')
        self.assertEqual(first['investigation_id'],second['investigation_id'])
        self.assertEqual(self.query('SELECT status FROM investigations'),[('RUNNING',)])
        state=json.loads(Path(first['state_path']).read_text())
        self.assertEqual(state['stage'],'RECEIVED')
        self.assertEqual(state['telemetry']['pressure_kpa'],620)
        self.assertEqual(state['maintenance_plan'],{})
        self.assertFalse(first['agents_executed'])
        self.assertNotIn('claim_token',json.dumps(state))

    def test_invalid_payload_rolls_back_claim(self):
        self.prepare()
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("UPDATE phase3_handoffs SET payload_json='{}'")
        with patch('tyremind.agents.receive.prepare_phase3_handoffs'):
            with self.assertRaises(ValueError):
                self.receive()
        self.assertEqual(self.query('SELECT status FROM investigations'),[('PENDING',)])

    def test_export_failure_resumes_committed_receipt(self):
        original=Path.replace
        def fail(path,target):
            if path.parent==self.root/'received':
                raise OSError('injected output failure')
            return original(path,target)
        with patch.object(Path,'replace',fail):
            with self.assertRaises(OSError):
                self.receive()
        self.assertEqual(self.query('SELECT status FROM investigations'),[('RUNNING',)])
        self.assertEqual(self.receive()['action'],'RESUMED')

    def test_corrupt_checkpoint_is_rejected(self):
        self.receive()
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("UPDATE phase3_receipts SET state_json='{}'")
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.receive()

    def test_no_work_when_other_consumer_owns_job(self):
        self.receive()
        result=receive_next(self.database,self.root/'other',consumer_id='other',inbox_output=self.root/'phase3')
        self.assertEqual(result['action'],'NO_WORK')

    def test_completion_frees_consumer(self):
        from tyremind.monitoring.events import finish_investigation
        self.receive()
        job,token=self.query('SELECT investigation_id,claim_token FROM phase3_receipts')[0]
        finish_investigation(self.database,job,token,result={'fixture':True})
        self.assertEqual(self.receive()['action'],'NO_WORK')
        self.assertEqual(self.query('SELECT stage FROM phase3_receipts'),[('COMPLETED',)])

    def test_invalid_consumer_is_rejected(self):
        with self.assertRaises(ValueError):
            receive_next(self.database,self.root/'received',consumer_id='../bad')
