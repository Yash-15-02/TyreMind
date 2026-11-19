"""LangGraph routing, durable resume and ownership boundaries."""
import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import test_handoff as fixtures
from tyremind.agents.receive import receive_next
from tyremind.agents.supervisor import start_investigation, build_supervisor_graph


class SupervisorTests(unittest.TestCase):
    setUp = fixtures.HandoffTests.setUp
    add = fixtures.HandoffTests.add
    route = fixtures.HandoffTests.route
    query = fixtures.HandoffTests.query

    def run_supervisor(self, consumer='local-phase3'):
        return start_investigation(self.database, self.root/'supervisor', consumer, self.root/'phase3')

    def execute_sql(self, sql, parameters=()):
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute(sql, parameters)

    def test_graph_routes_and_keeps_job_open_without_fake_results(self):
        result = self.run_supervisor()
        self.assertEqual(result['action'], 'STARTED')
        self.assertEqual(result['stage'], 'AWAITING_HEALTH_ASSESSMENT')
        self.assertEqual(result['next_task'], 'HEALTH_ASSESSMENT')
        self.assertEqual(result['trace'], ['validate_event', 'supervisor', 'prepare_health_assessment'])
        self.assertEqual(self.query('SELECT status,result_json FROM investigations'), [('RUNNING', None)])
        state = json.loads(Path(result['state_path']).read_text())
        self.assertEqual(state['investigation']['health_assessment'], {})
        self.assertEqual(state['investigation']['maintenance_plan'], {})
        self.assertEqual(state['investigation']['telemetry']['pressure_kpa'], 620)
        self.assertNotIn('claim_token', json.dumps(state))
        self.assertEqual(result['llm_calls'], 0)
        self.assertFalse(result['specialist_agents_executed'])

    def test_rerun_resumes_without_invoking_graph(self):
        first = self.run_supervisor()
        with patch('tyremind.agents.supervisor.build_supervisor_graph', side_effect=AssertionError('must not rerun')):
            second = self.run_supervisor()
        self.assertEqual(second['action'], 'RESUMED')
        self.assertEqual(first['investigation_id'], second['investigation_id'])
        self.assertEqual(len(self.query('SELECT * FROM phase3_supervisor_states')), 1)

    def test_partial_evidence_routes_to_data_review_preserving_risk(self):
        self.execute_sql("UPDATE risk_assessments SET data_status='partial',data_issues_json=?",
                         (json.dumps(['Temperature reading unavailable.']),))
        result = self.run_supervisor()
        self.assertEqual(result['next_task'], 'DATA_REVIEW')
        self.assertEqual(result['stage'], 'AWAITING_DATA_REVIEW')
        self.assertEqual(result['risk_level'], 'High')
        self.assertIn('Temperature reading unavailable.', result['review_reasons'])
        self.assertEqual(result['task_plan'][0], 'DATA_REVIEW')
        self.assertEqual(result['trace'][-1], 'request_data_review')

    def test_recovery_snapshot_requires_current_status_review(self):
        self.add('normal', 'Normal', 2)
        self.route()
        result = self.run_supervisor()
        self.assertEqual(result['next_task'], 'DATA_REVIEW')
        self.assertIn('current level', result['review_reasons'][0])
        self.assertEqual(self.query('SELECT status FROM risk_events'), [('OPEN',)])

    def test_graph_error_leaves_receipt_resumable(self):
        with patch('tyremind.agents.supervisor.build_supervisor_graph', side_effect=RuntimeError('injected graph error')):
            with self.assertRaises(RuntimeError):
                self.run_supervisor()
        self.assertEqual(self.query('SELECT status FROM investigations'), [('RUNNING',)])
        self.assertEqual(self.run_supervisor()['action'], 'STARTED')

    def test_export_failure_resumes_committed_graph_state(self):
        from tyremind.agents.supervisor import export_json
        def fail(target, value):
            if Path(target).name != 'supervisor_report.json':
                raise OSError('injected export error')
            return export_json(target, value)
        with patch('tyremind.agents.supervisor.export_json', side_effect=fail):
            with self.assertRaises(OSError):
                self.run_supervisor()
        self.assertEqual(self.run_supervisor()['action'], 'RESUMED')
        self.assertEqual(len(self.query('SELECT * FROM phase3_supervisor_states')), 1)

    def test_corrupt_supervisor_checkpoint_rejected(self):
        self.run_supervisor()
        self.execute_sql("UPDATE phase3_supervisor_states SET state_json='{}'")
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_supervisor()

    def test_graph_rejects_mismatched_source_identity(self):
        receipt = receive_next(self.database, self.root/'received', inbox_output=self.root/'phase3')
        state = json.loads(Path(receipt['state_path']).read_text())
        state['tyre_id'] = 'OTHER'
        with self.assertRaisesRegex(ValueError, 'source handoff'):
            build_supervisor_graph().invoke({'investigation': state})

    def test_existing_receipt_snapshot_stays_stable_after_live_escalation(self):
        first = self.run_supervisor()
        self.add('critical', 'Critical', 2)
        self.route()
        second = self.run_supervisor()
        self.assertEqual(second['action'], 'RESUMED')
        self.assertEqual(second['risk_level'], 'High')
        self.assertEqual(second['investigation_id'], first['investigation_id'])
        self.assertEqual(self.query('SELECT revision FROM risk_events'), [(2,)])
        self.assertEqual(len(self.query('SELECT * FROM investigations')), 1)

    def test_other_consumer_cannot_take_owned_job(self):
        self.run_supervisor()
        self.assertEqual(self.run_supervisor('another-consumer')['action'], 'NO_WORK')


if __name__ == '__main__':
    unittest.main()
