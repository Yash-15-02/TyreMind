import json
import unittest
from unittest.mock import patch

import test_handoff as fixtures
from tyremind.agents.investigation_loop import run_loop
from tyremind.agents.prepare_outcome import prepare_outcome
from tyremind.agents.phase4_handoff import publish
from tyremind.agents.save_outcome import save_outcome
from tyremind.agents.validate_outcome import validate_outcome


class InvestigationLoopTests(unittest.TestCase):
    setUp = fixtures.HandoffTests.setUp
    add = fixtures.HandoffTests.add
    route = fixtures.HandoffTests.route

    def simulate(self, decisions, expected):
        choice = '{"tool":"search_manuals","reason":"Check load","query":"load"}'
        reviews = [json.dumps(dict(finding='Load guidance', reason='Check readings', decision=d))
                   for d in decisions]
        with patch('tyremind.agents.select_tool.request_text', return_value=choice), patch(
                'tyremind.agents.execute_tool.search_passages', return_value={'passages': []}), patch(
                'tyremind.agents.review_evidence.request_text', side_effect=reviews) as model:
            result = run_loop(self.database, self.root, max_calls=2)
            self.assertEqual(result['stage'], expected)
            self.assertEqual(result['tool_call_count'], len(decisions))
            self.assertEqual(run_loop(self.database, self.root, max_calls=2), result)
            self.assertEqual(model.call_count, len(decisions))

    def test_repeat_then_finish(self):
        self.simulate(['MORE_EVIDENCE', 'ENOUGH_EVIDENCE'], 'READY_FOR_FINDINGS')

    def test_budget_persists_across_restart(self):
        self.simulate(['MORE_EVIDENCE'] * 2, 'NEEDS_HUMAN_REVIEW')

    def test_human_review_packet(self):
        self.simulate(['MORE_EVIDENCE'] * 2, 'NEEDS_HUMAN_REVIEW')
        outcome = prepare_outcome(self.database, self.root)['outcome']
        self.assertEqual(outcome['kind'], 'HUMAN_REVIEW')
        self.assertIsNone(outcome['maintenance_draft'])
        self.assertEqual(len(outcome['reviews']), 2)
        self.assertEqual(prepare_outcome(self.database, self.root)['action'], 'RESUMED')

    def test_findings_draft(self):
        self.simulate(['ENOUGH_EVIDENCE'], 'READY_FOR_FINDINGS')
        reply = json.dumps(dict(findings='Alert observed', uncertainty='Cause unknown',
                                proposed_inspection='Inspect the tyre'))
        with patch('tyremind.agents.prepare_outcome.request_text', return_value=reply) as model:
            outcome = prepare_outcome(self.database, self.root)['outcome']
            self.assertEqual(outcome['maintenance_draft']['status'], 'DRAFT')
            self.assertEqual(prepare_outcome(self.database, self.root)['outcome'], outcome)
            model.assert_called_once()
        validation = validate_outcome(self.database, self.root)
        self.assertTrue(save_outcome(validation['investigation_id'], self.database)['has_work_order'])
        self.assertEqual(publish(validation['investigation_id'], self.database, self.root/'inbox')['message_type'],
                         'INVESTIGATION_READY_FOR_REVIEW')

    def test_validation_corrects_mismatched_sources_and_missing_checks(self):
        from tyremind.agents.validate_outcome import validate_outcome
        from tyremind.monitoring.events import transaction

        self.simulate(['MORE_EVIDENCE'] * 2, 'NEEDS_HUMAN_REVIEW')
        prepare_outcome(self.database, self.root)
        with transaction(self.database) as db:
            db.execute("""UPDATE phase3_agent_states SET state_json=json_set(
                json_remove(state_json,'$.draft_outcome.checks'),
                '$.draft_outcome.retrieved_sources',json('[{}]'))""")
        result = validate_outcome(self.database, self.root)
        self.assertEqual(result['status'], 'PASSED')
        self.assertEqual(len(result['initial_issues']), 2)
        self.assertEqual(validate_outcome(self.database, self.root), result)
        identity = result['investigation_id']
        saved = save_outcome(identity, self.database)
        self.assertEqual(saved['kind'], 'HUMAN_REVIEW')
        self.assertFalse(saved['has_work_order'])
        self.assertEqual(save_outcome(identity, self.database)['action'], 'RESUMED')
        handoff = publish(identity, self.database, self.root/'inbox')
        self.assertEqual(handoff['message_type'], 'INVESTIGATION_NEEDS_HUMAN_REVIEW')
        self.assertEqual(publish(identity, self.database, self.root/'inbox')['action'], 'ALREADY_PUBLISHED')

    def test_failed_correction_is_saved(self):
        from tyremind.agents.validate_outcome import validate_outcome

        self.simulate(['ENOUGH_EVIDENCE'], 'READY_FOR_FINDINGS')
        reply = json.dumps(dict(findings='safe to drive', uncertainty='Cause unknown',
                                proposed_inspection='Inspect tyre'))
        with patch('tyremind.agents.prepare_outcome.request_text', return_value=reply) as model:
            prepare_outcome(self.database, self.root)
            result = validate_outcome(self.database, self.root)
            self.assertEqual(result['status'], 'FAILED')
            self.assertEqual(validate_outcome(self.database, self.root), result)
            self.assertEqual(model.call_count, 2)
        saved = save_outcome(result['investigation_id'], self.database)
        self.assertEqual(saved['status'], 'VALIDATION_FAILED')
        self.assertFalse(saved['has_work_order'])
        self.assertEqual(publish(result['investigation_id'], self.database, self.root/'inbox')['message_type'],
                         'INVESTIGATION_VALIDATION_FAILED')
