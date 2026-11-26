import json
import unittest
from unittest.mock import patch

import test_handoff as fixtures
from tyremind.agents.investigation_loop import run_loop


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
