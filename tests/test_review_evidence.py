import json
import unittest
from unittest.mock import patch

import test_handoff as fixtures
from tyremind.agents.execute_tool import execute_tool
from tyremind.agents.select_tool import select_tool
from tyremind.agents.review_evidence import review_evidence


class ReviewTests(unittest.TestCase):
    setUp = fixtures.HandoffTests.setUp
    add = fixtures.HandoffTests.add
    route = fixtures.HandoffTests.route

    def check_decision(self, decision, stage):
        with patch('tyremind.agents.select_tool.request_text', return_value='{"tool":"search_manuals","reason":"check guidance","query":"pressure"}'):
            select_tool(self.database, self.root)
        with patch('tyremind.agents.execute_tool.search_passages', return_value={'passages': []}):
            execute_tool(self.database, self.root)
        reply = json.dumps(dict(decision=decision, finding='No manual match', reason='Review current measurements'))
        with patch('tyremind.agents.review_evidence.request_text', side_effect=['invalid', reply]) as model:
            with self.assertRaises(ValueError):
                review_evidence(self.database, self.root)
            result = review_evidence(self.database, self.root)
            self.assertEqual(result['stage'], stage)
            self.assertEqual(result['review']['call_number'], 1)
            self.assertEqual(review_evidence(self.database, self.root)['action'], 'RESUMED')
            self.assertEqual(model.call_count, 2)

    def test_more_needed(self):
        self.check_decision('MORE_EVIDENCE', 'READY_FOR_TOOL_SELECTION')

    def test_ready_to_draft(self):
        self.check_decision('ENOUGH_EVIDENCE', 'READY_FOR_FINDINGS')
