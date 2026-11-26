import unittest
from unittest.mock import patch

import test_handoff as fixtures
from tyremind.agents.select_tool import select_tool
from tyremind.agents.investigation_state import open_state


class SelectionTests(unittest.TestCase):
    setUp = fixtures.HandoffTests.setUp
    add = fixtures.HandoffTests.add
    route = fixtures.HandoffTests.route

    @patch('tyremind.agents.select_tool.request_text')
    def test_selection_resumes_without_second_model_call(self, model):
        model.return_value = '{"tool":"read_tyre_history","reason":"Check pressure trend"}'
        first = select_tool(self.database, self.root)
        state = open_state(self.database, self.root)['state']
        self.assertEqual(first['pending_tool']['arguments'], {'tyre_id': state['investigation']['tyre_id']})
        self.assertEqual(state['tool_call_count'], 0)
        self.assertEqual(select_tool(self.database, self.root)['action'], 'RESUMED')
        model.assert_called_once()

    @patch('tyremind.agents.select_tool.request_text', return_value='{"tool":"delete","reason":"test"}')
    def test_invalid_choice_preserves_state(self, model):
        with self.assertRaises(ValueError):
            select_tool(self.database, self.root)
        self.assertEqual(open_state(self.database, self.root)['state']['stage'], 'READY_FOR_TOOL_SELECTION')
