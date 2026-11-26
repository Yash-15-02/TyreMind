import unittest
from unittest.mock import patch

import test_handoff as fixtures
from tyremind.agents.execute_tool import execute_tool
from tyremind.agents.investigation_state import open_state
from tyremind.agents.select_tool import select_tool


class ExecutionTests(unittest.TestCase):
    setUp = fixtures.HandoffTests.setUp
    add = fixtures.HandoffTests.add
    route = fixtures.HandoffTests.route

    @patch('tyremind.agents.select_tool.request_text', return_value='{"tool":"search_manuals","reason":"Find guidance","query":"truck tyre load"}')
    @patch('tyremind.agents.execute_tool.search_passages')
    def test_retry_after_failure_then_resume_saved_search(self, search, model):
        select_tool(self.database, self.root)
        search.side_effect = [FileNotFoundError(), {'passages': [{'source': {'pdf_page': 10}}]}]
        with self.assertRaises(FileNotFoundError):
            execute_tool(self.database, self.root)
        self.assertEqual(open_state(self.database, self.root)['state']['tool_call_count'], 0)
        first = execute_tool(self.database, self.root)
        self.assertEqual(first['action'], 'EXECUTED')
        self.assertEqual(execute_tool(self.database, self.root), {'action': 'RESUMED', 'result': first['result']})
        self.assertEqual(search.call_count, 2)
        self.assertEqual(open_state(self.database, self.root)['state']['tool_call_count'], 1)
