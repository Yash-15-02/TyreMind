import json
import unittest

import test_handoff as fixtures
from tyremind.agents.investigation_state import open_state
from tyremind.monitoring.events import transaction


class InvestigationStateTests(unittest.TestCase):
    setUp = fixtures.HandoffTests.setUp
    add = fixtures.HandoffTests.add
    route = fixtures.HandoffTests.route

    def test_resume_preserves_progress(self):
        first = open_state(self.database, self.root)
        self.assertEqual(first['action'], 'CREATED')
        state = first['state']
        self.assertEqual(state['tool_call_count'], 0)
        self.assertEqual(state['evidence'], [])
        state['tool_results'].append({'tool': 'pressure_history', 'readings': [620]})
        state['tool_call_count'] = 1
        with transaction(self.database) as db:
            db.execute('UPDATE phase3_agent_states SET state_json=?', (json.dumps(state),))
        self.assertEqual(open_state(self.database, self.root), {'action': 'RESUMED', 'state': state})
