"""Choose and checkpoint one pending action for the current investigation."""
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from tyremind.agents.investigation_state import open_state
from tyremind.agents.llm import request_text
from tyremind.agents.state import StrictModel
from tyremind.monitoring.events import transaction


class Choice(StrictModel):
    tool: Literal['read_tyre_history', 'read_ml_results', 'search_manuals']
    reason: str = Field(min_length=1, max_length=600)
    query: str = Field(default='', max_length=400)


def select_tool(database=Path('data/tyremind.db'), phase3=Path('data/phase3'),
                consumer='local-phase3'):
    opened = open_state(database, phase3, consumer)
    if opened['action'] == 'NO_WORK':
        return opened
    state = opened['state']
    case = state['investigation']
    if state['stage'] == 'TOOL_SELECTED':
        return {'action': 'RESUMED', 'pending_tool': state['pending_tool']}
    if state['stage'] != 'READY_FOR_TOOL_SELECTION':
        raise ValueError('This investigation is not ready for tool selection.')
    prompt = (
        'Choose the most useful next tool for this tyre. read_tyre_history gets past readings; '
        'read_ml_results gets model predictions; search_manuals retrieves technical guidance. '
        'Return only a JSON object with tool, reason and query. Use a nonempty search query '
        'only for search_manuals. The application supplies the tyre identity. Case data follows.\n'
        + json.dumps(state)
    )
    choice = Choice.model_validate_json(request_text('tool_selection', prompt))
    if not choice.reason.strip() or bool(choice.query.strip()) != (choice.tool == 'search_manuals'):
        raise ValueError('Invalid reason or search query for the selected tool.')
    arguments = ({'query': choice.query.strip()} if choice.tool == 'search_manuals'
                 else {'tyre_id': case['tyre_id']})
    pending = {'tool': choice.tool, 'arguments': arguments, 'reason': choice.reason}
    with transaction(database) as db:
        row = db.execute("""
            SELECT s.state_json FROM phase3_agent_states s
            JOIN investigations i USING(investigation_id)
            JOIN phase3_receipts r USING(investigation_id)
            WHERE s.investigation_id=? AND i.status='RUNNING'
              AND r.consumer_id=? AND i.claim_token=r.claim_token
        """, (case['investigation_id'], consumer)).fetchone()
        if row is None:
            raise ValueError('Investigation ownership changed during selection.')
        current = json.loads(row['state_json'])
        if current['stage'] == 'TOOL_SELECTED':
            return {'action': 'RESUMED', 'pending_tool': current['pending_tool']}
        if current != state:
            raise ValueError('Investigation changed during selection. Retry with current evidence.')
        current.update(stage='TOOL_SELECTED', pending_tool=pending)
        db.execute('UPDATE phase3_agent_states SET state_json=? WHERE investigation_id=?',
                   (json.dumps(current), case['investigation_id']))
    return {'action': 'SELECTED', 'pending_tool': pending}


if __name__ == '__main__':
    print(json.dumps(select_tool(), indent=2))
