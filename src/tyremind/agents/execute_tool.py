"""Execute the pending read-only investigation tool and checkpoint its result."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3

from tyremind.agents.evidence import load_ml_results, load_tyre_history
from tyremind.agents.investigation_state import open_state
from tyremind.agents.knowledge import search_passages
from tyremind.agents.state import InvestigationState
from tyremind.monitoring.events import transaction


def run_pending(database, index_dir, case, pending):
    name = pending['tool']
    args = pending['arguments']
    if name == 'search_manuals' and set(args) == {'query'}:
        return search_passages(index_dir, args['query'])
    if name not in ('read_tyre_history', 'read_ml_results') or args != {'tyre_id': case.tyre_id}:
        raise ValueError('Unexpected tool or arguments for this tyre.')
    with closing(sqlite3.connect(database)) as db:
        db.row_factory = sqlite3.Row
        if name == 'read_ml_results':
            return load_ml_results(db, case)
        return load_tyre_history(db, case.tyre_id, case.source_handoff['assessment']['feature_as_of'])


def execute_tool(database=Path('data/tyremind.db'), phase3=Path('data/phase3'),
                 consumer='local-phase3', index_dir=Path('data/knowledge/chroma')):
    opened = open_state(database, phase3, consumer)
    if opened['action'] == 'NO_WORK':
        return opened
    state = opened['state']
    if state['stage'] == 'TOOL_EXECUTED':
        return {'action': 'RESUMED', 'result': state['tool_results'][-1]}
    if state['stage'] != 'TOOL_SELECTED':
        raise ValueError('Select a tool with Phase 3 Step 3 first.')
    case = InvestigationState.model_validate(state['investigation'])
    pending = state['pending_tool']
    result = {**pending, 'output': run_pending(database, index_dir, case, pending)}
    with transaction(database) as db:
        row = db.execute("""SELECT s.state_json FROM phase3_agent_states s
            JOIN phase3_receipts r USING(investigation_id) JOIN investigations i USING(investigation_id)
            WHERE s.investigation_id=? AND r.consumer_id=? AND i.status='RUNNING'
            AND i.claim_token=r.claim_token""", (case.investigation_id, consumer)).fetchone()
        if row is None:
            raise ValueError('Investigation ownership changed during execution.')
        current = json.loads(row['state_json'])
        if current != state:
            if current['stage'] == 'TOOL_EXECUTED' and current['tool_call_count'] == state['tool_call_count'] + 1:
                return {'action': 'RESUMED', 'result': current['tool_results'][-1]}
            raise ValueError('Investigation changed during execution.')
        state['tool_results'].append(result)
        state['tool_call_count'] += 1
        state['stage'] = 'TOOL_EXECUTED'
        del state['pending_tool']
        db.execute('UPDATE phase3_agent_states SET state_json=? WHERE investigation_id=?',
                   (json.dumps(state, allow_nan=False), case.investigation_id))
    return {'action': 'EXECUTED', 'result': result}


if __name__ == '__main__':
    print(json.dumps(execute_tool(), indent=2))
