"""Resume a bounded investigation from the last committed agent stage."""
import argparse
import json
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tyremind.agents.execute_tool import execute_tool
from tyremind.agents.investigation_state import open_state
from tyremind.agents.review_evidence import review_evidence
from tyremind.agents.select_tool import select_tool
from tyremind.monitoring.events import transaction


class LoopStatus(TypedDict, total=False):
    stage: str
    investigation_id: str
    tool_call_count: int
    last_review: dict
    reason: str


def run_loop(database=Path('data/tyremind.db'), phase3=Path('data/phase3'),
             consumer='local-phase3', local_model=None, max_calls=4,
             index_dir=Path('data/knowledge/chroma')):
    if max_calls < 1:
        raise ValueError('max_calls must be positive.')
    args = (database, phase3, consumer)

    def checkpoint(status):
        opened = open_state(*args)
        if opened['action'] == 'NO_WORK':
            return {'stage': 'NO_WORK'}
        state = opened['state']
        identity = state['investigation']['investigation_id']
        if status.get('investigation_id', identity) != identity:
            raise ValueError('Active investigation changed. Restart to open the next investigation.')
        stage = state['stage']
        count = state['tool_call_count']
        if stage in ('READY_FOR_TOOL_SELECTION', 'TOOL_SELECTED') and count >= max_calls:
            reason = f'Evidence remains insufficient after {count} tool calls. Human review required.'
            with transaction(database) as db:
                changed = db.execute("""UPDATE phase3_agent_states SET state_json=json_set(
                    state_json,'$.stage','NEEDS_HUMAN_REVIEW','$.stop_reason',?)
                    WHERE investigation_id=? AND json_extract(state_json,'$.stage')=?
                    AND json_extract(state_json,'$.tool_call_count')=? AND EXISTS (
                    SELECT 1 FROM investigations i JOIN phase3_receipts r USING(investigation_id)
                    WHERE i.investigation_id=? AND i.status='RUNNING' AND r.consumer_id=?
                    AND i.claim_token=r.claim_token)""", (reason, identity, stage, count, identity, consumer))
                if changed.rowcount != 1:
                    raise ValueError('Investigation changed while stopping. Resume from checkpoint.')
            stage = 'NEEDS_HUMAN_REVIEW'
            state['stop_reason'] = reason
        return dict(stage=stage, investigation_id=identity, tool_call_count=count,
                    last_review=state['evidence'][-1] if state['evidence'] else {},
                    reason=state.get('stop_reason', ''))

    def select(status):
        select_tool(*args, local_model=local_model)
        return {}

    def execute(status):
        execute_tool(*args, index_dir=index_dir)
        return {}

    def review(status):
        review_evidence(*args, local_model=local_model)
        return {}

    graph = StateGraph(LoopStatus)
    graph.add_node('checkpoint', checkpoint)
    for name, node in [('select', select), ('execute', execute), ('review', review)]:
        graph.add_node(name, node)
        graph.add_edge(name, 'checkpoint')
    graph.add_edge(START, 'checkpoint')
    graph.add_conditional_edges('checkpoint', lambda status: status['stage'], {
        'READY_FOR_TOOL_SELECTION': 'select', 'TOOL_SELECTED': 'execute',
        'TOOL_EXECUTED': 'review', 'READY_FOR_FINDINGS': END,
        'NEEDS_HUMAN_REVIEW': END, 'NO_WORK': END})
    return graph.compile().invoke({}, config={'recursion_limit': 6 * max_calls + 10})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-ollama', metavar='MODEL')
    parser.add_argument('--max-calls', type=int, default=4)
    options = parser.parse_args()
    print(json.dumps(run_loop(local_model=options.local_ollama, max_calls=options.max_calls), indent=2))
