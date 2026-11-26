"""Open or resume the agent's working state from the Phase 3 receipt."""
import json
from pathlib import Path
from typing import TypedDict

from tyremind.agents.receive import receive_next
from tyremind.agents.supervisor import digest, validate_event
from tyremind.monitoring.events import transaction


class AgentState(TypedDict):
    investigation: dict
    tool_results: list[dict]
    evidence: list[dict]
    tool_call_count: int
    stage: str


def open_state(database=Path('data/tyremind.db'), phase3=Path('data/phase3'),
               consumer='local-phase3'):
    report = receive_next(database, phase3 / 'received', consumer, phase3)
    if report['action'] == 'NO_WORK':
        return {'action': 'NO_WORK'}
    identity = report['investigation_id']
    with transaction(database) as db:
        receipt = db.execute("""
            SELECT r.* FROM phase3_receipts r JOIN investigations i USING(investigation_id)
            WHERE r.investigation_id=? AND r.consumer_id=? AND r.claim_token=i.claim_token
            AND i.status='RUNNING'
        """, (identity, consumer)).fetchone()
        if receipt is None or digest(receipt['state_json']) != receipt['state_sha256']:
            raise ValueError('Invalid receipt or investigation ownership.')
        initial = json.loads(receipt['state_json'])
        validate_event({'investigation': initial})
        if initial['investigation_id'] != identity:
            raise ValueError('Receipt identity mismatch.')
        db.execute("""CREATE TABLE IF NOT EXISTS phase3_agent_states (
            investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
            state_json TEXT NOT NULL)""")
        row = db.execute('SELECT state_json FROM phase3_agent_states WHERE investigation_id=?',
                         (identity,)).fetchone()
        if row:
            state = json.loads(row['state_json'])
            if state['investigation'] != initial:
                raise ValueError('Saved investigation differs from the received alert.')
            return {'action': 'RESUMED', 'state': state}
        state: AgentState = dict(investigation=initial, tool_results=[], evidence=[],
                                 tool_call_count=0, stage='READY_FOR_TOOL_SELECTION')
        db.execute('INSERT INTO phase3_agent_states VALUES (?,?)', (identity, json.dumps(state)))
        return {'action': 'CREATED', 'state': state}


if __name__ == '__main__':
    print(json.dumps(open_state(), indent=2))
