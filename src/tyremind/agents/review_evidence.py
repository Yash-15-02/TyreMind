"""Review the latest completed tool call and save the next investigation stage."""
import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from tyremind.agents.investigation_state import open_state
from tyremind.agents.llm import request_text
from tyremind.agents.select_tool import request_local_json
from tyremind.agents.state import StrictModel
from tyremind.monitoring.events import transaction


class Review(StrictModel):
    finding: str = Field(min_length=1, max_length=1200)
    reason: str = Field(min_length=1, max_length=800)
    decision: Literal['MORE_EVIDENCE', 'ENOUGH_EVIDENCE']


def review_evidence(database=Path('data/tyremind.db'), phase3=Path('data/phase3'),
                    consumer='local-phase3', local_model=None):
    opened = open_state(database, phase3, consumer)
    if opened['action'] == 'NO_WORK':
        return opened
    state = opened['state']
    call = state['tool_call_count']
    if state['evidence'] and state['evidence'][-1].get('call_number') == call:
        return {'action': 'RESUMED', 'review': state['evidence'][-1], 'stage': state['stage']}
    if state['stage'] != 'TOOL_EXECUTED':
        raise ValueError('Execute the selected tool before reviewing its result.')
    result = state['tool_results'][-1]
    output = result['output']
    if result['tool'] == 'read_tyre_history':
        # Bound the model context and disclose the sampling window.
        output = {table: {'total_rows': len(rows), 'latest_rows': rows[-10:]}
                  for table, rows in output.items()}
    case = state['investigation']
    prompt = ('Review the evidence for this alert. Explain what it supports and what remains '
              'uncertain. Manual guidance alone cannot establish a tyre fault. Unused tools are '
              'unexamined. Decide whether to gather more evidence or draft findings. Return JSON '
              'matching this schema: ' + json.dumps(Review.model_json_schema()) + '\n' +
              json.dumps({'tyre': case['tyre_id'], 'alert': case['source_handoff']['assessment'],
                          'previous_reviews': state['evidence'], 'tool': result['tool'], 'output': output}))
    content = (request_local_json(local_model, prompt, Review.model_json_schema()) if local_model
               else request_text('evidence_review', prompt))
    review = Review.model_validate_json(content).model_dump()
    review.update(call_number=call, tool=result['tool'])
    stage = ('READY_FOR_TOOL_SELECTION' if review['decision'] == 'MORE_EVIDENCE'
             else 'READY_FOR_FINDINGS')
    with transaction(database) as db:
        row = db.execute("""SELECT s.state_json FROM phase3_agent_states s
            JOIN phase3_receipts r USING(investigation_id) JOIN investigations i USING(investigation_id)
            WHERE s.investigation_id=? AND r.consumer_id=? AND i.status='RUNNING'
            AND i.claim_token=r.claim_token""", (case['investigation_id'], consumer)).fetchone()
        if row is None:
            raise ValueError('Investigation ownership changed during review.')
        current = json.loads(row['state_json'])
        if current != state:
            for saved in current['evidence']:
                if saved.get('call_number') == call:
                    return {'action': 'RESUMED', 'review': saved, 'stage': current['stage']}
            raise ValueError('Investigation changed during review. Retry.')
        state['evidence'].append(review)
        state['stage'] = stage
        db.execute('UPDATE phase3_agent_states SET state_json=? WHERE investigation_id=?',
                   (json.dumps(state), case['investigation_id']))
    return {'action': 'REVIEWED', 'review': review, 'stage': stage}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-ollama', metavar='MODEL')
    print(json.dumps(review_evidence(local_model=parser.parse_args().local_ollama), indent=2))
