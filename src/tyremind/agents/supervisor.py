"""Start a bounded LangGraph investigation workflow from a durable receipt."""
import argparse
import hashlib
import json
import uuid
from pathlib import Path
from typing import Literal

from langgraph.graph import StateGraph, START, END
from pydantic import Field
from typing_extensions import TypedDict

from tyremind.agents.receive import receive_next
from tyremind.agents.state import InvestigationState, StrictModel, validate_handoff
from tyremind.monitoring.events import transaction, now

SUPERVISOR_VERSION = '1.0'
SUPERVISOR_DDL = '''CREATE TABLE IF NOT EXISTS phase3_supervisor_states (
 investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
 receipt_sha256 TEXT NOT NULL,supervisor_version TEXT NOT NULL,
 state_json TEXT NOT NULL,state_sha256 TEXT NOT NULL,created_at TEXT NOT NULL)'''
TASKS = ['HEALTH_ASSESSMENT', 'ROOT_CAUSE_ANALYSIS', 'RETRIEVE_TECHNICAL_EVIDENCE',
         'MAINTENANCE_PLANNING', 'HUMAN_REVIEW']


class GraphState(TypedDict, total=False):
    investigation: dict
    review_reasons: list[str]
    task_plan: list[str]
    next_task: str
    stage: str
    trace: list[str]
    supervisor_version: str


class SupervisorCheckpoint(StrictModel):
    investigation: InvestigationState
    review_reasons: list[str]
    task_plan: list[str] = Field(min_length=1)
    next_task: Literal['HEALTH_ASSESSMENT', 'DATA_REVIEW']
    stage: Literal['AWAITING_HEALTH_ASSESSMENT', 'AWAITING_DATA_REVIEW']
    trace: list[str]
    supervisor_version: Literal['1.0']


def validate_event(state: GraphState):
    received = InvestigationState.model_validate(state['investigation'])
    message = validate_handoff(received.source_handoff)
    expected = {'investigation_id': message.investigation.investigation_id,
                'event_id': message.event.event_id, 'tyre_id': message.event.tyre_id,
                'truck_id': message.event.truck_id, 'handoff_id': message.handoff_id,
                'target_revision': message.investigation.target_revision,
                'priority': message.investigation.priority, 'risk_level': message.assessment.risk_level}
    if any(getattr(received, key) != value for key, value in expected.items()):
        raise ValueError('Received state differs from its source handoff.')
    evidence = message.evidence
    expected_telemetry = {key: value for key, value in evidence.items()
                          if key not in ('prediction', 'feature_values')}
    if received.telemetry != expected_telemetry or received.ml_prediction != evidence.get('prediction'):
        raise ValueError('Received evidence differs from its source handoff.')
    reasons = list(message.data_issues)
    if message.assessment.data_status != 'complete':
        reasons.append('Actionable assessment has partial evidence.')
    if message.event.current_level not in ('High', 'Critical'):
        reasons.append('Case snapshot no longer has a High/Critical current level; review current status.')
    # A newer live reading never silently replaces the immutable evidence snapshot.
    return {'review_reasons': list(dict.fromkeys(reasons)),
            'trace': ['validate_event'], 'supervisor_version': SUPERVISOR_VERSION}


def supervise(state: GraphState):
    needs_review = bool(state['review_reasons'])
    return {'next_task': 'DATA_REVIEW' if needs_review else 'HEALTH_ASSESSMENT',
            'task_plan': (['DATA_REVIEW'] if needs_review else []) + TASKS,
            'trace': state['trace'] + ['supervisor']}


def prepare_health_assessment(state: GraphState):
    return {'stage': 'AWAITING_HEALTH_ASSESSMENT',
            'trace': state['trace'] + ['prepare_health_assessment']}


def request_data_review(state: GraphState):
    return {'stage': 'AWAITING_DATA_REVIEW',
            'trace': state['trace'] + ['request_data_review']}


def build_supervisor_graph():
    graph = StateGraph(GraphState)
    graph.add_node('validate_event', validate_event)
    graph.add_node('supervisor', supervise)
    graph.add_node('prepare_health_assessment', prepare_health_assessment)
    graph.add_node('request_data_review', request_data_review)
    graph.add_edge(START, 'validate_event')
    graph.add_edge('validate_event', 'supervisor')
    graph.add_conditional_edges('supervisor', lambda state: state['next_task'],
        {'HEALTH_ASSESSMENT': 'prepare_health_assessment', 'DATA_REVIEW': 'request_data_review'})
    graph.add_edge('prepare_health_assessment', END)
    graph.add_edge('request_data_review', END)
    return graph.compile()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def export_json(target, value):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def start_investigation(database=Path('data/tyremind.db'), output=Path('data/phase3/supervisor'),
                        consumer_id='local-phase3', phase3_output=Path('data/phase3')):
    # Reuse the same consumer and its ownership; never claim an additional job on rerun.
    receipt_report = receive_next(database, Path(phase3_output)/'received', consumer_id, phase3_output)
    report = {'action': 'NO_WORK', 'consumer_id': consumer_id, 'investigation_id': None,
              'supervisor_executed': False, 'specialist_agents_executed': False, 'llm_calls': 0}
    if receipt_report['action'] != 'NO_WORK':
        identity = receipt_report['investigation_id']
        with transaction(database) as connection:
            connection.execute(SUPERVISOR_DDL)
            receipt = connection.execute('SELECT * FROM phase3_receipts WHERE investigation_id=?', (identity,)).fetchone()
            job = connection.execute('SELECT * FROM investigations WHERE investigation_id=?', (identity,)).fetchone()
            if (receipt['consumer_id'] != consumer_id or receipt['stage'] != 'RECEIVED'
                    or job['status'] != 'RUNNING' or receipt['claim_token'] != job['claim_token']):
                raise ValueError('Investigation ownership changed before supervisor execution.')
            if digest(receipt['state_json']) != receipt['state_sha256']:
                raise ValueError('Receipt checksum mismatch.')
            initial = InvestigationState.model_validate_json(receipt['state_json'])
            if initial.investigation_id != identity:
                raise ValueError('Receipt identity mismatch.')
            saved = connection.execute('SELECT * FROM phase3_supervisor_states WHERE investigation_id=?', (identity,)).fetchone()
            if saved:
                if saved['supervisor_version'] != SUPERVISOR_VERSION or saved['receipt_sha256'] != receipt['state_sha256']:
                    raise ValueError('Saved supervisor version/input mismatch; explicit migration required.')
                if digest(saved['state_json']) != saved['state_sha256']:
                    raise ValueError('Supervisor checkpoint checksum mismatch.')
                checkpoint = SupervisorCheckpoint.model_validate_json(saved['state_json'])
                if checkpoint.investigation != initial:
                    raise ValueError('Supervisor checkpoint source mismatch.')
                action = 'RESUMED'
            else:
                result = build_supervisor_graph().invoke({'investigation': initial.model_dump(mode='json')},
                                                         config={'recursion_limit': 8})
                checkpoint = SupervisorCheckpoint.model_validate(result)
                serialized = json.dumps(checkpoint.model_dump(mode='json'), sort_keys=True, allow_nan=False)
                connection.execute('INSERT INTO phase3_supervisor_states VALUES (?,?,?,?,?,?)',
                    (identity, receipt['state_sha256'], SUPERVISOR_VERSION, serialized, digest(serialized), now()))
                action = 'STARTED'
        state_path = Path(output)/f'{identity}.json'
        export_json(state_path, checkpoint.model_dump(mode='json'))
        report.update(action=action, investigation_id=identity, tyre_id=initial.tyre_id,
                      priority=initial.priority, risk_level=initial.risk_level, stage=checkpoint.stage,
                      next_task=checkpoint.next_task, task_plan=checkpoint.task_plan,
                      review_reasons=checkpoint.review_reasons, trace=checkpoint.trace,
                      supervisor_executed=action == 'STARTED', state_path=str(state_path),
                      note='Supervisor routing is saved; specialist execution is the next implementation step.')
    export_json(Path(output)/'supervisor_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--output', type=Path, default=Path('data/phase3/supervisor'))
    parser.add_argument('--phase3-output', type=Path, default=Path('data/phase3'))
    parser.add_argument('--consumer-id', default='local-phase3')
    args = parser.parse_args()
    print(json.dumps(start_investigation(args.database, args.output, args.consumer_id, args.phase3_output), indent=2))


if __name__ == '__main__':
    main()
