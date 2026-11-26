"""Validate a Step 7 outcome against the investigation and allow one correction."""
import argparse
import json
from pathlib import Path

from tyremind.agents.investigation_state import open_state
from tyremind.agents.prepare_outcome import build_outcome
from tyremind.agents.supervisor import digest
from tyremind.agents.validate import inspect_sections
from tyremind.monitoring.events import transaction


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, allow_nan=False))


def inspect(state, outcome):
    case = state['investigation']
    issues = []
    if not isinstance(outcome, dict):
        return ['Expected a draft outcome object.']
    expected = dict(investigation_id=case['investigation_id'], tyre_id=case['tyre_id'],
                    risk_level=case['risk_level'], assessment=case['source_handoff']['assessment'],
                    reviews=state['evidence'])
    for key, value in expected.items():
        if outcome.get(key) != value:
            issues.append(f'{key} differs from the investigation.')
    refs = {}
    inspections = []
    prediction = None
    for result in state['tool_results']:
        output = result['output']
        if result['tool'] == 'search_manuals':
            for passage in output['passages']:
                src = passage['source']
                refs[passage['chunk_id']] = dict(chunk_id=passage['chunk_id'], title=src['title'],
                                                page=src['pdf_page'], url=src['citation_url'])
        elif result['tool'] == 'read_tyre_history':
            inspections = output['inspections']
        elif result['tool'] == 'read_ml_results':
            prediction = output
    if outcome.get('retrieved_sources') != list(refs.values()):
        issues.append('Source references differ from the retrieved passages.')
    if state['stage'] == 'NEEDS_HUMAN_REVIEW':
        checks = outcome.get('checks')
        if (outcome.get('kind') != 'HUMAN_REVIEW' or outcome.get('reason') != state['stop_reason']
                or 'maintenance_draft' not in outcome or outcome['maintenance_draft'] is not None
                or not isinstance(checks, list) or not checks
                or not all(isinstance(c, str) and c.strip() for c in checks)):
            issues.append('Human review requires a stop reason, checks and an empty maintenance draft.')
    elif state['stage'] == 'READY_FOR_FINDINGS':
        draft = outcome.get('maintenance_draft')
        if outcome.get('kind') != 'MAINTENANCE_DRAFT' or not isinstance(draft, dict) or draft.get('status') != 'DRAFT':
            issues.append('Expected a maintenance proposal awaiting approval.')
        sections = {k: outcome.get(k) for k in ('findings', 'uncertainty')}
        sections['inspection'] = draft.get('proposed_inspection') if isinstance(draft, dict) else None
        facts = {'history': {'inspections': inspections}, 'prediction': prediction}
        sources = [dict(reference=r['title'], **r) for r in refs.values()]
        findings, _ = inspect_sections({k: v if isinstance(v, str) else '' for k, v in sections.items()},
                                       facts, sources)
        issues.extend(f['detail'] for f in findings if f['code'] != 'MISSING_SOURCE' or refs)
    else:
        issues.append('Investigation has not finished evidence gathering.')
    return issues


def validate_outcome(database=Path('data/tyremind.db'), phase3=Path('data/phase3'),
                     consumer='local-phase3', local_model=None):
    opened = open_state(database, phase3, consumer)
    if opened['action'] == 'NO_WORK':
        return opened
    state = opened['state']
    if 'draft_outcome' not in state:
        raise ValueError('Run Step 7 prepare_outcome first.')
    outcome = state['draft_outcome']
    saved = state.get('outcome_validation')
    snapshot = {k: v for k, v in state.items() if k != 'outcome_validation'}
    if saved and saved['snapshot'] == fingerprint(snapshot):
        return saved
    issues = inspect(state, outcome)
    initial_issues = issues.copy()
    if issues:
        try:
            outcome = build_outcome(state, local_model, issues)
            issues = inspect(state, outcome)
        except (ValueError, RuntimeError, OSError) as exc:
            issues.append(f'Correction failed: {type(exc).__name__}')
    snapshot['draft_outcome'] = outcome
    report = dict(status='FAILED' if issues else 'PASSED', investigation_id=state['investigation']['investigation_id'],
                  kind=outcome.get('kind') if isinstance(outcome, dict) else None,
                  correction_attempts=int(bool(initial_issues)), initial_issues=initial_issues,
                  issues=issues, snapshot=fingerprint(snapshot))
    with transaction(database) as db:
        row = db.execute("""SELECT s.state_json FROM phase3_agent_states s JOIN investigations i
            USING(investigation_id) JOIN phase3_receipts r USING(investigation_id)
            WHERE s.investigation_id=? AND i.status='RUNNING' AND i.claim_token=r.claim_token
            AND r.consumer_id=?""", (report['investigation_id'], consumer)).fetchone()
        if row is None or json.loads(row['state_json']) != state:
            raise ValueError('Investigation changed during validation. Resume the current checkpoint.')
        snapshot['outcome_validation'] = report
        db.execute('UPDATE phase3_agent_states SET state_json=? WHERE investigation_id=?',
                   (json.dumps(snapshot), report['investigation_id']))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-ollama', metavar='MODEL')
    print(json.dumps(validate_outcome(local_model=parser.parse_args().local_ollama), indent=2))
