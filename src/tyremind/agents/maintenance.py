"""Prepare a maintenance plan for human review."""
import argparse
import json
from pathlib import Path

from tyremind.agents.root_cause import analyze_root_cause
from tyremind.agents.retrieval import retrieve_for_investigation
from tyremind.agents.llm import request_text
from tyremind.agents.supervisor import digest, export_json
from tyremind.monitoring.events import transaction


def plan_maintenance(database=Path('data/tyremind.db'), consumer_id='local-phase3',
                     root=Path('data/phase3'), index_dir=Path('data/knowledge/chroma')):
    causes = analyze_root_cause(database, consumer_id, root, index_dir)
    if causes['action'] in ('NO_WORK', 'AWAITING_DATA_REVIEW'):
        return causes
    identity = causes['investigation_id']
    rag = retrieve_for_investigation(database=database, consumer_id=consumer_id, root=root, index_dir=index_dir)
    if rag.get('investigation_id') != identity or rag.get('tyre_id') != causes['tyre_id']:
        raise ValueError('Maintenance inputs belong to different investigations.')
    sources = [{key: p[key] for key in ('text', 'reference', 'url', 'chunk_id')} for p in rag['passages']]
    with transaction(database) as connection:
        row = connection.execute("""SELECT e.payload_json,e.payload_sha256,j.priority
            FROM phase3_evidence e JOIN investigations j USING(investigation_id)
            JOIN phase3_receipts r USING(investigation_id) WHERE investigation_id=?
            AND j.status='RUNNING' AND j.claim_token=r.claim_token AND r.consumer_id=?""",
            (identity, consumer_id)).fetchone()
        if not row or digest(row['payload_json']) != row['payload_sha256']:
            raise ValueError('Investigation ownership or evidence checksum changed.')
        facts = json.loads(row['payload_json'])
        inputs = dict(unverified_hypotheses=causes['analysis'], passages=sources,
                      priority=row['priority'], cutoff=facts['cutoff'],
                      readings=facts['history']['telemetry'][-7:], inspections=facts['history']['inspections'][-2:],
                      recent_maintenance=facts['history']['maintenance_events'][-3:], prediction=facts['prediction'])
        serialized = json.dumps(inputs, sort_keys=True, allow_nan=False)
        fingerprint = digest(serialized)
        connection.execute("""CREATE TABLE IF NOT EXISTS phase3_maintenance_plans (
            investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
            input_sha256 TEXT NOT NULL, draft TEXT NOT NULL)""")
        saved = connection.execute('SELECT * FROM phase3_maintenance_plans WHERE investigation_id=?',
                                   (identity,)).fetchone()
        if saved:
            if saved['input_sha256'] != fingerprint:
                raise ValueError('Saved plan references different investigation inputs.')
            plan = saved['draft']
        else:
            prompt = ('Draft a prioritized checklist for a qualified maintenance reviewer. For each proposed '
                      'check give the observed reason, verification needed and applicable supplied manual/page. '
                      'Treat root-cause statements as unverified. Terrain is unmeasured; missing service data '
                      'cannot establish neglect; anomaly scores do not diagnose damage. Flag forecast tread '
                      'above measured tread for verification. Identify missing evidence. Preserve event priority. '
                      'Do not invent service limits or intervals, prescribe repairs as established needs, '
                      'authorize driving or claim that work was performed. Inputs:\n' + serialized)
            plan = request_text('maintenance_planning', prompt)
            connection.execute('INSERT INTO phase3_maintenance_plans VALUES (?,?,?)', (identity, fingerprint, plan))
    result = dict(action='RESUMED' if saved else 'PLANNED', stage='AWAITING_HUMAN_REVIEW',
                  investigation_id=identity, tyre_id=causes['tyre_id'], cutoff=causes['cutoff'],
                  priority=row['priority'], draft=plan,
                  references=[{k: s[k] for k in ('chunk_id','reference','url')} for s in sources])
    export_json(root/'maintenance'/f'{identity}.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--consumer-id', default='local-phase3')
    parser.add_argument('--index-dir', type=Path, default=Path('data/knowledge/chroma'))
    print(json.dumps(plan_maintenance(**vars(parser.parse_args())), indent=2))
