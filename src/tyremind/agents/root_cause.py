"""Investigate plausible causes using the Health Agent and cited manual passages."""
import argparse
import json
from pathlib import Path

from tyremind.agents.health import assess_health
from tyremind.agents.llm import request_text
from tyremind.agents.retrieval import retrieve_for_investigation
from tyremind.agents.supervisor import digest, export_json
from tyremind.monitoring.events import transaction


def analyze_root_cause(database=Path('data/tyremind.db'), consumer_id='local-phase3',
                       root=Path('data/phase3'), index_dir=Path('data/knowledge/chroma')):
    health = assess_health(database, consumer_id, root)
    if health['action'] in ('NO_WORK', 'AWAITING_DATA_REVIEW'):
        return health
    identity = health['investigation_id']
    rag = retrieve_for_investigation(database=database, consumer_id=consumer_id, root=root, index_dir=index_dir)
    if rag.get('investigation_id') != identity or rag.get('tyre_id') != health['tyre_id']:
        raise ValueError('Health assessment and retrieved passages must belong to the same investigation.')
    sources = [{'chunk_id': p['chunk_id'], 'reference': p['reference'], 'url': p['url'],
                'text': p['text']} for p in rag['passages']]
    with transaction(database) as connection:
        evidence = connection.execute("""SELECT e.* FROM phase3_evidence e JOIN investigations j USING(investigation_id)
            JOIN phase3_receipts r USING(investigation_id) WHERE investigation_id=? AND j.status='RUNNING'
            AND r.consumer_id=? AND r.claim_token=j.claim_token""", (identity, consumer_id)).fetchone()
        if evidence is None or digest(evidence['payload_json']) != evidence['payload_sha256']:
            raise ValueError('Investigation ownership or evidence checksum changed.')
        facts = json.loads(evidence['payload_json'])
        inputs = dict(health_draft=health['assessment'], cutoff=facts['cutoff'],
                      readings=facts['history']['telemetry'][-7:], inspections=facts['history']['inspections'][-2:],
                      prediction=facts['prediction'], manual_passages=sources)
        serialized = json.dumps(inputs, sort_keys=True, allow_nan=False)
        fingerprint = digest(serialized)
        connection.execute("""CREATE TABLE IF NOT EXISTS phase3_root_causes (
            investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id),
            input_sha256 TEXT NOT NULL, analysis TEXT NOT NULL)""")
        saved = connection.execute('SELECT * FROM phase3_root_causes WHERE investigation_id=?', (identity,)).fetchone()
        if saved:
            if saved['input_sha256'] != fingerprint:
                raise ValueError('Saved root-cause analysis references different inputs.')
            analysis = saved['analysis']
        else:
            prompt = ('Draft plausible root-cause hypotheses for human review. For each, explain the supporting '
                      'measurement, conflicting or missing evidence, and what would distinguish it. Cite applicable '
                      'manual excerpts by title and PDF page. Treat the health draft as unverified. Prioritize dated '
                      'measurements over draft claims. Flag a forecast above measured tread; an anomaly score is '
                      'not confidence. Manuals are general context, not proof. If evidence is insufficient, say so. '
                      'Do not invent terrain, probabilities or confirm a diagnosis. Input data follows:\n' + serialized)
            analysis = request_text('root_cause_analysis', prompt)
            connection.execute('INSERT INTO phase3_root_causes VALUES (?,?,?)', (identity, fingerprint, analysis))
    result = dict(action='RESUMED' if saved else 'ANALYZED', investigation_id=identity, tyre_id=health['tyre_id'],
                  cutoff=health['cutoff'], analysis=analysis, references=[
                      {k: p[k] for k in ('chunk_id', 'reference', 'url')} for p in sources])
    export_json(root/'root_cause'/f'{identity}.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--consumer-id', default='local-phase3')
    parser.add_argument('--index-dir', type=Path, default=Path('data/knowledge/chroma'))
    print(json.dumps(analyze_root_cause(**vars(parser.parse_args())), indent=2))
