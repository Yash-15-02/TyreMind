"""Check the investigation report and Step 9 draft before human review."""
import argparse
import json
import re
from pathlib import Path

from tyremind.agents.maintenance import plan_maintenance
from tyremind.agents.supervisor import digest, export_json
from tyremind.monitoring.events import transaction


def inspect_sections(sections, facts, references):
    inspections = facts['history']['inspections']
    measured = inspections[-1]['tread_depth_mm'] if inspections else None
    forecast = (facts['prediction'] or {}).get('predicted_tread_depth_5000km_mm')
    findings = []
    rules = [
        ('WRONG_GAUGE', r'tread depth (?:using|with) (?:a |an )?(?:tyre |tire )?pressure gauge', 'Use a tread-depth gauge for this measurement.'),
        ('UNSUPPORTED_CLEARANCE', r'authorize driving|safe to drive|cleared for service|claim work performed', 'Driving clearance and completed work require independent verification.'),
        ('UNMEASURED_TERRAIN', r'(?:varied|various) terrain', 'The operational evidence has no terrain measurement.'),
        ('ANOMALY_CONFIDENCE', r'moderately confident', 'The anomaly score measures unusualness and does not establish confidence.'),
    ]
    for section, content in sections.items():
        if not content.strip():
            findings.append(dict(section=section, code='EMPTY_SECTION', detail='Required text is empty.'))
        for line in content.splitlines():
            for code, pattern, detail in rules:
                if re.search(pattern, line, re.I) and not re.search(r'\b(?:not|never|cannot|do not)\b[^.!?]{0,35}(?:' + pattern + ')', line, re.I):
                    findings.append(dict(section=section, code=code, excerpt=line.strip(), detail=detail))
            match = re.search(r'tread depth (?:is )?below (?:[\d/]+ inch(?:es)?\s*\()?(\d+(?:\.\d+)?)\s*mm', line, re.I)
            if match and measured is not None and measured >= float(match[1]):
                findings.append(dict(section=section, code='CONTRADICTED_TREAD', excerpt=line.strip(),
                                     detail=f'Latest recorded inspection measured {measured} mm.'))
    if not references or any(not all(ref.get(k) for k in ('chunk_id','reference','url')) for ref in references):
        findings.append(dict(section='references', code='MISSING_SOURCE', detail='Review retrieval source metadata.'))
    observations = []
    if forecast is not None and measured is not None and forecast > measured:
        observations.append(f'Forecast {forecast:.3f} mm exceeds latest measured tread {measured:.3f} mm by {forecast-measured:.3f} mm. Verify before relying on forecast.')
    return findings, observations


def validate_report(database=Path('data/tyremind.db'), consumer_id='local-phase3',
                    root=Path('data/phase3'), index_dir=Path('data/knowledge/chroma')):
    plan = plan_maintenance(database, consumer_id, root, index_dir)
    if plan['action'] in ('NO_WORK', 'AWAITING_DATA_REVIEW'):
        return plan
    identity = plan['investigation_id']
    with transaction(database) as connection:
        row = connection.execute("""SELECT h.assessment,c.analysis,m.draft,e.payload_json,e.payload_sha256
            FROM investigations j JOIN phase3_receipts r USING(investigation_id)
            JOIN phase3_health h USING(investigation_id) JOIN phase3_root_causes c USING(investigation_id)
            JOIN phase3_maintenance_plans m USING(investigation_id) JOIN phase3_evidence e USING(investigation_id)
            WHERE investigation_id=? AND j.status='RUNNING' AND j.claim_token=r.claim_token AND r.consumer_id=?""",
            (identity, consumer_id)).fetchone()
        if not row or digest(row['payload_json']) != row['payload_sha256'] or row['draft'] != plan['draft']:
            raise ValueError('Investigation ownership or evidence changed during validation.')
        facts = json.loads(row['payload_json'])
        sections = {name: row[name] for name in ('assessment', 'analysis', 'draft')}
        findings, observations = inspect_sections(sections, facts, plan['references'])
        result = dict(investigation_id=identity, tyre_id=plan['tyre_id'], priority=plan['priority'],
                      cutoff=plan['cutoff'], stage='REVISION_REQUIRED' if findings else 'AWAITING_HUMAN_REVIEW',
                      findings=findings, evidence_notes=observations, references=plan['references'],
                      source_sha256=digest(json.dumps([sections, row['payload_sha256'], plan['references']], sort_keys=True)))
        connection.execute("""CREATE TABLE IF NOT EXISTS phase3_validations (
            investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id), result_json TEXT NOT NULL)""")
        connection.execute('INSERT INTO phase3_validations VALUES (?,?) ON CONFLICT(investigation_id) DO UPDATE SET result_json=excluded.result_json',
                           (identity, json.dumps(result, allow_nan=False)))
    export_json(root/'validation'/f'{identity}.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--consumer-id', default='local-phase3')
    parser.add_argument('--index-dir', type=Path, default=Path('data/knowledge/chroma'))
    print(json.dumps(validate_report(**vars(parser.parse_args())), indent=2))
