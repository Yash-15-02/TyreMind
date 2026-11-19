"""Save evidence-backed proposals for future TyreMind iterations."""
import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from tyremind.phase4.evaluate import evaluate


def save_improvements(database=Path('data/tyremind.db'), simulator=Path('data/simulator/simulator.db')):
    evaluation = evaluate(database, simulator)
    items = []

    def add(priority, topic, action, evidence):
        items.append(dict(priority=priority, topic=topic, action=action, evidence=evidence))

    alerts = evaluation['alerts']
    for metric, title, action in [
        ('false_negative', 'Missed simulated faults', 'Compare missed and detected readings by fault type before testing a risk-rule change.'),
        ('false_positive', 'Alerts on unlabelled conditions', 'Inspect the triggered rules and tread evidence before classifying these alerts as unnecessary.')]:
        if alerts[metric]:
            add('High' if metric == 'false_negative' else 'Normal', title, action,
                f"{alerts[metric]} of {alerts['matched']} matched readings")
    forecasts = evaluation['tread_forecasts']
    if forecasts['evaluated']:
        add('Normal', 'Forecast accuracy', 'Compare the wear model with the wear-rate baseline on these same matured targets.',
            f"{forecasts['evaluated']} forecasts; MAE {forecasts['mae_mm']:.3f} mm")
    if forecasts['awaiting_distance']:
        add('Monitor', 'Forecast follow-up', 'Repeat evaluation once the tyres accumulate another 5,000 km.',
            f"{forecasts['awaiting_distance']} forecasts awaiting distance")
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=rw', uri=True)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        grouped = {}
        if 'phase3_validations' in tables:
            for identity, raw in db.execute('SELECT investigation_id,result_json FROM phase3_validations ORDER BY investigation_id'):
                validation = json.loads(raw)
                for finding in validation['findings']:
                    key = (finding['section'], finding['code'], finding['detail'])
                    grouped.setdefault(key, []).append(identity)
                for note in validation['evidence_notes']:
                    add('High', 'Review forecast evidence', 'Check the forecast against the dated inspection before using it in a plan.',
                        dict(investigation=identity, note=note))
        for (section, code, detail), identities in sorted(grouped.items()):
            add('High', f'{section}: {code}', f'Add a regression case for the {section} generator. {detail}',
                dict(occurrences=len(identities), investigations=sorted(set(identities))))
        if 'fleet_manager_reviews' in tables:
            for identity, note in db.execute("""SELECT investigation_id,note FROM fleet_manager_reviews
                    WHERE decision IN ('REJECTED','CHANGES_REQUESTED') ORDER BY investigation_id"""):
                add('Normal', 'Manager feedback', 'Review the planning instructions against the manager feedback.',
                    dict(investigation=identity, reason=note))
        if 'maintenance_history' in tables:
            for identity, findings in db.execute("""SELECT investigation_id,actual_findings FROM (
                SELECT *,row_number() OVER(PARTITION BY investigation_id ORDER BY recorded_at DESC,rowid DESC) n
                FROM maintenance_history) WHERE n=1 AND status='FOLLOW_UP_REQUIRED' ORDER BY investigation_id"""):
                add('High', 'Technician follow-up', 'Compare the recorded findings with the recommended checks.',
                    dict(investigation=identity, findings=findings))
        items.sort(key=lambda item: ['High','Normal','Monitor'].index(item['priority']))
        payload = json.dumps(dict(evaluation=evaluation, improvements=items), sort_keys=True, allow_nan=False)
        snapshot = hashlib.sha256(payload.encode()).hexdigest()
        db.execute("""CREATE TABLE IF NOT EXISTS improvement_reports (
            snapshot TEXT PRIMARY KEY, report_json TEXT NOT NULL, saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        db.execute('INSERT OR IGNORE INTO improvement_reports(snapshot,report_json) VALUES (?,?)', (snapshot,payload))
    return dict(snapshot=snapshot, improvements=items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--simulator', type=Path, default=Path('data/simulator/simulator.db'))
    print(json.dumps(save_improvements(**vars(parser.parse_args())), indent=2))
