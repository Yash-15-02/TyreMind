"""Evaluate saved fleet outputs against ingested simulation outcomes and human decisions."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3


def ratio(a, b):
    return a / b if b else None


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def simulation_labels(db, simulator, tables):
    labels = {}
    if not Path(simulator).is_file() or 'ingestion_batches' not in tables:
        return labels
    with closing(readonly(simulator)) as source:
        source.execute('BEGIN')
        row = source.execute('SELECT payload FROM state WHERE id=1').fetchone()
        if not row:
            return labels
        state = json.loads(row[0])
        identity = {k: state[k] for k in ('version','seed','historical_config','historical_labels_sha256')}
        source_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        ledger = dict(db.execute('SELECT batch_id,payload_sha256 FROM ingestion_batches WHERE source_id=?', (source_id,)))
        for batch_id, payload in source.execute('SELECT batch_id,payload FROM batches WHERE exported=1 ORDER BY batch_id'):
            if batch_id not in ledger:
                continue
            batch = json.loads(payload)
            operational = {k: batch[k] for k in ('telemetry','inspections')}
            if hashlib.sha256(json.dumps(operational, sort_keys=True, allow_nan=False).encode()).hexdigest() != ledger[batch_id]:
                raise ValueError('Simulator batch differs from the ingestion ledger.')
            for row in batch['simulation_truth']:
                labels[row['tyre_id'],row['timestamp']] = row
    return labels


def evaluate(database=Path('data/tyremind.db'), simulator=Path('data/simulator/simulator.db')):
    with closing(readonly(database)) as db:
        db.row_factory = sqlite3.Row
        db.execute('BEGIN')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        labels = simulation_labels(db, simulator, tables)
        mileage = {}
        curves = defaultdict(lambda: ([], []))
        for r in db.execute('SELECT tyre_id,timestamp,cumulative_mileage_km FROM telemetry ORDER BY tyre_id,timestamp'):
            key = (r[0], r[1])
            if key in labels:
                mileage[key] = r[2]
                x,y = curves[r[0]]
                x.append(r[2])
                y.append(labels[key]['simulated_true_tread_depth_mm'])
        counts = Counter()
        for r in db.execute("""SELECT tyre_id,timestamp,risk_level FROM (SELECT *,
                row_number() OVER(PARTITION BY tyre_id,timestamp ORDER BY feature_as_of DESC, assessed_at DESC,rowid DESC) n
                FROM risk_assessments) WHERE n=1"""):
            key = (r[0], r[1])
            if key not in mileage or r[2] not in ('High','Critical','Normal','Medium'):
                counts['unscored'] += 1
                continue
            truth = labels[key]['scenario_active'] or labels[key]['overload_active']
            counts[('tp' if truth else 'fp') if r[2] in ('High','Critical') else ('fn' if truth else 'tn')] += 1
        errors = []
        pending = unavailable = 0
        for r in db.execute("""SELECT tyre_id,timestamp,predicted_tread_depth_5000km_mm FROM predictions
                WHERE status='scored' AND predicted_tread_depth_5000km_mm IS NOT NULL"""):
            key = (r[0], r[1])
            if key not in mileage:
                unavailable += 1
                continue
            target = mileage[key] + 5000
            x,y = curves[r[0]]
            j = bisect_left(x,target)
            if j == len(x):
                pending += 1
                continue
            actual = y[j] if x[j] == target else y[j-1]+(y[j]-y[j-1])*(target-x[j-1])/(x[j]-x[j-1])
            errors.append(r[2]-actual)
        outcomes = Counter()
        reviews = Counter()
        latest = Counter()
        if 'phase4_inbox' in tables:
            outcomes.update(r[0] for r in db.execute('SELECT message_type FROM phase4_inbox'))
        if 'fleet_manager_reviews' in tables:
            reviews.update(r[0] for r in db.execute('SELECT decision FROM fleet_manager_reviews'))
        if 'maintenance_history' in tables:
            latest.update(r[0] for r in db.execute("""SELECT status FROM (SELECT status,row_number() OVER(
                PARTITION BY investigation_id ORDER BY recorded_at DESC,rowid DESC) n FROM maintenance_history
                WHERE status IN ('COMPLETED','FOLLOW_UP_REQUIRED')) WHERE n=1"""))
    tp,fp,fn,tn = (counts[k] for k in ('tp','fp','fn','tn'))
    approved,rejected = reviews['APPROVED'],reviews['REJECTED']
    return dict(alerts=dict(reference='High/Critical vs active simulated fault or overload, one assessment per tyre and reading',
                            matched=tp+fp+fn+tn, unscored=counts['unscored'],true_positive=tp,false_positive=fp,
                            false_negative=fn,true_negative=tn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn)),
                tread_forecasts=dict(reference='Saved scored forecasts vs simulated tread interpolated at +5,000 km',
                                     evaluated=len(errors),awaiting_distance=pending,missing_reference=unavailable,
                                     mae_mm=ratio(sum(map(abs,errors)),len(errors)),
                                     rmse_mm=math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None),
                recommendations=dict(published=sum(outcomes.values()),validation_passed=outcomes['INVESTIGATION_READY_FOR_REVIEW'],
                                     validation_failed=outcomes['INVESTIGATION_VALIDATION_FAILED'],manager_decisions=dict(reviews),
                                     approval_rate_among_yes_no=ratio(approved,approved+rejected),
                                     latest_technician_outcomes=dict(latest)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--simulator', type=Path, default=Path('data/simulator/simulator.db'))
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))
