"""Validate durable raw batches and ingest operational records exactly once."""
import argparse
import bisect
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tyremind.data.preprocess import SCHEMAS, validate_table

AUDIT_DDL = [
    '''CREATE TABLE IF NOT EXISTS ingestion_batches (
        source_id TEXT NOT NULL, batch_id INTEGER NOT NULL, payload_sha256 TEXT NOT NULL,
        ingested_at TEXT NOT NULL, report_json TEXT NOT NULL,
        PRIMARY KEY(source_id,batch_id))''',
    '''CREATE TABLE IF NOT EXISTS ingestion_rejections (
        source_id TEXT NOT NULL,batch_id INTEGER NOT NULL,table_name TEXT NOT NULL,
        source_row INTEGER NOT NULL,original_json TEXT NOT NULL,reasons TEXT NOT NULL,
        PRIMARY KEY(source_id,batch_id,table_name,source_row),
        FOREIGN KEY(source_id,batch_id) REFERENCES ingestion_batches(source_id,batch_id))''',
]


def insert_valid_rows(connection, name, raw, parents):
    clean, rejected, validation = validate_table(name, raw, parents, include_source_rows=True)
    columns, key, _ = SCHEMAS[name]
    for column in ('timestamp',):
        if column in clean:
            clean[column] = clean[column].map(lambda value: value.isoformat())
    key_indices = [columns.index(column) for column in key]
    existing_rows = connection.execute(f'SELECT {", ".join(columns)} FROM {name}').fetchall()
    existing = {tuple(row[i] for i in key_indices): row for row in existing_rows}
    # Include historical neighbors so late/out-of-order readings cannot break mileage order.
    histories = {}
    if name == 'telemetry':
        for row in existing_rows:
            histories.setdefault(row[columns.index('tyre_id')], []).append(
                (row[columns.index('timestamp')], row[columns.index('cumulative_mileage_km')]))
        for history in histories.values():
            history.sort()
    extra_rejections, additions, unchanged = [], [], 0
    for values in clean[columns + ['_source_row']].itertuples(index=False, name=None):
        row, source_row = values[:-1], int(values[-1])
        identity = tuple(row[i] for i in key_indices)
        reason = None
        if identity in existing:
            if existing[identity] == row:
                unchanged += 1
                continue
            reason = 'existing_key_conflict'
        elif name == 'telemetry':
            tyre_id, timestamp = row[1], row[2]
            mileage = row[6]
            history = histories.setdefault(tyre_id, [])
            at = bisect.bisect_left(history, (timestamp, float('-inf')))
            if at and mileage < history[at-1][1]:
                reason = 'mileage_below_previous_record'
            elif at < len(history) and mileage > history[at][1]:
                reason = 'mileage_above_next_record'
            else:
                history.insert(at,(timestamp,mileage))
        if reason:
            original = raw.iloc[source_row-2].to_dict()
            original.update(source_row=source_row,rejection_reasons=reason)
            extra_rejections.append(original)
        else:
            additions.append(row)
            existing[identity] = row
    if extra_rejections:
        rejected = pd.concat([rejected,pd.DataFrame(extra_rejections)],ignore_index=True)
    connection.executemany(f'INSERT INTO {name} ({", ".join(columns)}) VALUES ({", ".join("?" for _ in columns)})', additions)
    report = {'input_rows':len(raw),'inserted_rows':len(additions),'unchanged_rows':unchanged,
              'rejected_rows':len(rejected),'total_rows':connection.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0],
              'rejection_reasons': rejected.rejection_reasons.value_counts().to_dict() if len(rejected) else {},
              'ignored_extra_columns':validation['ignored_extra_columns']}
    return report, rejected


def ingest_batch(database, source_id, batch):
    """Accepted rows, quarantine and batch ledger commit in a single transaction."""
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError('Operational database missing; run Phase 1 database loading first.')
    operational = {name:batch[name] for name in ('telemetry','inspections')}
    payload_hash = hashlib.sha256(json.dumps(operational,sort_keys=True,allow_nan=False).encode()).hexdigest()
    batch_id = batch['batch_id']
    with closing(sqlite3.connect(database,timeout=30)) as connection:
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('BEGIN IMMEDIATE')
        try:
            for statement in AUDIT_DDL:
                connection.execute(statement)
            previous = connection.execute('SELECT payload_sha256,report_json FROM ingestion_batches WHERE source_id=? AND batch_id=?',
                                          (source_id,batch_id)).fetchone()
            if previous:
                if previous[0] != payload_hash:
                    raise ValueError('Already ingested batch content changed; no overwrite is permitted.')
                connection.rollback()
                report = json.loads(previous[1])
                report['already_ingested'] = True
                for stats in report['tables'].values():
                    stats['original_inserted_rows'] = stats['inserted_rows']
                    stats['inserted_rows'] = 0
                return report
            parents = {name:pd.read_sql_query(f'SELECT * FROM {name}',connection) for name in ('trucks','tyres')}
            parents['tyres']['installation_timestamp'] = pd.to_datetime(parents['tyres'].installation_timestamp,utc=True)
            report = {'source_id':source_id,'batch_id':batch_id,'already_ingested':False,'imputation':'none','tables':{}}
            rejected_tables = {}
            for name in ('telemetry','inspections'):
                # An empty inspection batch still needs its schema.
                raw = pd.DataFrame(operational[name]) if operational[name] else pd.DataFrame(columns=SCHEMAS[name][0])
                stats,rejected = insert_valid_rows(connection,name,raw,parents)
                report['tables'][name] = stats
                rejected_tables[name] = rejected
            timestamp = datetime.now(timezone.utc).isoformat()
            report['ingested_at'] = timestamp
            connection.execute('INSERT INTO ingestion_batches VALUES (?,?,?,?,?)',
                               (source_id,batch_id,payload_hash,timestamp,json.dumps(report)))
            for name, rejected in rejected_tables.items():
                for row in json.loads(rejected.to_json(orient='records')):
                    source_row = int(row.pop('source_row'))
                    reasons = row.pop('rejection_reasons')
                    connection.execute('INSERT INTO ingestion_rejections VALUES (?,?,?,?,?,?)',
                                       (source_id,batch_id,name,source_row,json.dumps(row),reasons))
            if connection.execute('PRAGMA foreign_key_check').fetchall():
                raise RuntimeError('Foreign key verification failed.')
            connection.commit()
            return report
        except Exception:
            connection.rollback()
            raise


def ingest_pending(simulator=Path('data/simulator'), database=Path('data/tyremind.db')):
    simulator,database = Path(simulator),Path(database)
    raw_database = simulator/'simulator.db'
    if not raw_database.is_file():
        return {'processed_batches':0,'telemetry_inserted':0,'inspections_inserted':0,'rejected_rows':0,'batches':[]}
    # Read-only source; hidden truth is deliberately excluded from operational payloads.
    with closing(sqlite3.connect(raw_database.resolve().as_uri()+'?mode=ro',uri=True)) as connection:
        saved = connection.execute('SELECT payload FROM state WHERE id=1').fetchone()
        if saved is None:
            raise ValueError('Simulator checkpoint missing.')
        state = json.loads(saved[0])
        identity = {key:state[key] for key in ('version','seed','historical_config','historical_labels_sha256')}
        source_id = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        available = connection.execute('SELECT batch_id,payload FROM batches WHERE exported=1 ORDER BY batch_id').fetchall()
    # Recheck committed payloads too, so a source edit cannot silently change history.
    reports = []
    for batch_id,payload in available:
        batch=json.loads(payload)
        if batch['batch_id'] != batch_id:
            raise ValueError('Raw batch identity mismatch.')
        reports.append(ingest_batch(database,source_id,batch))
    new = [report for report in reports if not report['already_ingested']]
    return {'processed_batches':len(new),'telemetry_inserted':sum(r['tables']['telemetry']['inserted_rows'] for r in new),
            'inspections_inserted':sum(r['tables']['inspections']['inserted_rows'] for r in new),
            'rejected_rows':sum(sum(t['rejected_rows'] for t in r['tables'].values()) for r in new),
            'batches':new,'source_id':source_id}


def run_ingestion_cycle(source=Path('data/tyremind.db'),history=Path('data/generated'),
                        simulator=Path('data/simulator'),seed=43):
    from tyremind.monitoring.simulator import generate_next_batch
    backlog = ingest_pending(simulator,source)
    generated = generate_next_batch(source,history,simulator,seed)
    current = ingest_pending(simulator,source)
    return {**generated,'mode':'validated_telemetry','operational_database_updated':True,
            'telemetry_inserted':backlog['telemetry_inserted']+current['telemetry_inserted'],
            'inspections_inserted':backlog['inspections_inserted']+current['inspections_inserted'],
            'rejected_rows':backlog['rejected_rows']+current['rejected_rows'],
            'ingested_batches':backlog['processed_batches']+current['processed_batches'],
            'message':'Valid readings stored in operational SQLite; ML inference is not connected yet.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator',type=Path,default=Path('data/simulator'))
    parser.add_argument('--database',type=Path,default=Path('data/tyremind.db'))
    args=parser.parse_args()
    print(json.dumps(ingest_pending(args.simulator,args.database),indent=2))


if __name__=='__main__':
    main()
