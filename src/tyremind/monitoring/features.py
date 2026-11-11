"""Refresh latest operational features with the exact offline feature definitions."""
import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.features.build import FEATURE_COLUMNS, build_features

FEATURE_VERSION = '1.0'


def calculate_current(telemetry, tyres, inspections, maintenance, as_of=None):
    """Return one row per observed tyre, using only records at/before cutoff."""
    frames = [frame.copy() for frame in (telemetry, tyres, inspections, maintenance)]
    for index in (0,2,3):
        frames[index]['timestamp'] = pd.to_datetime(frames[index].timestamp, utc=True, errors='raise')
    if frames[0].empty:
        raise ValueError('No stored telemetry; ingest readings before building live features.')
    cutoff = pd.to_datetime(as_of,utc=True) if as_of is not None else frames[0].timestamp.max()
    if pd.isna(cutoff):
        raise ValueError('Feature cutoff must be a valid timestamp.')
    for index in (0,2,3):
        frames[index] = frames[index].loc[frames[index].timestamp <= cutoff]
    if frames[0].empty:
        raise ValueError('No telemetry at or before the requested cutoff.')
    features = build_features(*frames)
    current = features.sort_values(['tyre_id','timestamp']).groupby('tyre_id',sort=True).tail(1).reset_index(drop=True)
    if np.isinf(current[FEATURE_COLUMNS].to_numpy(dtype=float)).any():
        raise ValueError('Infinite computed feature values cannot be published.')
    current['feature_version'] = FEATURE_VERSION
    current['feature_as_of'] = cutoff
    current['is_stale'] = current.timestamp.lt(cutoff).astype(int)
    return current


def refresh_current_features(database=Path('data/tyremind.db'), output=Path('data/monitoring/features'), as_of=None):
    database,output = Path(database),Path(output)
    if not database.is_file():
        raise FileNotFoundError('Operational database missing; complete database loading/ingestion first.')
    if output.resolve() == Path('data/features').resolve():
        raise ValueError('Live output must not use the historical training feature directory.')
    with closing(sqlite3.connect(database,timeout=30)) as connection:
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('BEGIN IMMEDIATE')
        try:
            frames = [pd.read_sql_query(f'SELECT * FROM {table}',connection)
                      for table in ('telemetry','tyres','inspections','maintenance_events')]
            current = calculate_current(*frames, as_of=as_of)
            cutoff = current.feature_as_of.iloc[0].isoformat()
            numeric_columns = ', '.join(f'{column} REAL' for column in FEATURE_COLUMNS)
            connection.execute(f'''CREATE TABLE IF NOT EXISTS current_features (
                tyre_id TEXT PRIMARY KEY NOT NULL REFERENCES tyres(tyre_id),
                truck_id TEXT NOT NULL REFERENCES trucks(truck_id),timestamp TEXT NOT NULL,
                {numeric_columns},feature_version TEXT NOT NULL,feature_as_of TEXT NOT NULL,
                is_stale INTEGER NOT NULL CHECK(is_stale IN (0,1)),computed_at TEXT NOT NULL,
                FOREIGN KEY(tyre_id,truck_id) REFERENCES tyres(tyre_id,truck_id))''')
            previous = connection.execute('SELECT MAX(feature_as_of) FROM current_features').fetchone()[0]
            if previous and pd.to_datetime(previous,utc=True) > pd.to_datetime(cutoff,utc=True):
                raise ValueError('Cannot replace live features with an older cutoff.')
            # Snapshot replaces derived rows, not source telemetry. Nulls remain missing.
            computed_at = datetime.now(timezone.utc).isoformat()
            persisted = current.copy()
            for column in ('timestamp','feature_as_of'):
                persisted[column] = persisted[column].map(lambda value:value.isoformat())
            persisted['computed_at'] = computed_at
            columns = ['tyre_id','truck_id','timestamp']+FEATURE_COLUMNS+['feature_version','feature_as_of','is_stale','computed_at']
            records = [tuple(None if pd.isna(value) else value for value in row)
                       for row in persisted[columns].itertuples(index=False,name=None)]
            connection.execute('DELETE FROM current_features')
            connection.executemany(f'INSERT INTO current_features ({", ".join(columns)}) VALUES ({", ".join("?" for _ in columns)})',records)
            if connection.execute('SELECT COUNT(*) FROM current_features').fetchone()[0] != len(current):
                raise RuntimeError('Feature row count verification failed.')
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    output.mkdir(parents=True,exist_ok=True)
    temporary = output/'current_features.csv.tmp'
    current.to_csv(temporary,index=False)
    temporary.replace(output/'current_features.csv')
    report = {'feature_version':FEATURE_VERSION,'rows':len(current),'feature_columns':FEATURE_COLUMNS,
              'as_of':cutoff,'computed_at':computed_at,'stale_rows':int(current.is_stale.sum()),
              'missing_by_feature':{column:int(current[column].isna().sum()) for column in FEATURE_COLUMNS},
              'imputation':'none','source':'Operational telemetry, tyres, measured inspections and maintenance only.',
              'sqlite_table':'current_features','csv_path':str(output/'current_features.csv'),
              'csv_sha256':hashlib.sha256((output/'current_features.csv').read_bytes()).hexdigest(),
              'model_inference':'Not connected in this step.'}
    temporary = output/'current_feature_report.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    temporary.replace(output/'current_feature_report.json')
    return report


def run_feature_cycle(source=Path('data/tyremind.db'),history=Path('data/generated'),
                      simulator=Path('data/simulator'),seed=43,output=Path('data/monitoring/features')):
    from tyremind.monitoring.ingest import run_ingestion_cycle
    ingestion = run_ingestion_cycle(source,history,simulator,seed)
    features = refresh_current_features(source,output,as_of=ingestion['simulated_timestamp'])
    return {**ingestion,'mode':'current_features_ready','feature_rows':features['rows'],
            'feature_as_of':features['as_of'],'stale_feature_rows':features['stale_rows'],
            'feature_csv':features['csv_path'],
            'message':'Readings ingested and current features refreshed; ML inference is not connected yet.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('data/tyremind.db'))
    parser.add_argument('--output',type=Path,default=Path('data/monitoring/features'))
    args=parser.parse_args()
    print(json.dumps(refresh_current_features(args.database,args.output),indent=2))


if __name__=='__main__':
    main()
