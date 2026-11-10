"""Import validated operational CSVs into SQLite atomically and idempotently."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from tyremind.data.preprocess import SCHEMAS, validate_table

DDL = [
    '''CREATE TABLE IF NOT EXISTS trucks (
        truck_id TEXT PRIMARY KEY NOT NULL, vehicle_type TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS tyres (
        tyre_id TEXT PRIMARY KEY NOT NULL,
        truck_id TEXT NOT NULL REFERENCES trucks(truck_id),
        position TEXT NOT NULL, installation_timestamp TEXT NOT NULL,
        reference_pressure_kpa REAL NOT NULL CHECK(reference_pressure_kpa > 0),
        initial_tread_depth_mm REAL NOT NULL CHECK(initial_tread_depth_mm >= 0),
        UNIQUE(truck_id, position), UNIQUE(tyre_id, truck_id))''',
    '''CREATE TABLE IF NOT EXISTS telemetry (
        truck_id TEXT NOT NULL REFERENCES trucks(truck_id),
        tyre_id TEXT NOT NULL REFERENCES tyres(tyre_id), timestamp TEXT NOT NULL,
        pressure_kpa REAL NOT NULL CHECK(pressure_kpa >= 0), temperature_c REAL NOT NULL,
        daily_distance_km REAL NOT NULL CHECK(daily_distance_km >= 0),
        cumulative_mileage_km REAL NOT NULL CHECK(cumulative_mileage_km >= 0),
        load_ratio REAL NOT NULL CHECK(load_ratio >= 0),
        PRIMARY KEY(tyre_id, timestamp),
        FOREIGN KEY(tyre_id, truck_id) REFERENCES tyres(tyre_id, truck_id))''',
    '''CREATE TABLE IF NOT EXISTS inspections (
        inspection_id TEXT PRIMARY KEY NOT NULL,
        tyre_id TEXT NOT NULL REFERENCES tyres(tyre_id), timestamp TEXT NOT NULL,
        tread_depth_mm REAL NOT NULL CHECK(tread_depth_mm >= 0))''',
    '''CREATE TABLE IF NOT EXISTS maintenance_events (
        event_id TEXT PRIMARY KEY NOT NULL,
        tyre_id TEXT NOT NULL REFERENCES tyres(tyre_id), timestamp TEXT NOT NULL,
        event_type TEXT NOT NULL)''',
    'CREATE INDEX IF NOT EXISTS idx_telemetry_truck_time ON telemetry(truck_id, timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_inspections_tyre_time ON inspections(tyre_id, timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_maintenance_tyre_time ON maintenance_events(tyre_id, timestamp)',
]


def read_clean(source):
    tables = {}
    for name in SCHEMAS:
        raw = pd.read_csv(source / f'{name}.csv', dtype='string')
        clean, rejected, _ = validate_table(name, raw, tables)
        if len(rejected):
            raise ValueError(f'{name}: clean input contains {len(rejected)} invalid rows; rerun preprocessing.')
        tables[name] = clean
    for frame in tables.values():
        for column in ('timestamp', 'installation_timestamp'):
            if column in frame:
                frame[column] = frame[column].map(lambda value: value.isoformat())
    return tables


def load_database(source=Path('data/processed/clean'), database=Path('data/tyremind.db')):
    source, database = Path(source), Path(database)
    tables = read_clean(source)  # Fail before creating the database if input is invalid.
    database.parent.mkdir(parents=True, exist_ok=True)
    report = {'database': str(database), 'tables': {}}
    with closing(sqlite3.connect(database)) as connection:
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('BEGIN IMMEDIATE')
        try:
            for statement in DDL:
                connection.execute(statement)
            for name, frame in tables.items():
                columns, key, _ = SCHEMAS[name]
                key_indices = [columns.index(column) for column in key]
                existing = {tuple(row[i] for i in key_indices): row for row in
                            connection.execute(f'SELECT {", ".join(columns)} FROM {name}')}
                additions = []
                for row in frame[columns].itertuples(index=False, name=None):
                    identity = tuple(row[i] for i in key_indices)
                    if identity in existing:
                        if existing[identity] != row:
                            raise ValueError(f'{name}: existing record conflicts with input key {identity}. No changes saved.')
                    else:
                        additions.append(row)
                connection.executemany(
                    f'INSERT INTO {name} ({", ".join(columns)}) VALUES ({", ".join("?" for _ in columns)})',
                    additions)
                count = connection.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
                if count != len(existing) + len(additions):
                    raise RuntimeError(f'{name}: row count verification failed')
                report['tables'][name] = {'input_rows': len(frame), 'inserted_rows': len(additions),
                                          'unchanged_rows': len(frame) - len(additions), 'total_rows': count}
            if connection.execute('PRAGMA foreign_key_check').fetchall():
                raise RuntimeError('Database foreign key verification failed')
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/clean'))
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    args = parser.parse_args()
    print(json.dumps(load_database(args.input, args.database), indent=2))


if __name__ == '__main__':
    main()
