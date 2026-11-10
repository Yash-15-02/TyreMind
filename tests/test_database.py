"""Database persistence, repeat import, constraints and rollback tests."""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from tyremind.data.generate import Config, write_dataset
from tyremind.data.preprocess import preprocess
from tyremind.database.load import load_database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.raw, self.processed = root / 'raw', root / 'processed'
        self.source, self.database = self.processed / 'clean', root / 'tyremind.db'
        write_dataset(Config(trucks=2, days=60), self.raw)
        preprocess(self.raw, self.processed)

    def test_persistence_and_repeat_import(self):
        first = load_database(self.source, self.database)
        second = load_database(self.source, self.database)
        self.assertEqual(first['tables']['telemetry']['inserted_rows'], 720)
        for stats in second['tables'].values():
            self.assertEqual(stats['inserted_rows'], 0)
            self.assertEqual(stats['unchanged_rows'], stats['input_rows'])
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0], 720)
            self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])
            self.assertEqual(connection.execute('PRAGMA integrity_check').fetchone()[0], 'ok')

    def test_database_enforces_constraints(self):
        load_database(self.source, self.database)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute('PRAGMA foreign_keys = ON')
            for statement in (
                "UPDATE telemetry SET pressure_kpa = -1",
                "UPDATE telemetry SET truck_id = 'TRK-002' WHERE truck_id = 'TRK-001'",
                "INSERT INTO trucks SELECT * FROM trucks LIMIT 1",
            ):
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(statement)
                connection.rollback()

    def test_conflict_rolls_back_earlier_inserts(self):
        load_database(self.source, self.database)
        trucks = pd.read_csv(self.source / 'trucks.csv')
        trucks.loc[len(trucks)] = ['TRK-NEW', 'demo']
        trucks.to_csv(self.source / 'trucks.csv', index=False)
        telemetry = pd.read_csv(self.source / 'telemetry.csv')
        telemetry.loc[0, 'pressure_kpa'] += 1
        telemetry.to_csv(self.source / 'telemetry.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            load_database(self.source, self.database)
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM trucks').fetchone()[0], 2)

    def test_invalid_input_does_not_create_database(self):
        telemetry = pd.read_csv(self.source / 'telemetry.csv')
        telemetry.loc[0, 'tyre_id'] = 'UNKNOWN'
        telemetry.to_csv(self.source / 'telemetry.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'invalid rows'):
            load_database(self.source, self.database)
        self.assertFalse(self.database.exists())


if __name__ == '__main__':
    unittest.main()
