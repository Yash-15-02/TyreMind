"""Historical continuity, durable batches, restart RNG and worker integration."""
import hashlib
from contextlib import closing
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from tyremind.data.generate import Config, write_dataset
from tyremind.data.preprocess import preprocess
from tyremind.database.load import load_database
from tyremind.monitoring.simulator import generate_next_batch


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history, self.database = self.root / 'historical', self.root / 'fleet.db'
        self.output = self.root / 'simulator'
        write_dataset(Config(trucks=2, days=30), self.history)
        preprocess(self.history, self.root / 'processed')
        load_database(self.root / 'processed/clean', self.database)

    def batch(self, output=None, seed=43):
        return generate_next_batch(self.database, self.history, output or self.output, seed)

    def read(self, result, name='telemetry'):
        return pd.read_csv(Path(result['raw_batch_directory']) / f'{name}.csv')

    def test_continues_historical_identity_time_and_mileage(self):
        checksum = hashlib.sha256(self.database.read_bytes()).hexdigest()
        result = self.batch()
        future = self.read(result).set_index('tyre_id')
        history = pd.read_csv(self.history / 'telemetry.csv').groupby('tyre_id').tail(1).set_index('tyre_id')
        self.assertEqual(set(future.index), set(history.index))
        self.assertEqual(result['readings_generated'], 12)
        self.assertEqual(result['simulated_timestamp'], '2026-01-31T00:00:00+00:00')
        self.assertTrue((future.cumulative_mileage_km > history.cumulative_mileage_km).all())
        self.assertEqual(checksum, hashlib.sha256(self.database.read_bytes()).hexdigest())

    def test_restart_is_reproducible_and_does_not_reset(self):
        first, second = self.batch(), self.batch()
        other = self.root / 'second-simulator'
        repeated_first, repeated_second = self.batch(other), self.batch(other)
        pd.testing.assert_frame_equal(self.read(first), self.read(repeated_first))
        pd.testing.assert_frame_equal(self.read(second), self.read(repeated_second))
        self.assertEqual(second['batch_id'], 2)
        self.assertEqual(second['simulated_timestamp'], '2026-02-01T00:00:00+00:00')
        self.assertTrue((self.read(second).cumulative_mileage_km > self.read(first).cumulative_mileage_km).all())

    def test_shared_truck_values_and_tread_decline(self):
        first, second = self.batch(), self.batch()
        rows = self.read(second)
        for column in ('daily_distance_km', 'load_ratio'):
            self.assertTrue(rows.groupby('truck_id')[column].nunique().eq(1).all())
        old, new = self.read(first,'simulation_truth'), self.read(second,'simulation_truth')
        self.assertTrue((new.simulated_true_tread_depth_mm <= old.simulated_true_tread_depth_mm).all())
        self.assertNotIn('scenario_label', rows.columns)
        self.assertNotIn('simulated_true_tread_depth_mm', rows.columns)

    def test_export_failure_retries_same_durable_batch(self):
        with patch('tyremind.monitoring.simulator.export_batch', side_effect=OSError('Injected disk error')):
            with self.assertRaises(OSError):
                self.batch()
        recovered = self.batch()
        self.assertEqual(recovered['batch_id'], 1)
        self.assertEqual(self.batch()['batch_id'], 2)
        with closing(sqlite3.connect(self.output / 'simulator.db')) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM batches').fetchone()[0], 2)

    def test_seed_mismatch_rejected(self):
        self.batch()
        with self.assertRaisesRegex(ValueError, 'different source paths or seed'):
            self.batch(seed=99)
        self.assertEqual(self.batch()['batch_id'], 2)

    def test_incompatible_history_rejected(self):
        labels = pd.read_csv(self.history / 'evaluation_labels.csv')
        labels.loc[0, 'simulated_true_tread_depth_mm'] = 999
        labels.to_csv(self.history / 'evaluation_labels.csv',index=False)
        with self.assertRaisesRegex(ValueError, 'labels do not match'):
            self.batch()

    def test_real_worker_generates_and_stops(self):
        result = subprocess.run([sys.executable,'-m','tyremind.monitoring.worker','--run-once',
                                 '--source-database',str(self.database),'--history',str(self.history),
                                 '--simulator-output',str(self.output),'--output',str(self.root/'worker'),
                                 '--feature-output',str(self.root/'features')],
                                capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        status = json.loads((self.root/'worker/worker_status.json').read_text())
        self.assertEqual(status['state'],'stopped')
        self.assertEqual(status['last_result']['readings_generated'],12)
        self.assertTrue(status['last_result']['operational_database_updated'])
        self.assertEqual(status['last_result']['telemetry_inserted'],12)


if __name__ == '__main__':
    unittest.main()
