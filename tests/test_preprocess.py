"""Exercise real CSV round trips and corrupted inputs."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from tyremind.data.generate import Config, write_dataset
from tyremind.data.preprocess import preprocess


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'raw'
        self.output = Path(self.temp.name) / 'processed'
        write_dataset(Config(trucks=2, days=60), self.source)

    def telemetry(self):
        return pd.read_csv(self.source / 'telemetry.csv', dtype='string')

    def save(self, frame):
        frame.to_csv(self.source / 'telemetry.csv', index=False)

    def test_clean_round_trip(self):
        report = preprocess(self.source, self.output)
        for stats in report['tables'].values():
            self.assertEqual(stats['rejected_rows'], 0)
        self.assertEqual(report['tables']['telemetry']['accepted_rows'], 720)
        self.assertFalse((self.output / 'clean/evaluation_labels.csv').exists())

    def test_corrupted_values_and_original_preserved(self):
        frame = self.telemetry()
        frame.loc[0, 'timestamp'] = 'bad-date'
        frame.loc[1, 'pressure_kpa'] = 'broken-sensor'
        frame.loc[2, 'load_ratio'] = None
        frame.loc[3, 'tyre_id'] = 'UNKNOWN'
        frame.loc[4, 'truck_id'] = 'TRK-002'
        frame.loc[5, 'daily_distance_km'] = '-3'
        frame.loc[6, 'pressure_kpa'] = 'inf'
        self.save(frame)
        before = (self.source / 'telemetry.csv').read_bytes()
        report = preprocess(self.source, self.output)
        self.assertEqual(report['tables']['telemetry']['rejected_rows'], 7)
        rejected = pd.read_csv(self.output / 'rejected/telemetry.csv')
        self.assertIn('broken-sensor', rejected.pressure_kpa.tolist())
        self.assertEqual(before, (self.source / 'telemetry.csv').read_bytes())

    def test_duplicates_conflicts_and_sorting(self):
        frame = self.telemetry()
        conflicting = frame.iloc[[1]].copy()
        conflicting['pressure_kpa'] = '700'
        frame = pd.concat([frame, frame.iloc[[0]], conflicting], ignore_index=True)
        self.save(frame.sample(frac=1, random_state=3))
        stats = preprocess(self.source, self.output)['tables']['telemetry']
        self.assertEqual(stats['rejected_rows'], 3)
        self.assertEqual(stats['accepted_rows'], 719)
        self.assertEqual(stats['reason_counts']['exact_duplicate'], 1)
        self.assertEqual(stats['reason_counts']['conflicting_key'], 2)
        clean = pd.read_csv(self.output / 'clean/telemetry.csv')
        self.assertTrue(clean.equals(clean.sort_values(['tyre_id', 'timestamp']).reset_index(drop=True)))

    def test_missing_schema_stops_without_outputs(self):
        self.save(self.telemetry().drop(columns='pressure_kpa'))
        with self.assertRaisesRegex(ValueError, 'missing required columns'):
            preprocess(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_abnormal_reading_retained_and_regression_rejected(self):
        frame = self.telemetry()
        frame.loc[0, 'pressure_kpa'] = '200'
        frame.loc[1, 'temperature_c'] = '110'
        frame.loc[2, 'cumulative_mileage_km'] = '1'
        self.save(frame)
        stats = preprocess(self.source, self.output)['tables']['telemetry']
        self.assertEqual(stats['rejected_rows'], 1)
        self.assertEqual(stats['reason_counts']['mileage_regression'], 1)
        clean = pd.read_csv(self.output / 'clean/telemetry.csv')
        self.assertIn(200, clean.pressure_kpa.tolist())
        self.assertIn(110, clean.temperature_c.tolist())

    def test_invalid_parent_and_before_installation(self):
        tyres = pd.read_csv(self.source / 'tyres.csv')
        tyres.loc[0, 'truck_id'] = 'UNKNOWN'
        tyres.to_csv(self.source / 'tyres.csv', index=False)
        frame = self.telemetry()
        frame.loc[60, 'timestamp'] = '2025-01-01'
        self.save(frame)
        report = preprocess(self.source, self.output)
        self.assertEqual(report['tables']['tyres']['rejected_rows'], 1)
        reasons = report['tables']['telemetry']['reason_counts']
        self.assertEqual(reasons['unknown_tyre'], 60)
        self.assertEqual(reasons['before_installation'], 1)


if __name__ == '__main__':
    unittest.main()
