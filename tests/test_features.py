"""Point-in-time correctness and feature calculations."""
import unittest

import pandas as pd

from tyremind.features.build import build_features


class FeatureTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2026-01-01', periods=10, tz='UTC')
        self.telemetry = pd.DataFrame(dict(
            truck_id=['T1'] * 10, tyre_id=['Y1'] * 10, timestamp=dates,
            pressure_kpa=[800 - 10*i for i in range(10)], temperature_c=[40]*10,
            daily_distance_km=[100]*10, cumulative_mileage_km=[100*(i+1) for i in range(10)],
            load_ratio=[1.2]*10))
        self.tyres = pd.DataFrame([dict(tyre_id='Y1', installation_timestamp=dates[0],
                                        reference_pressure_kpa=800)])
        self.inspections = pd.DataFrame(dict(inspection_id=['I1', 'I2'], tyre_id=['Y1', 'Y1'],
                                              timestamp=[dates[0], dates[4]], tread_depth_mm=[15, 14.6]))
        self.events = pd.DataFrame(dict(tyre_id=['Y1'], timestamp=[dates[6]]))

    def build(self):
        return build_features(self.telemetry, self.tyres, self.inspections, self.events)

    def test_known_values_and_warmup(self):
        rows = self.build()
        self.assertAlmostEqual(rows.iloc[2].pressure_mean_7d_kpa, 790)
        self.assertAlmostEqual(rows.iloc[2].pressure_change_kpa_per_day, -10)
        self.assertAlmostEqual(rows.iloc[2].pressure_deviation_ratio, -0.025)
        self.assertAlmostEqual(rows.iloc[4].historical_wear_mm_per_1000km, 1)
        self.assertEqual(rows.iloc[7].distance_sum_7d_km, 700)
        self.assertEqual(rows.iloc[7].overload_fraction_7d, 1)
        self.assertEqual(rows.iloc[7].days_since_maintenance, 1)
        self.assertEqual(rows.iloc[3].latest_measured_tread_mm, 15)
        self.assertTrue(pd.isna(rows.iloc[0].pressure_mean_7d_kpa))
        self.assertTrue(pd.isna(rows.iloc[0].historical_wear_mm_per_1000km))

    def test_future_changes_cannot_change_past(self):
        before = self.build().iloc[:4].copy()
        self.telemetry.loc[4:, 'pressure_kpa'] = 100
        self.inspections.loc[1, 'tread_depth_mm'] = 3
        self.events.loc[0, 'timestamp'] = pd.Timestamp('2026-01-06', tz='UTC')
        pd.testing.assert_frame_equal(before, self.build().iloc[:4])

    def test_prefix_matches_full_history(self):
        full = self.build().iloc[:4].copy()
        self.telemetry = self.telemetry.iloc[:4]
        self.inspections = self.inspections.iloc[:1]
        self.events = self.events.iloc[:0]
        pd.testing.assert_frame_equal(full, self.build(), check_dtype=False)

    def test_no_cross_tyre_history(self):
        first = self.build()
        other = self.telemetry.copy()
        other['tyre_id'] = 'Y2'
        other['pressure_kpa'] = 500
        metadata = self.tyres.copy()
        metadata['tyre_id'] = 'Y2'
        combined = build_features(pd.concat([self.telemetry, other]), pd.concat([self.tyres, metadata]),
                                  self.inspections, self.events)
        pd.testing.assert_frame_equal(first, combined[combined.tyre_id == 'Y1'].reset_index(drop=True))
        second = combined[combined.tyre_id == 'Y2']
        self.assertTrue(second.latest_measured_tread_mm.isna().all())
        self.assertTrue(second.days_since_maintenance.isna().all())

    def test_time_window_with_gap(self):
        self.telemetry.loc[9, 'timestamp'] = pd.Timestamp('2026-02-01', tz='UTC')
        last = self.build().iloc[-1]
        self.assertEqual(last.observations_7d, 1)
        self.assertTrue(pd.isna(last.pressure_mean_7d_kpa))

    def test_sqlite_export_leaves_database_unchanged(self):
        import hashlib
        import tempfile
        from pathlib import Path
        from tyremind.data.generate import Config, write_dataset
        from tyremind.data.preprocess import preprocess
        from tyremind.database.load import load_database
        from tyremind.features.build import export_features
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(Config(trucks=1, days=30), root / 'raw')
            preprocess(root / 'raw', root / 'processed')
            database = root / 'tyremind.db'
            load_database(root / 'processed/clean', database)
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            report = export_features(database, root / 'features')
            self.assertEqual(report['rows'], 180)
            self.assertEqual(report['tyres'], 6)
            self.assertEqual(len(report['feature_columns']), 19)
            self.assertTrue((root / 'features/tyre_features.csv').exists())
            self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
