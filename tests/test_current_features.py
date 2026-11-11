"""Live formulas, point-in-time consistency, missing data and atomic snapshots."""
import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from tyremind.data.generate import Config,write_dataset
from tyremind.data.preprocess import preprocess
from tyremind.database.load import load_database
from tyremind.features.build import FEATURE_COLUMNS,export_features
from tyremind.monitoring.simulator import generate_next_batch
from tyremind.monitoring.ingest import ingest_batch
from tyremind.monitoring.features import refresh_current_features,run_feature_cycle


class CurrentFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.history,self.database=self.root/'history',self.root/'fleet.db'
        self.output,self.simulator=self.root/'live',self.root/'simulator'
        write_dataset(Config(trucks=2,days=30),self.history)
        preprocess(self.history,self.root/'processed')
        load_database(self.root/'processed/clean',self.database)

    def rows(self):
        with closing(sqlite3.connect(self.database)) as connection:
            return pd.read_sql_query('SELECT * FROM current_features ORDER BY tyre_id',connection)

    def test_current_values_use_full_history_and_correct_formulas(self):
        report=refresh_current_features(self.database,self.output)
        rows=self.rows()
        self.assertEqual(report['rows'],12)
        self.assertEqual(len(rows),12)
        self.assertEqual(len(report['feature_columns']),19)
        raw=pd.read_csv(self.history/'telemetry.csv')
        history=raw.loc[raw.tyre_id=='TYR-0001'].sort_values('timestamp')
        result=rows.loc[rows.tyre_id=='TYR-0001'].iloc[0]
        self.assertAlmostEqual(result.pressure_deviation_ratio,history.iloc[-1].pressure_kpa/800-1)
        self.assertAlmostEqual(result.pressure_change_kpa_per_day,history.iloc[-1].pressure_kpa-history.iloc[-2].pressure_kpa)
        self.assertAlmostEqual(result.pressure_mean_7d_kpa,history.tail(7).pressure_kpa.mean())
        self.assertAlmostEqual(result.distance_sum_7d_km,history.tail(7).daily_distance_km.sum())
        self.assertEqual(report['stale_rows'],0)

    def test_repeat_has_no_duplicates_or_source_training_changes(self):
        export_features(self.database,self.root/'training')
        original=hashlib.sha256((self.root/'training/tyre_features.csv').read_bytes()).hexdigest()
        first=refresh_current_features(self.database,self.output)
        before=self.rows()[FEATURE_COLUMNS]
        second=refresh_current_features(self.database,self.output)
        pd.testing.assert_frame_equal(before,self.rows()[FEATURE_COLUMNS])
        self.assertEqual(first['rows'],second['rows'])
        self.assertEqual(original,hashlib.sha256((self.root/'training/tyre_features.csv').read_bytes()).hexdigest())
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0],360)

    def test_future_inspections_maintenance_and_telemetry_excluded(self):
        refresh_current_features(self.database,self.output,as_of='2026-01-30')
        before=self.rows()[FEATURE_COLUMNS]
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("INSERT INTO inspections VALUES ('FUTURE','TYR-0001','2030-01-01T00:00:00+00:00',1)")
                connection.execute("INSERT INTO maintenance_events VALUES ('FUTURE','TYR-0001','2030-01-01T00:00:00+00:00','repair')")
                connection.execute("INSERT INTO telemetry VALUES ('TRK-001','TYR-0001','2030-01-01T00:00:00+00:00',1,200,100,999999,2)")
        refresh_current_features(self.database,self.output,as_of='2026-01-30')
        pd.testing.assert_frame_equal(before,self.rows()[FEATURE_COLUMNS])

    def test_rejected_tyre_is_marked_stale_not_fresh(self):
        generate_next_batch(self.database,self.history,self.simulator)
        with closing(sqlite3.connect(self.simulator/'simulator.db')) as connection:
            batch=json.loads(connection.execute('SELECT payload FROM batches').fetchone()[0])
        batch['telemetry'][0]['pressure_kpa']='invalid'
        ingest_batch(self.database,'test',batch)
        report=refresh_current_features(self.database,self.output,as_of=batch['timestamp'])
        self.assertEqual(report['stale_rows'],1)
        self.assertEqual(int(self.rows().iloc[0].is_stale),1)
        self.assertEqual(self.rows().iloc[0].timestamp,'2026-01-30T00:00:00+00:00')

    def test_missing_inspections_stay_null(self):
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("DELETE FROM inspections WHERE tyre_id='TYR-0001'")
        refresh_current_features(self.database,self.output)
        row=self.rows().iloc[0]
        self.assertTrue(pd.isna(row.latest_measured_tread_mm))
        self.assertTrue(pd.isna(row.historical_wear_mm_per_1000km))
        self.assertEqual(row.has_inspection,0)

    def test_failed_snapshot_rolls_back_and_older_cutoff_rejected(self):
        refresh_current_features(self.database,self.output)
        before=self.rows()
        with self.assertRaisesRegex(ValueError,'older cutoff'):
            refresh_current_features(self.database,self.output,as_of='2026-01-29')
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("CREATE TRIGGER feature_test_failure BEFORE INSERT ON current_features WHEN NEW.tyre_id='TYR-0002' BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError,'injected failure'):
            refresh_current_features(self.database,self.output)
        pd.testing.assert_frame_equal(before,self.rows())

    def test_full_live_cycle_updates_current_rows(self):
        result=run_feature_cycle(self.database,self.history,self.simulator,output=self.output)
        self.assertEqual(result['feature_rows'],12)
        self.assertEqual(result['stale_feature_rows'],0)
        self.assertEqual(result['telemetry_inserted'],12)
        self.assertTrue(self.rows().timestamp.eq('2026-01-31T00:00:00+00:00').all())
        self.assertEqual(len(pd.read_csv(self.output/'current_features.csv')),12)
        self.assertTrue((self.output/'current_feature_report.json').exists())


if __name__=='__main__':
    unittest.main()
