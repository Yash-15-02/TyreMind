"""Explicit continuation branch and persisted decisions without events."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from tyremind.monitoring.risk import evaluate_risks,monitoring_decision


class MonitoringDecisionTests(unittest.TestCase):
    def test_normal_and_medium_continue(self):
        for level in ('Normal','Medium'):
            result=monitoring_decision({'risk_level':level,'data_status':'complete'})
            self.assertEqual(result['next_action'],'CONTINUE_MONITORING')
            self.assertEqual(result['monitoring_continues'],1)
            self.assertEqual(result['data_review_required'],0)

    def test_high_critical_not_sent_down_normal_branch(self):
        for level in ('High','Critical'):
            result=monitoring_decision({'risk_level':level,'data_status':'complete'})
            self.assertEqual(result['next_action'],'AWAIT_EVENT_HANDLER')
            self.assertEqual(result['monitoring_continues'],1)

    def test_unknown_or_partial_evidence_keeps_review_flag(self):
        result=monitoring_decision({'risk_level':None,'data_status':'stale'})
        self.assertEqual(result['next_action'],'DATA_REVIEW_REQUIRED')
        self.assertEqual(result['data_review_required'],1)
        partial=monitoring_decision({'risk_level':'Medium','data_status':'partial'})
        self.assertEqual(partial['next_action'],'CONTINUE_MONITORING')
        self.assertEqual(partial['data_review_required'],1)

    def test_decisions_saved_once_and_no_investigation_tables_created(self):
        from tyremind.data.generate import Config,write_dataset
        from tyremind.data.preprocess import preprocess
        from tyremind.database.load import load_database
        from tyremind.monitoring.features import refresh_current_features
        from tyremind.monitoring.risk import canonical_inputs
        import pandas as pd
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            database=root/'fleet.db'
            write_dataset(Config(trucks=1,days=30),root/'raw')
            preprocess(root/'raw',root/'processed')
            load_database(root/'processed/clean',database)
            # Keep this fixture entirely normal to isolate the continuation branch.
            with closing(sqlite3.connect(database)) as connection:
                with connection:
                    connection.execute('UPDATE telemetry SET pressure_kpa=800,temperature_c=45,load_ratio=.85')
            refresh_current_features(database,root/'features')
            with closing(sqlite3.connect(database)) as connection:
                frame=pd.read_sql_query('SELECT * FROM current_features',connection)
                with connection:
                    connection.execute('''CREATE TABLE predictions (
                        prediction_id TEXT,wear_model TEXT,predicted_at TEXT,tyre_id TEXT,timestamp TEXT,
                        feature_as_of TEXT,feature_version TEXT,input_features_json TEXT,status TEXT,
                        predicted_tread_depth_5000km_mm REAL,anomaly_score REAL,anomaly_flag INTEGER,anomaly_threshold REAL)''')
                    for _,row in frame.iterrows():
                        connection.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (row.tyre_id,'random_forest','2026-01-31',row.tyre_id,row.timestamp,row.feature_as_of,
                             row.feature_version,json.dumps(canonical_inputs(row)),'scored',10,.4,0,.58))
            first=evaluate_risks(database,output=root/'risk')
            repeat=evaluate_risks(database,output=root/'risk')
            self.assertEqual(first['continue_monitoring_rows'],6)
            self.assertEqual(first['inserted_decisions'],6)
            self.assertEqual(repeat['inserted_decisions'],0)
            self.assertFalse(first['events_created'])
            with closing(sqlite3.connect(database)) as connection:
                self.assertEqual(connection.execute('SELECT COUNT(*) FROM monitoring_decisions').fetchone()[0],6)
                actions={row[0] for row in connection.execute('SELECT next_action FROM monitoring_decisions')}
                self.assertEqual(actions,{'CONTINUE_MONITORING'})
                self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(),[])
                tables={row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertFalse({'risk_events','investigations','work_orders'} & tables)


if __name__=='__main__':
    unittest.main()
