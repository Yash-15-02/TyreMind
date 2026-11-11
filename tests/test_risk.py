"""Deterministic rule priorities, evidence freshness and risk persistence."""
import copy
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from tyremind.monitoring.risk import assess_risk,load_policy,evaluate_risks


class RiskRuleTests(unittest.TestCase):
    def setUp(self):
        self.policy=load_policy()
        self.evidence=dict(pressure_kpa=800,reference_pressure_kpa=800,temperature_c=45,
            latest_measured_tread_mm=9,days_since_inspection=3,load_mean_7d=.85,is_stale=False,
            pressure_history=[{'timestamp':f'2026-01-0{day}T00:00:00+00:00','pressure_kpa':800} for day in range(1,5)],
            prediction=dict(status='scored',predicted_tread_depth_5000km_mm=8,anomaly_flag=0,anomaly_score=.4,anomaly_threshold=.58))

    def test_complete_normal(self):
        result=assess_risk(self.evidence,self.policy)
        self.assertEqual(result['risk_level'],'Normal')
        self.assertEqual(result['data_status'],'complete')

    def test_pressure_boundaries(self):
        for pressure,expected in [(721,'Normal'),(720,'Medium'),(640,'High'),(560,'Critical')]:
            with self.subTest(pressure=pressure):
                evidence=copy.deepcopy(self.evidence)
                evidence['pressure_kpa']=pressure
                self.assertEqual(assess_risk(evidence,self.policy)['risk_level'],expected)

    def test_highest_rule_wins(self):
        self.evidence.update(pressure_kpa=630,temperature_c=80)
        result=assess_risk(self.evidence,self.policy)
        self.assertEqual(result['risk_level'],'Critical')
        self.assertEqual(len(result['rules']),2)

    def test_negative_if_does_not_cancel_pressure_risk(self):
        self.evidence['pressure_kpa']=620
        self.assertEqual(assess_risk(self.evidence,self.policy)['risk_level'],'High')

    def test_anomaly_alone_medium_and_combination_high(self):
        self.evidence['prediction']['anomaly_flag']=1
        self.assertEqual(assess_risk(self.evidence,self.policy)['risk_level'],'Medium')
        self.evidence['pressure_kpa']=720
        self.assertEqual(assess_risk(self.evidence,self.policy)['risk_level'],'High')

    def test_three_declines_need_meaningful_drop(self):
        for row,pressure in zip(self.evidence['pressure_history'],[800,794,788,782]):
            row['pressure_kpa']=pressure
        self.evidence['pressure_kpa']=782
        result=assess_risk(self.evidence,self.policy)
        self.assertEqual(result['risk_level'],'High')
        self.assertEqual(result['rules'][0]['rule_id'],'sustained_pressure_decline')

    def test_stale_data_not_classified(self):
        self.evidence.update(is_stale=True,pressure_kpa=500)
        result=assess_risk(self.evidence,self.policy)
        self.assertIsNone(result['risk_level'])
        self.assertEqual(result['data_status'],'stale')
        self.assertEqual(result['rules'],[])

    def test_old_tread_is_not_a_current_critical_measurement(self):
        self.evidence.update(latest_measured_tread_mm=0,days_since_inspection=30)
        result=assess_risk(self.evidence,self.policy)
        self.assertIsNone(result['risk_level'])
        self.assertEqual(result['data_status'],'insufficient_data')

    def test_missing_ml_keeps_independent_critical_rule(self):
        self.evidence.update(prediction=None,pressure_kpa=500)
        result=assess_risk(self.evidence,self.policy)
        self.assertEqual(result['risk_level'],'Critical')
        self.assertEqual(result['data_status'],'partial')

    def test_missing_pressure_cannot_be_normal(self):
        self.evidence['pressure_kpa']=None
        result=assess_risk(self.evidence,self.policy)
        self.assertIsNone(result['risk_level'])

    def test_predicted_and_observed_tread_are_distinct(self):
        self.evidence['prediction']['predicted_tread_depth_5000km_mm']=1
        self.assertEqual(assess_risk(self.evidence,self.policy)['risk_level'],'High')
        self.evidence['latest_measured_tread_mm']=1
        self.assertEqual(assess_risk(self.evidence,self.policy)['risk_level'],'Critical')

    def test_invalid_policy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=copy.deepcopy(self.policy)
            policy['pressure_high_deviation']=.4
            path=Path(directory)/'bad.json'
            path.write_text(json.dumps(policy))
            with self.assertRaisesRegex(ValueError,'Pressure deviations'):
                load_policy(path)


class RiskStorageTests(unittest.TestCase):
    def setUp(self):
        from tyremind.data.generate import Config,write_dataset
        from tyremind.data.preprocess import preprocess
        from tyremind.database.load import load_database
        from tyremind.monitoring.features import refresh_current_features
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.database,self.output=self.root/'fleet.db',self.root/'risk'
        write_dataset(Config(trucks=2,days=30),self.root/'raw')
        preprocess(self.root/'raw',self.root/'processed')
        load_database(self.root/'processed/clean',self.database)
        refresh_current_features(self.database,self.root/'features')

    def test_storage_replay_and_independent_critical_rule(self):
        first=evaluate_risks(self.database,output=self.output)
        second=evaluate_risks(self.database,output=self.output)
        self.assertEqual(first['rows'],12)
        self.assertEqual(first['inserted_assessments'],12)
        self.assertEqual(second['inserted_assessments'],0)
        self.assertFalse(first['events_created'])
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("UPDATE telemetry SET pressure_kpa=500 WHERE tyre_id='TYR-0001' AND timestamp='2026-01-30T00:00:00+00:00'")
        from tyremind.monitoring.features import refresh_current_features
        refresh_current_features(self.database,self.root/'features')
        changed=evaluate_risks(self.database,output=self.output)
        self.assertGreaterEqual(changed['risk_counts']['Critical'],1)
        with closing(sqlite3.connect(self.database)) as connection:
            row=connection.execute("SELECT evidence_json FROM risk_assessments WHERE risk_level='Critical' LIMIT 1").fetchone()
        self.assertEqual(json.loads(row[0])['pressure_kpa'],500)
        self.assertTrue((self.output/'risk_report.json').exists())

    def test_skipped_prediction_nulls_remain_unclassified(self):
        from tyremind.monitoring.risk import canonical_inputs
        import pandas as pd
        with closing(sqlite3.connect(self.database)) as connection:
            feature=pd.read_sql_query("SELECT * FROM current_features WHERE tyre_id='TYR-0001'",connection).iloc[0]
            with connection:
                connection.execute("CREATE TABLE predictions (prediction_id TEXT,wear_model TEXT,predicted_at TEXT,tyre_id TEXT,timestamp TEXT,feature_as_of TEXT,feature_version TEXT,input_features_json TEXT,status TEXT,predicted_tread_depth_5000km_mm REAL,anomaly_score REAL,anomaly_flag INTEGER,anomaly_threshold REAL)")
                connection.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    ('skipped','random_forest','2026-01-31','TYR-0001',feature.timestamp,feature.feature_as_of,
                     feature.feature_version,json.dumps(canonical_inputs(feature)),'insufficient_history',None,None,None,.58))
        result=evaluate_risks(self.database,output=self.output)
        self.assertGreaterEqual(result['unclassified_rows'],1)
        with closing(sqlite3.connect(self.database)) as connection:
            payload=connection.execute("SELECT evidence_json FROM risk_assessments WHERE tyre_id='TYR-0001'").fetchone()[0]
        self.assertIsNone(json.loads(payload)['prediction']['anomaly_flag'])

    def test_partial_write_failure_rolls_back_snapshot(self):
        evaluate_risks(self.database,output=self.output)
        with closing(sqlite3.connect(self.database)) as connection:
            with connection:
                connection.execute("UPDATE telemetry SET pressure_kpa=500 WHERE tyre_id='TYR-0001' AND timestamp='2026-01-30T00:00:00+00:00'")
                connection.execute("CREATE TRIGGER fail_risk BEFORE INSERT ON risk_assessments WHEN NEW.tyre_id='TYR-0002' BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        from tyremind.monitoring.features import refresh_current_features
        refresh_current_features(self.database,self.root/'features')
        with self.assertRaisesRegex(sqlite3.IntegrityError,'injected failure'):
            evaluate_risks(self.database,output=self.output)
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM risk_assessments').fetchone()[0],12)


if __name__=='__main__':
    unittest.main()
