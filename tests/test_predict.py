"""Saved-model inference, skipped-data handling, persistence and retry tests."""
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Install .[ml] for inference tests')
class PredictionTests(unittest.TestCase):
    def setUp(self):
        import joblib
        from tyremind.data.generate import Config,write_dataset
        from tyremind.data.preprocess import preprocess
        from tyremind.database.load import load_database
        from tyremind.features.build import FEATURE_COLUMNS
        from tyremind.ml.train import make_model
        from tyremind.ml.anomaly import ANOMALY_FEATURES,make_pipeline
        from tyremind.monitoring.features import refresh_current_features
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.history,self.database=self.root/'history',self.root/'fleet.db'
        self.output=self.root/'predictions'
        write_dataset(Config(trucks=2,days=30),self.history)
        preprocess(self.history,self.root/'processed')
        load_database(self.root/'processed/clean',self.database)
        refresh_current_features(self.database,self.root/'features')
        with closing(sqlite3.connect(self.database)) as c:
            frame=pd.read_sql_query('SELECT * FROM current_features ORDER BY tyre_id',c)
        wear=make_model('random_forest',dict(n_estimators=3,max_depth=3))
        wear.fit(frame[FEATURE_COLUMNS],frame.latest_measured_tread_mm-.5)
        anomaly=make_pipeline(dict(n_estimators=10,max_samples=8,max_features=1.0))
        anomaly.fit(frame[ANOMALY_FEATURES])
        self.wear_path,self.anomaly_path=self.root/'wear.joblib',self.root/'anomaly.joblib'
        joblib.dump(dict(pipeline=wear,feature_columns=FEATURE_COLUMNS,model_family='random_forest',horizon_km=5000),self.wear_path)
        joblib.dump(dict(pipeline=anomaly,feature_columns=ANOMALY_FEATURES,model_family='isolation_forest',threshold=.5,
                         warmup_min_observations_7d=3),self.anomaly_path)
        from tyremind.monitoring.predict import SavedModels
        self.models=SavedModels(wear_artifact=self.wear_path,anomaly_artifact=self.anomaly_path)

    def predict(self):
        from tyremind.monitoring.predict import run_saved_models
        return run_saved_models(self.database,self.output,models=self.models)

    def test_real_predictions_match_loaded_pipelines_and_are_saved(self):
        from tyremind.features.build import FEATURE_COLUMNS
        from tyremind.ml.anomaly import predict_anomalies
        report=self.predict()
        self.assertEqual(report['scored_rows'],12)
        self.assertFalse(report['retrained'])
        with closing(sqlite3.connect(self.database)) as c:
            frame=pd.read_sql_query('SELECT * FROM current_features ORDER BY tyre_id',c)
            stored=pd.read_sql_query('SELECT * FROM predictions ORDER BY tyre_id',c)
        expected=self.models.wear['pipeline'].predict(frame[FEATURE_COLUMNS])
        np.testing.assert_allclose(stored.predicted_tread_depth_5000km_mm,expected)
        scores,flags=predict_anomalies(self.models.anomaly,frame)
        np.testing.assert_allclose(stored.anomaly_score,scores)
        np.testing.assert_array_equal(stored.anomaly_flag,flags.astype(int))
        self.assertEqual(set(json.loads(stored.input_features_json.iloc[0])),set(FEATURE_COLUMNS))
        self.assertTrue((self.output/'latest_predictions.csv').exists())

    def test_repeat_is_duplicate_free_and_does_not_retrain(self):
        from sklearn.ensemble import RandomForestRegressor,IsolationForest
        with patch.object(RandomForestRegressor,'fit',side_effect=AssertionError('No fit during inference')), \
             patch.object(IsolationForest,'fit',side_effect=AssertionError('No fit during inference')):
            first,second=self.predict(),self.predict()
        self.assertEqual(first['inserted_prediction_rows'],12)
        self.assertEqual(second['inserted_prediction_rows'],0)
        with closing(sqlite3.connect(self.database)) as c:
            self.assertEqual(c.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],12)

    def test_stale_and_warmup_are_skipped_with_null_outputs(self):
        with closing(sqlite3.connect(self.database)) as c:
            with c:
                c.execute("UPDATE current_features SET is_stale=1 WHERE tyre_id='TYR-0001'")
                c.execute("UPDATE current_features SET observations_7d=2 WHERE tyre_id='TYR-0002'")
        report=self.predict()
        self.assertEqual(report['scored_rows'],10)
        self.assertEqual(report['skipped_rows'],2)
        with closing(sqlite3.connect(self.database)) as c:
            rows=c.execute("SELECT status,anomaly_flag,predicted_tread_depth_5000km_mm FROM predictions WHERE status!='scored' ORDER BY tyre_id").fetchall()
        self.assertEqual(rows,[('stale_features',None,None),('insufficient_history',None,None)])

    def test_wrong_model_metadata_and_missing_models_fail(self):
        import joblib
        from tyremind.monitoring.predict import SavedModels
        with self.assertRaises(FileNotFoundError):
            SavedModels(wear_artifact=self.root/'missing.joblib',anomaly_artifact=self.anomaly_path)
        artifact=joblib.load(self.wear_path)
        artifact['feature_columns']=['wrong']
        joblib.dump(artifact,self.wear_path)
        with self.assertRaisesRegex(ValueError,'schema'):
            SavedModels(wear_artifact=self.wear_path,anomaly_artifact=self.anomaly_path)

    def test_old_feature_snapshot_is_rejected(self):
        with closing(sqlite3.connect(self.database)) as c:
            with c:
                c.execute("UPDATE current_features SET feature_as_of='2026-01-29T00:00:00+00:00'")
        with self.assertRaisesRegex(ValueError,'older than stored telemetry'):
            self.predict()

    def test_write_failure_rolls_back_partial_predictions(self):
        self.predict()
        with closing(sqlite3.connect(self.database)) as c:
            with c:
                c.execute("UPDATE current_features SET pressure_deviation_ratio=-.3 WHERE tyre_id='TYR-0001'")
                c.execute("CREATE TRIGGER fail_prediction BEFORE INSERT ON predictions WHEN NEW.tyre_id='TYR-0002' BEGIN SELECT RAISE(ABORT,'injected failure'); END")
        with self.assertRaisesRegex(sqlite3.IntegrityError,'injected failure'):
            self.predict()
        with closing(sqlite3.connect(self.database)) as c:
            self.assertEqual(c.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],12)

    def test_worker_loads_and_runs_saved_models(self):
        result=subprocess.run([sys.executable,'-m','tyremind.monitoring.worker','--run-once',
                               '--source-database',str(self.database),'--history',str(self.history),
                               '--simulator-output',str(self.root/'simulator'),'--output',str(self.root/'worker'),
                               '--feature-output',str(self.root/'features'),'--prediction-output',str(self.output),
                               '--wear-artifact',str(self.wear_path),'--anomaly-artifact',str(self.anomaly_path)],
                              capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        status=json.loads((self.root/'worker/worker_status.json').read_text())
        self.assertEqual(status['last_result']['mode'],'ml_predictions_ready')
        self.assertEqual(status['last_result']['prediction_rows'],12)
        self.assertEqual(status['last_result']['wear_model'],'random_forest')


if __name__=='__main__':
    unittest.main()
