"""Anomaly labeling, calibration, isolation and saved inference checks."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Install .[ml] for anomaly tests')
class AnomalyTests(unittest.TestCase):
    def fixture(self):
        from tyremind.ml.anomaly import ANOMALY_FEATURES
        dates = pd.date_range('2026-01-01', periods=40, tz='UTC')
        features = pd.DataFrame({column: np.ones(40) for column in ANOMALY_FEATURES})
        features['truck_id'], features['tyre_id'], features['timestamp'] = 'T1', 'Y1', dates
        features['observations_7d'] = 7
        labels = pd.DataFrame(dict(tyre_id=['Y1']*40, timestamp=dates, scenario_label=['normal']*40,
                                   scenario_active=[False]*40, overload_active=[False]*40))
        return features, labels

    def test_confusion_metrics(self):
        from tyremind.ml.anomaly import classification_metrics
        result = classification_metrics([False, False, True, True], [.1, .9, .8, .2], .5)
        self.assertEqual(result['confusion_matrix'], dict(tn=1, fp=1, fn=1, tp=1))
        for name in ('precision', 'recall', 'f1', 'accuracy', 'false_positive_rate'):
            self.assertEqual(result[name], .5)

    def test_threshold_uses_validation_and_respects_fpr(self):
        from tyremind.ml.anomaly import select_threshold
        validation = {'seen': ([0,0,0,1,1], np.array([.1,.2,.3,.4,.9])),
                      'unseen': ([0,0,1,1], np.array([.1,.2,.6,.8]))}
        result = select_threshold(validation, fpr_limit=0)
        self.assertEqual(result['balanced_recall'], 1)
        self.assertTrue(all(row['false_positive_rate'] == 0 for row in result['validation'].values()))
        with self.assertRaises(ValueError):
            select_threshold(validation, -1)

    def test_active_overload_excluded_even_when_assigned_normal(self):
        from tyremind.ml.anomaly import prepare_data
        features, labels = self.fixture()
        labels.loc[5, 'overload_active'] = True
        rows = prepare_data(features, labels)
        self.assertTrue(rows.loc[5, 'is_fault'])
        self.assertFalse(rows.loc[5:32, 'normal_reference'].any())
        self.assertTrue(rows.loc[33, 'normal_reference'])

    def test_future_labels_do_not_change_prior_reference(self):
        from tyremind.ml.anomaly import prepare_data
        features, labels = self.fixture()
        before = prepare_data(features, labels).iloc[:20]
        labels.loc[20:, 'scenario_active'] = True
        after = prepare_data(features, labels).iloc[:20]
        pd.testing.assert_frame_equal(before, after)

    def test_invalid_boolean_rejected(self):
        from tyremind.ml.anomaly import prepare_data
        features, labels = self.fixture()
        labels['scenario_active'] = labels.scenario_active.astype(str)
        labels.loc[3, 'scenario_active'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'boolean'):
            prepare_data(features, labels)

    def test_operational_inference_requires_no_labels(self):
        from tyremind.ml.anomaly import ANOMALY_FEATURES, make_pipeline, predict_anomalies
        rng = np.random.default_rng(42)
        values = pd.DataFrame(rng.normal(size=(120, len(ANOMALY_FEATURES))), columns=ANOMALY_FEATURES)
        pipeline = make_pipeline(dict(n_estimators=10, max_samples=64, max_features=1.0))
        pipeline.fit(values)
        artifact = {'pipeline': pipeline, 'feature_columns': ANOMALY_FEATURES, 'threshold': .5}
        scores, flags = predict_anomalies(artifact, values)
        self.assertTrue(np.isfinite(scores).all())
        np.testing.assert_array_equal(flags, scores >= .5)

    def test_end_to_end_no_test_truck_training_and_saved_model(self):
        import joblib
        from tyremind.data.generate import Config, write_dataset
        from tyremind.data.preprocess import preprocess
        from tyremind.database.load import load_database
        from tyremind.features.build import export_features
        from tyremind.ml.split import export_splits
        from tyremind.ml.anomaly import train_anomaly, predict_anomalies
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(Config(trucks=4), root / 'raw')
            preprocess(root / 'raw', root / 'processed')
            load_database(root / 'processed/clean', root / 'tyremind.db')
            export_features(root / 'tyremind.db', root / 'features')
            features, labels = root / 'features/tyre_features.csv', root / 'raw/evaluation_labels.csv'
            export_splits(features, labels, root / 'splits')
            with patch('tyremind.ml.anomaly.CANDIDATES', [dict(n_estimators=10, max_samples=64, max_features=1.0)]):
                report = train_anomaly(features, labels, root / 'splits/split_report.json', root / 'models')
            self.assertEqual(report['training_fault_rows'], 0)
            self.assertTrue(all(row['overall']['false_positive_rate'] <= .05
                                for name, row in report['evaluations'].items() if name.startswith('validation')))
            self.assertFalse(set(report['truck_groups']['train']) & set(report['truck_groups']['test']))
            artifact = joblib.load(root / 'models/isolation_forest.joblib')
            self.assertNotIn('scenario_label', artifact['feature_columns'])
            scored = pd.read_csv(root / 'models/anomaly_predictions.csv')
            self.assertTrue((scored['evaluation'].str.startswith(('validation', 'test'))).all())
            self.assertTrue((root / 'models/metrics_summary.csv').exists())
            self.assertEqual(len(predict_anomalies(artifact, pd.read_csv(features).iloc[:5])[0]), 5)


if __name__ == '__main__':
    unittest.main()
