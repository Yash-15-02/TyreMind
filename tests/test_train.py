"""Regression metrics, train-only preprocessing and temporal/group isolation."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import sklearn
except ImportError:
    sklearn = None


@unittest.skipUnless(sklearn, 'Install .[ml] for training tests')
class TrainingTests(unittest.TestCase):
    def test_regression_metrics(self):
        from tyremind.ml.train import metrics
        result = metrics([1, 2, 3], [1.1, 1.8, 3.3])
        self.assertAlmostEqual(result['mae_mm'], 0.2)
        self.assertAlmostEqual(result['rmse_mm'], np.sqrt(0.14 / 3))
        self.assertAlmostEqual(result['within_0_25mm_fraction'], 2 / 3)
        self.assertNotIn('precision', result)
        with self.assertRaises(ValueError):
            metrics([1], [np.nan])

    def test_imputer_fits_training_only(self):
        from tyremind.ml.train import make_model
        model = make_model('random_forest', {'n_estimators': 3, 'min_samples_leaf': 1})
        training = pd.DataFrame({'a': [1, np.nan, 3], 'b': [np.nan]*3})
        model.fit(training, [1, 2, 3])
        before = model.named_steps['imputer'].statistics_.copy()
        self.assertEqual(before[0], 2)
        self.assertTrue(np.isfinite(model.predict(pd.DataFrame({'a': [999], 'b': [100]}))).all())
        np.testing.assert_array_equal(before, model.named_steps['imputer'].statistics_)

    def test_baseline_accounts_for_inspection_distance(self):
        from tyremind.ml.train import baseline_predict
        rows = pd.DataFrame({'latest_measured_tread_mm': [10],
                             'historical_wear_mm_per_1000km': [0.1],
                             'distance_since_inspection_km': [1000]})
        self.assertAlmostEqual(baseline_predict(rows, {'wear_rate': 0, 'tread_mm': 0})[0], 9.4)

    def fixture(self):
        from tyremind.features.build import FEATURE_COLUMNS
        from tyremind.ml.split import TARGET
        rows = []
        for truck in range(6):
            for day, stamp in enumerate(pd.date_range('2026-01-01', periods=180, tz='UTC')):
                row = {column: 1.0 for column in FEATURE_COLUMNS}
                row.update(truck_id=f'T{truck}', tyre_id=f'Y{truck}', timestamp=stamp,
                           target_available_at=stamp + pd.Timedelta(days=10))
                row[TARGET] = 10.0
                rows.append(row)
        dataset = pd.DataFrame(rows)
        from tyremind.ml.split import split_dataset
        _, _, report = split_dataset(dataset)
        return dataset, report

    def test_folds_have_no_future_or_test_trucks(self):
        from tyremind.ml.train import make_folds
        dataset, report = self.fixture()
        # Fixture has 4 training, 1 validation truck; enlarge validation rows through multiple tyres.
        duplicate = dataset.copy()
        duplicate['tyre_id'] += '-extra'
        dataset = pd.concat([dataset, duplicate], ignore_index=True)
        folds = make_folds(dataset, report)
        val_keys = []
        for fold in folds:
            self.assertTrue((fold['train'].target_available_at < fold['start']).all())
            self.assertTrue((fold['train'].timestamp < fold['start']).all())
            for group in ('seen', 'unseen'):
                self.assertTrue((fold[group].target_available_at < fold['end']).all())
                self.assertTrue((fold[group].timestamp >= fold['start']).all())
                self.assertFalse(set(fold[group].truck_id) & set(report['truck_groups']['test']))
                val_keys.extend(zip(fold[group].tyre_id, fold[group].timestamp))
        self.assertEqual(len(val_keys), len(set(val_keys)))
        self.assertTrue(all(fold['end'] <= pd.Timestamp(report['test_start']) for fold in folds))

    def test_changed_final_test_targets_cannot_change_folds(self):
        from tyremind.ml.train import make_folds
        from tyremind.ml.split import TARGET
        dataset, report = self.fixture()
        duplicate = dataset.copy()
        duplicate['tyre_id'] += '-extra'
        dataset = pd.concat([dataset, duplicate], ignore_index=True)
        before = make_folds(dataset, report)
        dataset.loc[dataset.timestamp >= pd.Timestamp(report['test_start']), TARGET] = -100
        after = make_folds(dataset, report)
        for left, right in zip(before, after):
            for group in ('train', 'seen', 'unseen'):
                pd.testing.assert_frame_equal(left[group], right[group])

    @unittest.skipUnless(importlib.util.find_spec('xgboost'), 'Install .[ml] for XGBoost integration test')
    def test_end_to_end_training_saves_models_and_reports(self):
        import joblib
        from tyremind.data.generate import Config, write_dataset
        from tyremind.data.preprocess import preprocess
        from tyremind.database.load import load_database
        from tyremind.features.build import export_features
        from tyremind.ml.train import train_compare
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(Config(trucks=4), root / 'raw')
            preprocess(root / 'raw', root / 'processed')
            load_database(root / 'processed/clean', root / 'tyremind.db')
            export_features(root / 'tyremind.db', root / 'features')
            candidates = {'random_forest': [{'n_estimators': 3, 'max_depth': 3}],
                          'xgboost': [{'n_estimators': 3, 'max_depth': 2, 'learning_rate': 0.1}]}
            with patch('tyremind.ml.train.CANDIDATES', candidates):
                report = train_compare(root / 'features/tyre_features.csv', root / 'raw/evaluation_labels.csv', root / 'models')
            self.assertEqual(len(report['folds']), 3)
            self.assertEqual(set(report['final_test_results']), {'random_forest', 'xgboost', 'latest_tread', 'wear_rate'})
            self.assertTrue((root / 'models/metrics_summary.csv').exists())
            model = joblib.load(root / 'models/random_forest.joblib')
            self.assertNotIn('target_available_at', model['feature_columns'])
            self.assertNotIn('scenario_label', model['feature_columns'])


if __name__ == '__main__':
    unittest.main()
