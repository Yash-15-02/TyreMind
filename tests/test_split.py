"""Check exact-mileage targets, boundary purging and truck isolation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.features.build import FEATURE_COLUMNS
from tyremind.ml.split import TARGET, prepare_targets, split_dataset, export_splits


def fixture():
    dates = pd.date_range('2026-01-01', periods=60, tz='UTC')
    rows, labels = [], []
    for truck in range(6):
        for day, timestamp in enumerate(dates):
            row = {column: 1.0 for column in FEATURE_COLUMNS}
            row.update(truck_id=f'T{truck}', tyre_id=f'Y{truck}', timestamp=timestamp,
                       cumulative_mileage_km=day * 1000.0)
            rows.append(row)
            labels.append(dict(tyre_id=f'Y{truck}', timestamp=timestamp,
                               simulated_true_tread_depth_mm=15 - day * 0.1))
    return pd.DataFrame(rows), pd.DataFrame(labels)


class SplitTests(unittest.TestCase):
    def test_exact_and_interpolated_targets(self):
        features, labels = fixture()
        exact = prepare_targets(features, labels)
        first = exact.iloc[0]
        self.assertAlmostEqual(first[TARGET], 14.5)
        self.assertEqual(first.target_available_at, first.timestamp + pd.Timedelta(days=5))
        interpolated = prepare_targets(features, labels, horizon_km=5500)
        self.assertAlmostEqual(interpolated.iloc[0][TARGET], 14.45)
        self.assertEqual(interpolated.iloc[0].target_available_at,
                         first.timestamp + pd.Timedelta(days=6))
        self.assertTrue(exact.groupby('tyre_id').tail(5)[TARGET].isna().all())
        self.assertNotIn('simulated_true_tread_depth_mm', exact.columns)

    def test_time_boundaries_and_whole_truck_groups(self):
        features, labels = fixture()
        dataset = prepare_targets(features, labels)
        parts, excluded, report = split_dataset(dataset)
        validation = pd.Timestamp(report['validation_start'])
        test = pd.Timestamp(report['test_start'])
        self.assertTrue((parts['train'].target_available_at < validation).all())
        for name in ('validation_seen', 'validation_unseen'):
            self.assertTrue((parts[name].timestamp >= validation).all())
            self.assertTrue((parts[name].target_available_at < test).all())
        for name in ('test_seen', 'test_unseen'):
            self.assertTrue((parts[name].timestamp >= test).all())
        train_trucks = set(parts['train'].truck_id)
        self.assertTrue(set(parts['validation_seen'].truck_id) <= train_trucks)
        self.assertTrue(set(parts['test_seen'].truck_id) <= train_trucks)
        unseen_validation = set(parts['validation_unseen'].truck_id)
        unseen_test = set(parts['test_unseen'].truck_id)
        self.assertFalse(train_trucks & (unseen_validation | unseen_test))
        self.assertFalse(unseen_validation & unseen_test)
        all_rows = pd.concat([*parts.values(), excluded])
        self.assertEqual(len(all_rows), len(features))
        self.assertFalse(all_rows.duplicated(['tyre_id', 'timestamp']).any())
        self.assertGreater((excluded.exclusion_reason == 'target_crosses_time_cutoff').sum(), 0)

    def test_deterministic_export(self):
        features, labels = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features.to_csv(root / 'features.csv', index=False)
            labels.to_csv(root / 'labels.csv', index=False)
            export_splits(root / 'features.csv', root / 'labels.csv', root / 'splits')
            before = {p.name: p.read_bytes() for p in (root / 'splits').iterdir()}
            export_splits(root / 'features.csv', root / 'labels.csv', root / 'splits')
            after = {p.name: p.read_bytes() for p in (root / 'splits').iterdir()}
            self.assertEqual(before, after)

    def test_invalid_or_missing_truth_rejected(self):
        features, labels = fixture()
        with self.assertRaisesRegex(ValueError, 'matching'):
            prepare_targets(features, labels.iloc[1:])
        features.loc[1, 'cumulative_mileage_km'] = 0
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            prepare_targets(features, labels)
        with self.assertRaisesRegex(ValueError, 'horizon'):
            prepare_targets(features, labels, horizon_km=np.inf)

    def test_future_labels_cannot_change_training_targets(self):
        features, labels = fixture()
        before, _, report = split_dataset(prepare_targets(features, labels))
        cutoff = pd.Timestamp(report['validation_start'])
        labels.loc[labels.timestamp >= cutoff, 'simulated_true_tread_depth_mm'] = 2
        after, _, _ = split_dataset(prepare_targets(features, labels))
        pd.testing.assert_frame_equal(before['train'], after['train'])

    def test_small_dataset_fails_clearly(self):
        features, labels = fixture()
        dataset = prepare_targets(features, labels)
        with self.assertRaisesRegex(ValueError, '3 trucks'):
            split_dataset(dataset[dataset.truck_id == 'T0'])


if __name__ == '__main__':
    unittest.main()
