"""Reload saved models, verify held-out metrics and inventory today's artifacts."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tyremind.features.build import FEATURE_COLUMNS
from tyremind.ml.anomaly import (ANOMALY_FEATURES, classification_metrics, partition_data,
                                  predict_anomalies, prepare_data)
from tyremind.ml.split import TARGET, prepare_targets, split_dataset
from tyremind.ml.train import metrics


def verify_inputs(report, root):
    for filename, expected in report['input_sha256'].items():
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Input changed since training: {filename}. Reconcile artifacts before finalizing.')


def verify_metrics(actual, expected, names):
    for name in names:
        if actual[name] is None or expected[name] is None:
            if actual[name] != expected[name]:
                raise ValueError(f'Reloaded metric mismatch: {name}')
        elif not np.isclose(actual[name], expected[name], rtol=1e-5, atol=1e-7):
            raise ValueError(f'Reloaded metric mismatch: {name}')


def finalize(root=Path('.')):
    root = Path(root).resolve()
    wear_report = json.loads((root / 'models/wear/comparison_report.json').read_text())
    anomaly_report = json.loads((root / 'models/anomaly/anomaly_report.json').read_text())
    verify_inputs(wear_report, root)
    verify_inputs(anomaly_report, root)
    features = pd.read_csv(root / 'data/features/tyre_features.csv')
    labels = pd.read_csv(root / 'data/generated/evaluation_labels.csv')
    dataset = prepare_targets(features, labels)
    partitions, _, _ = split_dataset(dataset, wear_report['seed'])
    entries, summary = [], []
    for family in ('random_forest', 'xgboost'):
        relative = f'models/wear/{family}.joblib'
        path = root / relative
        artifact = joblib.load(path)
        if artifact['feature_columns'] != FEATURE_COLUMNS or artifact['model_family'] != family:
            raise ValueError(f'{family}: artifact feature/schema mismatch.')
        evaluations = {}
        for group in ('test_seen', 'test_unseen'):
            frame = partitions[group]
            result = metrics(frame[TARGET], artifact['pipeline'].predict(frame[FEATURE_COLUMNS]))
            expected = wear_report['final_test_results'][family][group]['overall']
            verify_metrics(result, expected, ('rows', 'mae_mm', 'rmse_mm', 'r2'))
            evaluations[group] = result
            summary.append({'model': family, 'task': 'wear_regression', 'evaluation': group, **result})
        entries.append({'model': family, 'task': 'wear_regression', 'path': relative,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
                        'feature_columns': artifact['feature_columns'], 'horizon_km': artifact['horizon_km'],
                        'reloaded_test_metrics': evaluations})
    path = root / 'models/anomaly/isolation_forest.joblib'
    artifact = joblib.load(path)
    if artifact['feature_columns'] != ANOMALY_FEATURES or not np.isclose(artifact['threshold'], anomaly_report['selected']['threshold']):
        raise ValueError('Anomaly artifact/schema/threshold mismatch.')
    split_report = json.loads((root / 'data/splits/split_report.json').read_text())
    parts = partition_data(prepare_data(features, labels), split_report)
    evaluations = {}
    for group in ('test_seen', 'test_unseen'):
        frame = parts[group]
        scores, _ = predict_anomalies(artifact, frame)
        result = classification_metrics(frame.is_fault, scores, artifact['threshold'])
        expected = anomaly_report['evaluations'][group]['overall']
        verify_metrics(result, expected, ('rows', 'precision', 'recall', 'f1', 'false_positive_rate'))
        if result['confusion_matrix'] != expected['confusion_matrix']:
            raise ValueError('Reloaded anomaly confusion counts differ.')
        evaluations[group] = result
        summary.append({'model': 'isolation_forest', 'task': 'anomaly_detection', 'evaluation': group,
                        **{key: value for key, value in result.items() if key != 'confusion_matrix'}})
    entries.append({'model': 'isolation_forest', 'task': 'anomaly_detection',
                    'path': 'models/anomaly/isolation_forest.joblib',
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size,
                    'feature_columns': artifact['feature_columns'], 'threshold': artifact['threshold'],
                    'reloaded_test_metrics': evaluations})
    manifest = {'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'models': entries,
                'verification': 'All three persisted models reproduced their recorded held-out metrics.',
                'best_ml_wear_candidate_by_cv': wear_report['best_ml_model_by_cv'],
                'best_wear_candidate_including_baselines': wear_report['recommended_by_cv_including_baselines'],
                'status': 'Experimental synthetic-data models; no live deployment selected.',
                'limitations': ['Wear-rate baseline outperformed the fitted ML candidates in this experiment.',
                                'Isolation Forest held-out recall is low; it must not be the sole alert trigger.',
                                'No retraining, threshold tuning or model selection occurs during finalization.']}
    output = root / 'models/evaluation'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'model_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    pd.DataFrame(summary).to_csv(output / 'saved_model_metrics.csv', index=False)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('.'))
    args = parser.parse_args()
    result = finalize(args.project)
    print(json.dumps({'verified_models': len(result['models']), 'verification': result['verification'],
                      'best_ml_wear_candidate_by_cv': result['best_ml_wear_candidate_by_cv'],
                      'best_wear_candidate_including_baselines': result['best_wear_candidate_including_baselines'],
                      'output': 'models/evaluation'}, indent=2))


if __name__ == '__main__':
    main()
