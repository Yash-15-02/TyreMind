"""Fit Isolation Forest on normal history and calibrate alerts on validation only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

ANOMALY_FEATURES = ['pressure_deviation_ratio', 'pressure_change_kpa_per_day',
                    'temperature_mean_7d_c', 'temperature_std_7d_c',
                    'load_mean_7d', 'overload_fraction_7d', 'historical_wear_mm_per_1000km']
CANDIDATES = [dict(n_estimators=200, max_samples=256, max_features=1.0),
              dict(n_estimators=200, max_samples=512, max_features=1.0),
              dict(n_estimators=200, max_samples=256, max_features=0.8)]


def parse_boolean(values, name):
    result = values.astype('string').str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False})
    if result.isna().any():
        raise ValueError(f'{name}: invalid or missing boolean labels.')
    return result.astype(bool)


def prepare_data(features, labels):
    columns = ['truck_id', 'tyre_id', 'timestamp', 'observations_7d'] + ANOMALY_FEATURES
    frame = features[columns].copy()
    truth = labels[['tyre_id', 'timestamp', 'scenario_label', 'scenario_active', 'overload_active']].copy()
    for table in (frame, truth):
        table['timestamp'] = pd.to_datetime(table.timestamp, utc=True, errors='raise', format='mixed')
        if table[['tyre_id', 'timestamp']].isna().any().any() or table.duplicated(['tyre_id', 'timestamp']).any():
            raise ValueError('Missing or duplicate tyre/timestamp identities.')
    if frame.truck_id.isna().any() or frame.groupby('tyre_id').truck_id.nunique().gt(1).any():
        raise ValueError('Invalid tyre/truck relationship.')
    for column in ANOMALY_FEATURES + ['observations_7d']:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError('Infinite feature values.')
    frame = frame.merge(truth, on=['tyre_id', 'timestamp'], how='left', validate='one_to_one')
    if frame.scenario_label.isna().any():
        raise ValueError('Feature rows require matching evaluation labels.')
    for column in ('scenario_active', 'overload_active'):
        frame[column] = parse_boolean(frame[column], column)
    frame['is_fault'] = frame.scenario_active | frame.overload_active
    frames = []
    for _, history in frame.groupby('tyre_id', sort=True):
        history = history.sort_values('timestamp').copy()
        # Avoid fitting on residual rolling/inspection effects immediately after repair.
        recent_fault = history.set_index('timestamp').is_fault.astype(int).rolling('28D', min_periods=1).max()
        history['normal_reference'] = recent_fault.to_numpy() == 0
        history['eligible'] = history.observations_7d.ge(3)
        frames.append(history)
    if not frames:
        raise ValueError('No data available.')
    return pd.concat(frames, ignore_index=True)


def partition_data(frame, report):
    groups = report['truck_groups']
    train, validation, test = [set(groups[name]) for name in ('train', 'validation', 'test')]
    if train & validation or train & test or validation & test:
        raise ValueError('Train/validation/test trucks overlap.')
    if set(frame.truck_id) != train | validation | test:
        raise ValueError('Truck groups do not match the supplied data.')
    start, end = pd.Timestamp(report['validation_start']), pd.Timestamp(report['test_start'])
    if start >= end:
        raise ValueError('Invalid time boundaries.')
    eligible = frame.eligible
    output = {'train_normal': frame.loc[eligible & frame.normal_reference & frame.truck_id.isin(train) & (frame.timestamp < start)].copy()}
    for name, period, trucks in (
        ('validation_seen', (frame.timestamp >= start) & (frame.timestamp < end), train),
        ('validation_unseen', (frame.timestamp >= start) & (frame.timestamp < end), validation),
        ('test_seen', frame.timestamp >= end, train),
        ('test_unseen', frame.timestamp >= end, test)):
        output[name] = frame.loc[eligible & period & frame.truck_id.isin(trucks)].copy()
    if len(output['train_normal']) < 100 or any(rows.empty for rows in output.values()):
        raise ValueError('Insufficient normal training or evaluation rows.')
    for name in ('validation_seen', 'validation_unseen'):
        if output[name].is_fault.nunique() != 2:
            raise ValueError(f'{name} requires both normal and abnormal validation examples.')
    return output


def classification_metrics(actual, scores, threshold):
    actual = np.asarray(actual, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if not len(actual) or len(actual) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Need aligned finite scores and nonempty labels.')
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    both = np.unique(actual).size == 2
    return {'rows': len(actual), 'normal_rows': int(tn + fp), 'fault_rows': int(tp + fn),
            'precision': float(precision_score(actual, predicted, zero_division=0)),
            'recall': float(recall_score(actual, predicted, zero_division=0)),
            'f1': float(fbeta_score(actual, predicted, beta=1, zero_division=0)),
            'f2': float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
            'accuracy': float(np.mean(actual == predicted)),
            'false_positive_rate': float(fp / (fp + tn)) if fp + tn else None,
            'specificity': float(tn / (fp + tn)) if fp + tn else None,
            'average_precision': float(average_precision_score(actual, scores)) if both else None,
            'roc_auc': float(roc_auc_score(actual, scores)) if both else None,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}}


def select_threshold(evaluations, fpr_limit=0.05):
    """Favor recall via F2, subject to the FPR limit in BOTH validation groups."""
    if not 0 <= fpr_limit <= 1:
        raise ValueError('False-positive-rate limit must be between 0 and 1.')
    all_scores = np.concatenate([scores for _, scores in evaluations.values()])
    # A bounded validation-only grid, including a no-alert candidate.
    thresholds = np.unique(np.r_[np.quantile(all_scores, np.linspace(0, 1, 101)),
                                 np.nextafter(all_scores.max(), np.inf)])
    best = None
    for threshold in thresholds:
        results = {name: classification_metrics(actual, scores, threshold)
                   for name, (actual, scores) in evaluations.items()}
        if any(result['false_positive_rate'] is None or result['false_positive_rate'] > fpr_limit for result in results.values()):
            continue
        f2 = float(np.mean([result['f2'] for result in results.values()]))
        recall = float(np.mean([result['recall'] for result in results.values()]))
        mean_fpr = float(np.mean([result['false_positive_rate'] for result in results.values()]))
        ranking = (f2, recall, -mean_fpr, float(threshold))
        if best is None or ranking > best[0]:
            best = (ranking, {'threshold': float(threshold), 'balanced_f2': f2,
                             'balanced_recall': recall, 'validation': results})
    if best is None:
        raise ValueError('No feasible validation threshold.')
    return best[1]


def make_pipeline(parameters, seed=42, jobs=2):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('isolation_forest', IsolationForest(**parameters, contamination='auto', random_state=seed, n_jobs=jobs)),
    ])


def predict_anomalies(artifact, features):
    """Inference needs only seven operational features, never simulator labels."""
    values = features[artifact['feature_columns']].apply(pd.to_numeric, errors='raise')
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError('Infinite features cannot be scored.')
    scores = -artifact['pipeline'].score_samples(values)
    return scores, scores >= artifact['threshold']


def evaluate(frame, scores, threshold):
    result = {'overall': classification_metrics(frame.is_fault, scores, threshold), 'by_assigned_scenario': {},
              'active_fault_recall': {}}
    rows = frame.copy()
    rows['score'] = scores
    for scenario, subset in rows.groupby('scenario_label'):
        result['by_assigned_scenario'][scenario] = classification_metrics(subset.is_fault, subset.score, threshold)
    for scenario, subset in rows.loc[rows.scenario_active].groupby('scenario_label'):
        result['active_fault_recall'][scenario] = {'rows': len(subset), 'recall': float((subset.score >= threshold).mean())}
    overload = rows.loc[rows.overload_active]
    if len(overload):
        result['active_fault_recall']['truck_overloading'] = {'rows': len(overload), 'recall': float((overload.score >= threshold).mean())}
    return result


def train_anomaly(features=Path('data/features/tyre_features.csv'), labels=Path('data/generated/evaluation_labels.csv'),
                  split=Path('data/splits/split_report.json'), output=Path('models/anomaly'),
                  seed=42, jobs=2, fpr_limit=0.05):
    if jobs < 1:
        raise ValueError('jobs must be positive.')
    features, labels, split, output = map(Path, (features, labels, split, output))
    frame = prepare_data(pd.read_csv(features), pd.read_csv(labels))
    split_report = json.loads(split.read_text())
    if int(split_report['input_rows']) != len(frame):
        raise ValueError('Split report row count does not match feature export.')
    parts = partition_data(frame, split_report)
    trials, pipelines = [], []
    for index, parameters in enumerate(CANDIDATES):
        pipeline = make_pipeline(parameters, seed, jobs)
        pipeline.fit(parts['train_normal'][ANOMALY_FEATURES])
        validation = {name: (parts[name].is_fault, -pipeline.score_samples(parts[name][ANOMALY_FEATURES]))
                      for name in ('validation_seen', 'validation_unseen')}
        selection = select_threshold(validation, fpr_limit)
        trials.append({'candidate': index, 'parameters': parameters, **selection})
        pipelines.append(pipeline)
        print(f'Isolation Forest candidate {index+1}/{len(CANDIDATES)}: '
              f'validation F2={selection["balanced_f2"]:.4f}, recall={selection["balanced_recall"]:.4f}', flush=True)
    selected = max(trials, key=lambda trial: (trial['balanced_f2'], trial['balanced_recall'], -trial['candidate']))
    # Retain the selected fitted pipeline: refitting would invalidate its calibrated threshold.
    artifact = {'pipeline': pipelines[selected['candidate']], 'threshold': selected['threshold'],
                'feature_columns': ANOMALY_FEATURES, 'model_family': 'isolation_forest',
                'parameters': selected['parameters'], 'seed': seed,
                'trained_before': split_report['validation_start'],
                'score_definition': '-score_samples; higher means more unusual, not a probability',
                'warmup_min_observations_7d': 3, 'synthetic_only': True}
    final_results, predictions = {}, []
    for name in ('validation_seen', 'validation_unseen', 'test_seen', 'test_unseen'):
        rows = parts[name]
        scores, flags = predict_anomalies(artifact, rows)
        final_results[name] = evaluate(rows, scores, artifact['threshold'])
        result = rows[['truck_id', 'tyre_id', 'timestamp', 'scenario_label', 'scenario_active', 'overload_active', 'is_fault']].copy()
        result['evaluation'], result['anomaly_score'], result['anomaly_flag'] = name, scores, flags
        predictions.append(result)
    report = {'model': 'IsolationForest', 'feature_columns': ANOMALY_FEATURES,
              'label_definition': 'scenario_active OR overload_active; no tread safety threshold',
              'normal_reference_rule': 'No active injected fault or truck overload in trailing 28 days; at least 3 readings in trailing 7 days.',
              'training_rows': len(parts['train_normal']), 'warmup_excluded_rows': int((~frame.eligible).sum()),
              'training_fault_rows': int(parts['train_normal'].is_fault.sum()),
              'validation_start': split_report['validation_start'], 'test_start': split_report['test_start'],
              'truck_groups': split_report['truck_groups'], 'seed': seed, 'validation_fpr_limit': fpr_limit,
              'selection_policy': 'Maximize balanced validation F2 with FPR <= limit separately in seen/unseen groups.',
              'selected': selected, 'candidates': trials, 'evaluations': final_results,
              'versions': {name: importlib.metadata.version(name) for name in ('scikit-learn', 'numpy', 'pandas', 'joblib')},
              'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (features, labels, split)},
              'limitations': 'Synthetic row-level detection only. Labels curate normal reference and tune threshold; forest fitting is unsupervised. No diagnosis, safety certification or event-level detection guarantee.'}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output / 'isolation_forest.joblib')
    reloaded = joblib.load(output / 'isolation_forest.joblib')
    expected = predict_anomalies(artifact, parts['test_seen'].iloc[:10])
    actual = predict_anomalies(reloaded, parts['test_seen'].iloc[:10])
    np.testing.assert_allclose(expected[0], actual[0])
    np.testing.assert_array_equal(expected[1], actual[1])
    (output / 'anomaly_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pd.concat(predictions, ignore_index=True).to_csv(output / 'anomaly_predictions.csv', index=False)
    summaries = []
    for name, result in final_results.items():
        overall = result['overall'].copy()
        confusion = overall.pop('confusion_matrix')
        summaries.append({'evaluation': name, **overall, **confusion})
    pd.DataFrame(summaries).to_csv(output / 'metrics_summary.csv', index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features/tyre_features.csv'))
    parser.add_argument('--labels', type=Path, default=Path('data/generated/evaluation_labels.csv'))
    parser.add_argument('--split-report', type=Path, default=Path('data/splits/split_report.json'))
    parser.add_argument('--output', type=Path, default=Path('models/anomaly'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--max-fpr', type=float, default=0.05)
    args = parser.parse_args()
    report = train_anomaly(args.features, args.labels, args.split_report, args.output, args.seed, args.jobs, args.max_fpr)
    print(json.dumps({'training_rows': report['training_rows'], 'threshold': report['selected']['threshold'],
                     'evaluations': {name: result['overall'] for name, result in report['evaluations'].items()}}, indent=2))


if __name__ == '__main__':
    main()
