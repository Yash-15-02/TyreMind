"""Compare Random Forest and XGBoost with purged time/truck cross-validation."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from tyremind.features.build import FEATURE_COLUMNS
from tyremind.ml.split import TARGET, prepare_targets, split_dataset

CANDIDATES = {
    'random_forest': [
        {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 3, 'max_features': 0.7},
        {'n_estimators': 200, 'max_depth': None, 'min_samples_leaf': 2, 'max_features': 1.0},
        {'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 5, 'max_features': 1.0},
    ],
    'xgboost': [
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 5},
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 5},
        {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.08, 'subsample': 1.0, 'colsample_bytree': 1.0, 'reg_lambda': 10},
    ],
}


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not len(actual) or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Metrics require nonempty, finite targets and predictions.')
    error = predicted - actual
    return {'rows': len(actual), 'mae_mm': float(mean_absolute_error(actual, predicted)),
            'rmse_mm': float(np.sqrt(mean_squared_error(actual, predicted))),
            'r2': float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None,
            'median_absolute_error_mm': float(np.median(np.abs(error))),
            'p90_absolute_error_mm': float(np.quantile(np.abs(error), 0.9)),
            'bias_mm': float(np.mean(error)),
            'within_0_25mm_fraction': float(np.mean(np.abs(error) <= 0.25)),
            'within_0_5mm_fraction': float(np.mean(np.abs(error) <= 0.5))}


def baseline_fit(training):
    wear = training.historical_wear_mm_per_1000km.dropna().clip(lower=0)
    tread = training.latest_measured_tread_mm.dropna()
    return {'wear_rate': float(wear.median()) if len(wear) else 0.0,
            'tread_mm': float(tread.median()) if len(tread) else 0.0}


def baseline_predict(frame, settings, kind='wear_rate'):
    tread = frame.latest_measured_tread_mm.fillna(settings['tread_mm']).to_numpy()
    if kind == 'latest_tread':
        return tread
    wear = frame.historical_wear_mm_per_1000km.fillna(settings['wear_rate']).clip(lower=0).to_numpy()
    distance = frame.distance_since_inspection_km.fillna(0).to_numpy() + 5000
    return np.maximum(0, tread - wear * distance / 1000)


def make_model(family, parameters, seed=42, jobs=2):
    if family == 'random_forest':
        regressor = RandomForestRegressor(**parameters, bootstrap=True, random_state=seed, n_jobs=jobs)
    elif family == 'xgboost':
        from xgboost import XGBRegressor
        regressor = XGBRegressor(**parameters, objective='reg:squarederror', tree_method='hist',
                                 random_state=seed, n_jobs=jobs)
    else:
        raise ValueError(f'Unknown model family: {family}')
    return Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                     ('regressor', regressor)])


def make_folds(dataset, split_report):
    """Three nonoverlapping validation windows before final testing."""
    test_start = pd.Timestamp(split_report['test_start'])
    dev_times = dataset.loc[dataset.timestamp < test_start, 'timestamp'].drop_duplicates().sort_values().tolist()
    if len(dev_times) < 40:
        raise ValueError('Insufficient development history for three useful temporal folds.')
    boundaries = [dev_times[int(len(dev_times) * fraction)] for fraction in (0.25, 0.5, 0.75)] + [test_start]
    groups = split_report['truck_groups']
    train_trucks, validation_trucks = set(groups['train']), set(groups['validation'])
    if train_trucks & validation_trucks or (train_trucks | validation_trucks) & set(groups['test']):
        raise ValueError('Truck groups must be disjoint.')
    supported = dataset[TARGET].notna() & dataset.target_available_at.notna()
    folds = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        train = dataset.loc[supported & dataset.truck_id.isin(train_trucks) &
                            (dataset.timestamp < start) & (dataset.target_available_at < start)].copy()
        validation = supported & (dataset.timestamp >= start) & (dataset.timestamp < end) & (dataset.target_available_at < end)
        seen = dataset.loc[validation & dataset.truck_id.isin(train_trucks)].copy()
        unseen = dataset.loc[validation & dataset.truck_id.isin(validation_trucks)].copy()
        if min(len(train), len(seen), len(unseen)) < 20:
            raise ValueError('A fold has too little 5,000-km target-supported history; generate more days.')
        folds.append({'fold': index, 'start': start, 'end': end, 'train': train,
                      'seen': seen, 'unseen': unseen})
    return folds


def evaluate_predictions(frame, predicted, scenarios):
    result = {'overall': metrics(frame[TARGET], predicted), 'by_scenario': {}}
    evidence = frame[['tyre_id', 'timestamp']].copy()
    evidence['actual'] = frame[TARGET].to_numpy()
    evidence['predicted'] = predicted
    evidence = evidence.merge(scenarios, on=['tyre_id', 'timestamp'], how='left', validate='one_to_one')
    for scenario, rows in evidence.groupby('scenario_label', dropna=False):
        result['by_scenario'][str(scenario)] = metrics(rows.actual, rows.predicted)
    return result


def train_compare(features=Path('data/features/tyre_features.csv'), labels=Path('data/generated/evaluation_labels.csv'),
                  output=Path('models/wear'), seed=42, jobs=2):
    if jobs < 1:
        raise ValueError('jobs must be positive.')
    features, labels, output = Path(features), Path(labels), Path(output)
    raw_features = pd.read_csv(features)
    truth = pd.read_csv(labels)
    dataset = prepare_targets(raw_features, truth)
    partitions, _, split_report = split_dataset(dataset, seed)
    folds = make_folds(dataset, split_report)
    # Scenario labels are evaluation metadata only, never inputs or row-selection criteria.
    scenarios = truth[['tyre_id', 'timestamp', 'scenario_label']].copy()
    scenarios['timestamp'] = pd.to_datetime(scenarios.timestamp, utc=True, format='mixed')
    trials, fold_scores, baselines = [], [], []
    fold_manifest = []
    for fold in folds:
        fold_manifest.append({'fold': fold['fold'], 'validation_start': fold['start'].isoformat(),
                              'validation_end': fold['end'].isoformat(),
                              'training_rows': len(fold['train']), 'seen_rows': len(fold['seen']),
                              'unseen_rows': len(fold['unseen']),
                              'latest_training_target': fold['train'].target_available_at.max().isoformat()})
        settings = baseline_fit(fold['train'])
        for kind in ('latest_tread', 'wear_rate'):
            scores = {group: metrics(fold[group][TARGET], baseline_predict(fold[group], settings, kind)) for group in ('seen', 'unseen')}
            baselines.append({'fold': fold['fold'], 'model': kind, **scores,
                              'balanced_mae_mm': float(np.mean([scores[g]['mae_mm'] for g in scores]))})
    for family, candidates in CANDIDATES.items():
        for index, parameters in enumerate(candidates):
            candidate_scores = []
            for fold in folds:
                started = time.perf_counter()
                pipeline = make_model(family, parameters, seed, jobs)
                pipeline.fit(fold['train'][FEATURE_COLUMNS], fold['train'][TARGET])
                elapsed = time.perf_counter() - started
                scores = {group: metrics(fold[group][TARGET], pipeline.predict(fold[group][FEATURE_COLUMNS])) for group in ('seen', 'unseen')}
                score = float(np.mean([scores[g]['mae_mm'] for g in scores]))
                candidate_scores.append(score)
                fold_scores.append({'model': family, 'candidate': index, 'fold': fold['fold'],
                                    'fit_seconds': elapsed, 'balanced_mae_mm': score, **scores})
                print(f'{family} candidate {index+1}/{len(candidates)}, fold {fold["fold"]}: MAE {score:.4f} mm', flush=True)
            trials.append({'model': family, 'candidate': index, 'parameters': parameters,
                           'cv_balanced_mae_mm': float(np.mean(candidate_scores)),
                           'cv_fold_mae_std_mm': float(np.std(candidate_scores)),
                           'cv_fold_mae_mm': candidate_scores})
    best = {family: min([trial for trial in trials if trial['model'] == family],
                         key=lambda trial: trial['cv_balanced_mae_mm']) for family in CANDIDATES}
    winner = min(best, key=lambda family: best[family]['cv_balanced_mae_mm'])
    baseline_summary = {kind: float(np.mean([row['balanced_mae_mm'] for row in baselines if row['model'] == kind]))
                        for kind in ('latest_tread', 'wear_rate')}
    selection_scores = {**{family: trial['cv_balanced_mae_mm'] for family, trial in best.items()}, **baseline_summary}
    recommended = min(selection_scores, key=selection_scores.get)
    # Freeze selections BEFORE inspecting final test metrics. Refit on training trucks only,
    # using all labels available before final test start (including formerly purged gaps).
    test_start = pd.Timestamp(split_report['test_start'])
    training = dataset.loc[dataset.truck_id.isin(split_report['truck_groups']['train']) &
                           (dataset.timestamp < test_start) & (dataset.target_available_at < test_start) & dataset[TARGET].notna()]
    output.mkdir(parents=True, exist_ok=True)
    final_results, prediction_rows = {}, []
    trained = {}
    for family, trial in best.items():
        pipeline = make_model(family, trial['parameters'], seed, jobs)
        pipeline.fit(training[FEATURE_COLUMNS], training[TARGET])
        trained[family] = pipeline
        artifact = {'pipeline': pipeline, 'feature_columns': FEATURE_COLUMNS, 'target': TARGET,
                    'horizon_km': 5000, 'model_family': family, 'parameters': trial['parameters'],
                    'seed': seed, 'trained_before': test_start.isoformat(), 'synthetic_only': True}
        path = output / f'{family}.joblib'
        joblib.dump(artifact, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(pipeline.predict(training[FEATURE_COLUMNS].iloc[:10]),
                                   restored['pipeline'].predict(training[FEATURE_COLUMNS].iloc[:10]))
    settings = baseline_fit(training)
    for family in (*best.keys(), 'latest_tread', 'wear_rate'):
        final_results[family] = {}
        for group in ('test_seen', 'test_unseen'):
            frame = partitions[group]
            predicted = trained[family].predict(frame[FEATURE_COLUMNS]) if family in trained else baseline_predict(frame, settings, family)
            final_results[family][group] = evaluate_predictions(frame, predicted, scenarios)
            rows = frame[['truck_id', 'tyre_id', 'timestamp', TARGET]].copy()
            rows['model'], rows['evaluation'], rows['predicted_tread_mm'] = family, group, predicted
            prediction_rows.append(rows)
    versions = {name: importlib.metadata.version(name) for name in ('scikit-learn', 'xgboost', 'pandas', 'numpy', 'joblib')}
    cv_summary = {}
    for family, trial in best.items():
        chosen = [row for row in fold_scores if row['model'] == family and row['candidate'] == trial['candidate']]
        cv_summary[family] = {'balanced_mae_mm': trial['cv_balanced_mae_mm'],
                              'fold_mae_std_mm': trial['cv_fold_mae_std_mm']}
        for metric in ('rmse_mm', 'r2', 'within_0_25mm_fraction', 'within_0_5mm_fraction'):
            cv_summary[family][f'balanced_mean_{metric}'] = float(np.mean([
                0.5 * (row['seen'][metric] + row['unseen'][metric]) for row in chosen]))
    report = {'seed': seed, 'feature_columns': FEATURE_COLUMNS, 'target': TARGET, 'horizon_km': 5000,
              'selection_metric': 'Mean across folds of 0.5 × seen MAE + 0.5 × unseen MAE',
              'best_ml_model_by_cv': winner, 'recommended_by_cv_including_baselines': recommended,
              'best_candidates': best, 'cv_summary': cv_summary, 'cv_baseline_mae_mm': baseline_summary, 'trials': trials,
              'fold_results': fold_scores, 'baseline_fold_results': baselines, 'folds': fold_manifest,
              'truck_groups': split_report['truck_groups'], 'final_training_rows': len(training),
              'test_start': test_start.isoformat(), 'final_test_results': final_results,
              'baseline_settings': settings, 'versions': versions,
              'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (features, labels)},
              'limitations': 'Synthetic-data performance only. No precision/recall for regression. Fold variation is not a confidence interval. Final test metrics never select parameters/models.'}
    (output / 'comparison_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pd.concat(prediction_rows, ignore_index=True).to_csv(output / 'test_predictions.csv', index=False)
    rows = []
    for model, evaluations in final_results.items():
        for group, result in evaluations.items():
            rows.append({'model': model, 'evaluation': group, **result['overall']})
    pd.DataFrame(rows).to_csv(output / 'metrics_summary.csv', index=False)
    print(f'Best ML model by CV: {winner}; recommendation including baselines: {recommended}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features/tyre_features.csv'))
    parser.add_argument('--labels', type=Path, default=Path('data/generated/evaluation_labels.csv'))
    parser.add_argument('--output', type=Path, default=Path('models/wear'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    report = train_compare(args.features, args.labels, args.output, args.seed, args.jobs)
    print(json.dumps({'best_ml_model_by_cv': report['best_ml_model_by_cv'],
                      'recommended_by_cv_including_baselines': report['recommended_by_cv_including_baselines'],
                      'results': report['final_test_results']}, indent=2))


if __name__ == '__main__':
    main()
