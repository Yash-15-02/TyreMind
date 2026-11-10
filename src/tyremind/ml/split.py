"""Prepare synthetic 5,000-km targets and leakage-aware time/truck partitions."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.features.build import FEATURE_COLUMNS

TARGET = 'target_tread_depth_5000km_mm'
METADATA = ['truck_id', 'tyre_id', 'timestamp', 'target_available_at']
SPLITS = ('train', 'validation_seen', 'validation_unseen', 'test_seen', 'test_unseen')


def prepare_targets(features, truth, horizon_km=5000.0):
    """Interpolate simulator truth at exactly the requested additional mileage.

    target_available_at is the later bracket timestamp, not interpolated time.
    Neither truth nor availability timestamps are predictors.
    """
    if not np.isfinite(horizon_km) or horizon_km <= 0:
        raise ValueError('Prediction horizon must be positive and finite.')
    required = ['truck_id', 'tyre_id', 'timestamp'] + FEATURE_COLUMNS
    missing = sorted(set(required) - set(features.columns))
    if missing:
        raise ValueError(f'Missing feature columns: {missing}')
    features = features[required].copy()
    truth = truth[['tyre_id', 'timestamp', 'simulated_true_tread_depth_mm']].copy()
    for frame in (features, truth):
        frame['timestamp'] = pd.to_datetime(frame.timestamp, utc=True, errors='raise', format='mixed')
        if frame[['tyre_id', 'timestamp']].isna().any().any():
            raise ValueError('Missing tyre ID or timestamp.')
        if frame.duplicated(['tyre_id', 'timestamp']).any():
            raise ValueError('Duplicate tyre/timestamp keys.')
    if features.truck_id.isna().any() or features.groupby('tyre_id').truck_id.nunique().gt(1).any():
        raise ValueError('Each tyre must belong to exactly one known truck.')
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors='raise')
        if np.isinf(features[column].to_numpy(dtype=float)).any():
            raise ValueError(f'Infinite feature values: {column}')
    truth['simulated_true_tread_depth_mm'] = pd.to_numeric(truth.simulated_true_tread_depth_mm, errors='raise')
    joined = features.merge(truth, on=['tyre_id', 'timestamp'], how='left', validate='one_to_one')
    if not np.isfinite(joined.simulated_true_tread_depth_mm).all() or (joined.simulated_true_tread_depth_mm < 0).any():
        raise ValueError('Each feature row needs matching finite, nonnegative simulator truth.')
    results = []
    for _, history in joined.groupby('tyre_id', sort=True):
        history = history.sort_values('timestamp').reset_index(drop=True)
        mileage = history.cumulative_mileage_km.to_numpy(dtype=float)
        if not np.isfinite(mileage).all() or (mileage < 0).any() or (np.diff(mileage) <= 0).any():
            raise ValueError('Target preparation requires strictly increasing tyre mileage.')
        depth = history.simulated_true_tread_depth_mm.to_numpy(dtype=float)
        desired = mileage + horizon_km
        upper = np.searchsorted(mileage, desired, side='left')
        supported = upper < len(history)
        targets = np.full(len(history), np.nan)
        available = pd.Series(pd.NaT, index=history.index, dtype='datetime64[ns, UTC]')
        indices = np.flatnonzero(supported)
        hi = upper[supported]
        lo = hi - 1
        weight = (desired[supported] - mileage[lo]) / (mileage[hi] - mileage[lo])
        targets[supported] = depth[lo] + weight * (depth[hi] - depth[lo])
        available.iloc[indices] = history.timestamp.iloc[hi].to_numpy()
        history[TARGET] = targets
        history['target_available_at'] = available
        results.append(history.drop(columns='simulated_true_tread_depth_mm'))
    if not results:
        raise ValueError('No feature rows available.')
    return pd.concat(results, ignore_index=True)


def split_dataset(dataset, seed=42):
    """60/20/20 time periods and 70/15/15 whole-truck groups.

    Seen evaluation uses training trucks; unseen evaluation uses disjoint groups.
    Validation/test periods purge targets extending into the next period/end.
    """
    times = dataset.timestamp.drop_duplicates().sort_values().tolist()
    trucks = sorted(dataset.truck_id.unique())
    if len(times) < 10 or len(trucks) < 3:
        raise ValueError('Need at least 10 timestamps and 3 trucks for these splits.')
    validation_start = times[int(len(times) * 0.6)]
    test_start = times[int(len(times) * 0.8)]
    shuffled = np.random.default_rng(seed).permutation(trucks).tolist()
    heldout_count = max(1, int(len(trucks) * 0.15))
    groups = {'train': sorted(shuffled[:-2 * heldout_count]),
              'validation': sorted(shuffled[-2 * heldout_count:-heldout_count]),
              'test': sorted(shuffled[-heldout_count:])}
    frame = dataset.copy()
    frame['partition'] = 'excluded'
    frame['exclusion_reason'] = 'reserved_truck_outside_evaluation_period'
    supported = frame[TARGET].notna() & frame.target_available_at.notna()
    frame.loc[~supported, 'exclusion_reason'] = 'insufficient_future_distance'
    periods = {
        'train': (frame.timestamp < validation_start, validation_start, groups['train']),
        'validation_seen': ((frame.timestamp >= validation_start) & (frame.timestamp < test_start), test_start, groups['train']),
        'validation_unseen': ((frame.timestamp >= validation_start) & (frame.timestamp < test_start), test_start, groups['validation']),
        'test_seen': (frame.timestamp >= test_start, None, groups['train']),
        'test_unseen': (frame.timestamp >= test_start, None, groups['test']),
    }
    for name, (period, cutoff, truck_group) in periods.items():
        candidate = period & frame.truck_id.isin(truck_group) & supported
        eligible = candidate if cutoff is None else candidate & (frame.target_available_at < cutoff)
        frame.loc[eligible, 'partition'] = name
        frame.loc[eligible, 'exclusion_reason'] = ''
        if cutoff is not None:
            frame.loc[candidate & ~eligible, 'exclusion_reason'] = 'target_crosses_time_cutoff'
    partitions = {name: frame.loc[frame.partition == name].drop(
        columns=['partition', 'exclusion_reason']).reset_index(drop=True) for name in SPLITS}
    empty = [name for name, rows in partitions.items() if rows.empty]
    if empty:
        raise ValueError(f'Not enough target-supported history for: {empty}. Generate a longer history.')
    report = {
        'seed': seed, 'time_fractions': [0.6, 0.2, 0.2],
        'validation_start': validation_start.isoformat(), 'test_start': test_start.isoformat(),
        'truck_groups': groups, 'input_rows': len(frame),
        'row_counts': {name: len(rows) for name, rows in partitions.items()},
        'excluded_rows': int(frame.partition.eq('excluded').sum()),
        'exclusion_reasons': frame.loc[frame.partition == 'excluded', 'exclusion_reason'].value_counts().to_dict(),
        'feature_columns': FEATURE_COLUMNS, 'target_column': TARGET,
        'metadata_columns': METADATA, 'imputation': 'none',
        'evaluation': 'Seen uses training trucks later in time; unseen uses reserved validation/test trucks.',
        'warning': 'Simulator-truth targets; results do not establish real-world accuracy.',
    }
    excluded = frame.loc[frame.partition == 'excluded'].drop(columns='partition').reset_index(drop=True)
    return partitions, excluded, report


def export_splits(features=Path('data/features/tyre_features.csv'),
                  labels=Path('data/generated/evaluation_labels.csv'),
                  output=Path('data/splits'), seed=42, horizon_km=5000.0):
    # Read only the truth fields required for targets, never scenario labels.
    feature_rows = pd.read_csv(features, dtype={'truck_id': 'string', 'tyre_id': 'string'})
    truth = pd.read_csv(labels, usecols=['tyre_id', 'timestamp', 'simulated_true_tread_depth_mm'],
                        dtype={'tyre_id': 'string'})
    dataset = prepare_targets(feature_rows, truth, horizon_km)
    partitions, excluded, report = split_dataset(dataset, seed)
    report['horizon_km'] = horizon_km
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in partitions.items():
        frame.to_csv(output / f'{name}.csv', index=False)
    excluded.to_csv(output / 'excluded.csv', index=False)
    (output / 'split_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, default=Path('data/features/tyre_features.csv'))
    parser.add_argument('--labels', type=Path, default=Path('data/generated/evaluation_labels.csv'))
    parser.add_argument('--output', type=Path, default=Path('data/splits'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(export_splits(args.features, args.labels, args.output, args.seed), indent=2))


if __name__ == '__main__':
    main()
