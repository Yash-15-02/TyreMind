"""Build point-in-time features from SQLite without future observations."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    'pressure_deviation_ratio', 'pressure_change_kpa_per_day',
    'pressure_mean_7d_kpa', 'temperature_mean_7d_c', 'temperature_std_7d_c',
    'distance_sum_7d_km', 'load_mean_7d', 'overload_fraction_7d',
    'observations_7d', 'tyre_age_days', 'cumulative_mileage_km',
    'latest_measured_tread_mm', 'days_since_inspection',
    'distance_since_inspection_km', 'historical_wear_mm_per_1000km',
    'days_since_maintenance', 'has_inspection', 'has_wear_history', 'has_maintenance',
]


def build_features(telemetry, tyres, inspections, maintenance):
    """One output row per reading; same-timestamp records are available by assumption."""
    telemetry, tyres, inspections, maintenance = [
        frame.copy() for frame in (telemetry, tyres, inspections, maintenance)]
    for frame, column in ((telemetry, 'timestamp'), (tyres, 'installation_timestamp'),
                          (inspections, 'timestamp'), (maintenance, 'timestamp')):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors='raise')
    if telemetry.duplicated(['tyre_id', 'timestamp']).any():
        raise ValueError('Duplicate telemetry keys; run preprocessing first.')
    metadata = tyres.set_index('tyre_id')
    result = []
    for tyre_id, raw in telemetry.groupby('tyre_id', sort=True):
        if tyre_id not in metadata.index:
            raise ValueError(f'Unknown tyre: {tyre_id}')
        history = raw.sort_values('timestamp').reset_index(drop=True)
        meta = metadata.loc[tyre_id]
        if meta.reference_pressure_kpa <= 0:
            raise ValueError('Reference pressure must be positive.')
        output = history[['truck_id', 'tyre_id', 'timestamp', 'cumulative_mileage_km']].copy()
        output['pressure_deviation_ratio'] = history.pressure_kpa / meta.reference_pressure_kpa - 1
        elapsed = history.timestamp.diff().dt.total_seconds() / 86400
        output['pressure_change_kpa_per_day'] = history.pressure_kpa.diff() / elapsed
        timed = history.set_index('timestamp')
        rolling = timed.rolling('7D', min_periods=3)
        # Right-closed window: (timestamp - 7 days, timestamp], never centered.
        output['pressure_mean_7d_kpa'] = rolling.pressure_kpa.mean().to_numpy()
        output['temperature_mean_7d_c'] = rolling.temperature_c.mean().to_numpy()
        output['temperature_std_7d_c'] = rolling.temperature_c.std().to_numpy()
        output['distance_sum_7d_km'] = rolling.daily_distance_km.sum().to_numpy()
        output['load_mean_7d'] = rolling.load_ratio.mean().to_numpy()
        output['overload_fraction_7d'] = timed.load_ratio.gt(1).astype(float).rolling(
            '7D', min_periods=3).mean().to_numpy()
        output['observations_7d'] = timed.pressure_kpa.rolling('7D').count().to_numpy()
        output['tyre_age_days'] = (history.timestamp - meta.installation_timestamp).dt.total_seconds() / 86400

        observed = inspections.loc[inspections.tyre_id == tyre_id].sort_values(
            ['timestamp', 'inspection_id']).drop_duplicates('timestamp', keep='last')
        if len(observed):
            observed = observed[['timestamp', 'tread_depth_mm']].rename(
                columns={'timestamp': 'inspection_timestamp', 'tread_depth_mm': 'latest_measured_tread_mm'})
            # Mileage at the inspection uses the latest reading AT OR BEFORE it.
            observed = pd.merge_asof(observed, history[['timestamp', 'cumulative_mileage_km']],
                                     left_on='inspection_timestamp', right_on='timestamp', direction='backward')
            observed = observed.drop(columns='timestamp').rename(
                columns={'cumulative_mileage_km': 'inspection_mileage_km'})
            distance = observed.inspection_mileage_km.diff()
            observed['historical_wear_mm_per_1000km'] = (
                -observed.latest_measured_tread_mm.diff() * 1000 / distance.where(distance > 0))
            output = pd.merge_asof(output, observed, left_on='timestamp',
                                   right_on='inspection_timestamp', direction='backward')
            output['days_since_inspection'] = (
                output.timestamp - output.inspection_timestamp).dt.total_seconds() / 86400
            output['distance_since_inspection_km'] = output.cumulative_mileage_km - output.inspection_mileage_km
            output = output.drop(columns=['inspection_timestamp', 'inspection_mileage_km'])
        else:
            for column in ('latest_measured_tread_mm', 'historical_wear_mm_per_1000km',
                           'days_since_inspection', 'distance_since_inspection_km'):
                output[column] = float('nan')
        events = maintenance.loc[maintenance.tyre_id == tyre_id, ['timestamp']].sort_values('timestamp')
        if len(events):
            output = pd.merge_asof(output, events.rename(columns={'timestamp': 'maintenance_timestamp'}),
                                   left_on='timestamp', right_on='maintenance_timestamp', direction='backward')
            output['days_since_maintenance'] = (
                output.timestamp - output.maintenance_timestamp).dt.total_seconds() / 86400
            output = output.drop(columns='maintenance_timestamp')
        else:
            output['days_since_maintenance'] = float('nan')
        output['has_inspection'] = output.latest_measured_tread_mm.notna().astype(int)
        output['has_wear_history'] = output.historical_wear_mm_per_1000km.notna().astype(int)
        output['has_maintenance'] = output.days_since_maintenance.notna().astype(int)
        output[FEATURE_COLUMNS] = output[FEATURE_COLUMNS].astype(float)
        result.append(output[['truck_id', 'tyre_id', 'timestamp'] + FEATURE_COLUMNS])
    if not result:
        return pd.DataFrame(columns=['truck_id', 'tyre_id', 'timestamp'] + FEATURE_COLUMNS)
    return pd.concat(result, ignore_index=True)


def export_features(database=Path('data/tyremind.db'), output=Path('data/features')):
    database, output = Path(database), Path(output)
    # Read-only connection: a typo must not create an empty database.
    with closing(sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        connection.execute('BEGIN')  # Consistent snapshot across all four reads.
        frames = [pd.read_sql_query(f'SELECT * FROM {table}', connection)
                  for table in ('telemetry', 'tyres', 'inspections', 'maintenance_events')]
    features = build_features(*frames)
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / 'tyre_features.csv', index=False)
    report = {'feature_version': '1.0', 'rows': len(features),
              'tyres': int(features.tyre_id.nunique()), 'feature_columns': FEATURE_COLUMNS,
              'missing_by_feature': {column: int(features[column].isna().sum()) for column in FEATURE_COLUMNS},
              'imputation': 'none', 'window': '(timestamp - 7 days, timestamp], minimum 3 observations',
              'availability_assumption': 'Stored records are available at their event timestamps.',
              'target': 'Not created yet; no model trained.'}
    (output / 'feature_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--output', type=Path, default=Path('data/features'))
    args = parser.parse_args()
    print(json.dumps(export_features(args.database, args.output), indent=2))


if __name__ == '__main__':
    main()
