"""Validate operational CSVs without imputing missing measurements."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# table: (required columns, unique key, numeric columns)
SCHEMAS = {
    'trucks': (['truck_id', 'vehicle_type'], ['truck_id'], []),
    'tyres': (['tyre_id', 'truck_id', 'position', 'installation_timestamp',
               'reference_pressure_kpa', 'initial_tread_depth_mm'], ['tyre_id'],
              ['reference_pressure_kpa', 'initial_tread_depth_mm']),
    'telemetry': (['truck_id', 'tyre_id', 'timestamp', 'pressure_kpa', 'temperature_c',
                   'daily_distance_km', 'cumulative_mileage_km', 'load_ratio'],
                  ['tyre_id', 'timestamp'], ['pressure_kpa', 'temperature_c',
                   'daily_distance_km', 'cumulative_mileage_km', 'load_ratio']),
    'inspections': (['inspection_id', 'tyre_id', 'timestamp', 'tread_depth_mm'],
                    ['inspection_id'], ['tread_depth_mm']),
    'maintenance_events': (['event_id', 'tyre_id', 'timestamp', 'event_type'],
                           ['event_id'], []),
}


def validate_table(name, raw, parents, *, include_source_rows=False):
    """Return clean rows, original rejected rows with reasons, and counts."""
    columns, key, numeric = SCHEMAS[name]
    missing = sorted(set(columns) - set(raw.columns))
    if missing:
        raise ValueError(f'{name}: missing required columns: {missing}')
    original = raw.reset_index(drop=True).copy()
    frame = original[columns].copy()
    reasons = [[] for _ in range(len(frame))]

    def flag(mask, reason):
        for index in frame.index[mask.fillna(True)]:
            reasons[index].append(reason)

    for column in columns:
        if column in numeric:
            frame[column] = pd.to_numeric(frame[column], errors='coerce')
            flag(frame[column].isna() | ~np.isfinite(frame[column]), f'invalid_numeric:{column}')
            # Temperature can legitimately be below zero; other physical quantities cannot.
            if column != 'temperature_c':
                flag(frame[column] < 0, f'negative_value:{column}')
            if column == 'reference_pressure_kpa':
                flag(frame[column] == 0, 'zero_reference_pressure')
        elif column in ('timestamp', 'installation_timestamp'):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors='coerce', format='mixed')
            flag(frame[column].isna(), f'invalid_timestamp:{column}')
        else:
            frame[column] = frame[column].astype('string').str.strip()
            flag(frame[column].isna() | frame[column].eq(''), f'missing_value:{column}')

    # Exact normalized duplicates keep the first occurrence; conflicting keys keep none.
    exact = frame.duplicated(keep='first')
    flag(exact, 'exact_duplicate')
    unique = frame.loc[~exact]
    conflict = unique.duplicated(key, keep=False)
    conflict_keys = pd.MultiIndex.from_frame(unique.loc[conflict, key])
    if len(conflict_keys):
        flag(pd.Series(pd.MultiIndex.from_frame(frame[key]).isin(conflict_keys)), 'conflicting_key')

    if name == 'tyres':
        flag(~frame.truck_id.isin(parents['trucks'].truck_id), 'unknown_truck')
        positions = {'FRONT_LEFT', 'FRONT_RIGHT', 'REAR_LEFT_OUTER', 'REAR_LEFT_INNER',
                     'REAR_RIGHT_INNER', 'REAR_RIGHT_OUTER'}
        flag(~frame.position.isin(positions), 'unknown_position')
        position_keys = pd.MultiIndex.from_frame(unique.loc[
            unique.duplicated(['truck_id', 'position'], keep=False), ['truck_id', 'position']])
        flag(pd.Series(pd.MultiIndex.from_frame(frame[['truck_id', 'position']]).isin(position_keys)),
             'conflicting_installed_position')
    elif name in ('telemetry', 'inspections', 'maintenance_events'):
        tyres = parents['tyres'].set_index('tyre_id')
        flag(~frame.tyre_id.isin(tyres.index), 'unknown_tyre')
        installation = frame.tyre_id.map(tyres.installation_timestamp)
        flag(frame.timestamp < installation, 'before_installation')
        if name == 'telemetry':
            flag(~frame.truck_id.isin(parents['trucks'].truck_id), 'unknown_truck')
            flag(frame.truck_id.ne(frame.tyre_id.map(tyres.truck_id)), 'tyre_truck_mismatch')
            # Running maximum catches continued regressions, not just the first drop.
            eligible = frame.loc[[not r for r in reasons]].sort_values(['tyre_id', 'timestamp'])
            previous_max = eligible.groupby('tyre_id').cumulative_mileage_km.transform(
                lambda series: series.cummax().shift())
            bad = (eligible.cumulative_mileage_km < previous_max).fillna(False).astype(bool)
            mask = pd.Series(False, index=frame.index)
            mask.loc[eligible.index] = bad
            flag(mask, 'mileage_regression')

    accepted = pd.Series([not r for r in reasons], index=frame.index, dtype=bool)
    clean = frame.loc[accepted].copy()
    sort_columns = [c for c in ('tyre_id', 'timestamp') if c in clean.columns]
    if sort_columns:
        clean = clean.sort_values(sort_columns)
    if include_source_rows:
        clean['_source_row'] = clean.index + 2
    rejected = original.loc[~accepted].copy()
    rejected.insert(0, 'source_row', rejected.index + 2)  # CSV header is row 1
    rejected['rejection_reasons'] = [';'.join(reasons[i]) for i in rejected.index]
    counts = Counter(reason for row in reasons for reason in row)
    report = {'input_rows': len(frame), 'accepted_rows': len(clean),
              'rejected_rows': len(rejected), 'reason_counts': dict(counts),
              'ignored_extra_columns': sorted(set(original.columns) - set(columns))}
    return clean.reset_index(drop=True), rejected.reset_index(drop=True), report


def preprocess(source: Path, output: Path):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or source.resolve() in output.resolve().parents:
        raise ValueError('Output must be separate from the input directory.')
    # Read and validate all tables before writing any results.
    raw = {name: pd.read_csv(source / f'{name}.csv', dtype='string') for name in SCHEMAS}
    clean, rejected, tables = {}, {}, {}
    for name in SCHEMAS:
        clean[name], rejected[name], tables[name] = validate_table(name, raw[name], clean)
    report = {'imputation': 'none', 'tables': tables,
              'evaluation_labels': 'Excluded: simulation truth is not operational input.',
              'reason_counts_note': 'A rejected row can have multiple reasons.'}
    for directory in ('clean', 'rejected'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    for name in SCHEMAS:
        clean[name].to_csv(output / 'clean' / f'{name}.csv', index=False)
        rejected[name].to_csv(output / 'rejected' / f'{name}.csv', index=False)
    (output / 'validation_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/generated'))
    parser.add_argument('--output', type=Path, default=Path('data/processed'))
    args = parser.parse_args()
    print(json.dumps(preprocess(args.input, args.output), indent=2))


if __name__ == '__main__':
    main()
