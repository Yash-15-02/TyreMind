"""Continue the historical fleet one simulated day per durable raw batch."""
import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from tyremind.data.generate import Config, generate


def bootstrap(source, history, seed):
    manifest = json.loads((history / 'manifest.json').read_text())
    config = Config(**manifest['config'])
    # Recreate hidden simulator parameters without changing historical files.
    generated = generate(config, include_state=True)
    with closing(sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        for name in ('trucks', 'tyres', 'telemetry'):
            actual = pd.read_sql_query(f'SELECT * FROM {name}', connection)
            expected = generated[name]
            columns = expected.columns.tolist()
            order = ['tyre_id', 'timestamp'] if name == 'telemetry' else [columns[0]]
            try:
                pd.testing.assert_frame_equal(actual[columns].sort_values(order).reset_index(drop=True),
                                              expected.sort_values(order).reset_index(drop=True),
                                              check_dtype=False, check_exact=False, rtol=1e-7, atol=1e-6)
            except AssertionError as error:
                raise ValueError(f'{name}: database does not match the historical generator manifest. Use matching source files.') from error
    labels = pd.read_csv(history / 'evaluation_labels.csv')
    expected = generated['evaluation_labels']
    try:
        pd.testing.assert_frame_equal(labels.sort_values(['tyre_id','timestamp']).reset_index(drop=True),
                                      expected.sort_values(['tyre_id','timestamp']).reset_index(drop=True),
                                      check_dtype=False, check_exact=False, rtol=1e-7, atol=1e-6)
    except AssertionError as error:
        raise ValueError('Historical simulation labels do not match the manifest.') from error
    rng = np.random.default_rng(seed)
    return {'version': 1, 'seed': seed, 'source': str(source.resolve()), 'history': str(history.resolve()),
            'historical_config': manifest['config'], 'next_day_index': config.days, 'batch_id': 0,
            'rng_state': rng.bit_generator.state,
            'tyres': json.loads(generated['simulator_state'].to_json(orient='records', double_precision=15)),
            'historical_labels_sha256': hashlib.sha256((history / 'evaluation_labels.csv').read_bytes()).hexdigest()}


def advance(state):
    rng = np.random.default_rng()
    rng.bit_generator.state = state['rng_state']
    day = state['next_day_index']
    stamp = (pd.Timestamp(state['historical_config']['start_date'], tz='UTC') + pd.Timedelta(days=day)).isoformat()
    truck_values = {}
    for truck in sorted({tyre['truck_id'] for tyre in state['tyres']}):
        truck_values[truck] = (float(np.clip(rng.normal(260,45),80,450)),
                               float(np.clip(rng.normal(.85,.07),.5,1.05)))
    telemetry, inspections, truth = [], [], []
    for tyre in state['tyres']:
        distance, load = truck_values[tyre['truck_id']]
        overloaded = tyre['truck_index'] % 4 == 3 and day >= state['historical_config']['days'] // 3
        load += .3 if overloaded else 0
        active = day >= tyre['onset']
        scenario = tyre['scenario']
        repaired = scenario == 'maintenance_intervention' and day >= tyre['service_day']
        leaking = active and (scenario == 'slow_leak' or (scenario == 'maintenance_intervention' and not repaired))
        loss = min(210, (day - tyre['onset'] + 1)*3) if leaking else 0
        pressure = tyre['reference_pressure_kpa'] - loss + rng.normal(0,5)
        temperature = (23 + 6*np.sin(day*2*np.pi/90) + 22 + 18*max(0,load-.85)
                       + loss*.035 + (22 if active and scenario == 'elevated_temperature' else 0) + rng.normal(0,2))
        fp = 1 + 2*max(0,(tyre['reference_pressure_kpa']-pressure)/tyre['reference_pressure_kpa'])
        fl = 1 + 1.5*max(0,load-1)
        ft = 1 + max(0,temperature-60)/80
        wear = tyre['base_wear']*distance/1000*fp*fl*ft*(2.3 if active and scenario == 'accelerated_wear' else 1)*rng.uniform(.95,1.05)
        tyre['tread_depth_mm'] = max(0,tyre['tread_depth_mm']-wear)
        tyre['mileage_km'] += distance
        telemetry.append(dict(truck_id=tyre['truck_id'],tyre_id=tyre['tyre_id'],timestamp=stamp,
                              pressure_kpa=round(float(pressure),3),temperature_c=round(float(temperature),3),
                              daily_distance_km=round(distance,3),cumulative_mileage_km=round(tyre['mileage_km'],3),
                              load_ratio=round(load,4)))
        if day % 14 == 0:
            measured = np.clip(tyre['tread_depth_mm']+rng.normal(0,.06),0,tyre['initial_tread_depth_mm'])
            inspections.append(dict(inspection_id=f"LIVE-INSP-{day:06d}-{tyre['tyre_id']}",
                                    tyre_id=tyre['tyre_id'],timestamp=stamp,tread_depth_mm=round(float(measured),3)))
        truth.append(dict(tyre_id=tyre['tyre_id'],timestamp=stamp,scenario_label=scenario,
                          scenario_active=bool(overloaded if scenario == 'overloading' else active and scenario != 'normal' and not repaired),
                          overload_active=bool(overloaded),simulated_true_tread_depth_mm=round(tyre['tread_depth_mm'],4)))
    state['next_day_index'] += 1
    state['batch_id'] += 1
    state['rng_state'] = rng.bit_generator.state
    return {'batch_id':state['batch_id'],'timestamp':stamp,'telemetry':telemetry,
            'inspections':inspections,'simulation_truth':truth}


def export_batch(batch, output):
    folder = output / 'batches' / f"batch-{batch['batch_id']:06d}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, columns in [('telemetry', ['truck_id','tyre_id','timestamp','pressure_kpa','temperature_c','daily_distance_km','cumulative_mileage_km','load_ratio']),
                          ('inspections', ['inspection_id','tyre_id','timestamp','tread_depth_mm']),
                          ('simulation_truth', ['tyre_id','timestamp','scenario_label','scenario_active','overload_active','simulated_true_tread_depth_mm'])]:
        temporary = folder / f'{name}.csv.tmp'
        pd.DataFrame(batch[name],columns=columns).to_csv(temporary,index=False)
        temporary.replace(folder / f'{name}.csv')
    return folder


def generate_next_batch(source=Path('data/tyremind.db'), history=Path('data/generated'),
                        output=Path('data/simulator'), seed=43):
    source, history, output = map(Path, (source,history,output))
    output.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(output / 'simulator.db',timeout=30)) as connection:
        connection.execute('BEGIN IMMEDIATE')
        try:
            connection.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)')
            connection.execute('CREATE TABLE IF NOT EXISTS batches (batch_id INTEGER PRIMARY KEY, payload TEXT NOT NULL, exported INTEGER NOT NULL DEFAULT 0)')
            saved = connection.execute('SELECT payload FROM state WHERE id=1').fetchone()
            state = json.loads(saved[0]) if saved else bootstrap(source,history,seed)
            if state['seed'] != seed or state['source'] != str(source.resolve()) or state['history'] != str(history.resolve()):
                raise ValueError('Existing simulator uses different source paths or seed. Use its original settings or a new output directory.')
            pending = connection.execute('SELECT payload FROM batches WHERE exported=0 ORDER BY batch_id LIMIT 1').fetchone()
            if pending:
                batch = json.loads(pending[0])
            else:
                batch = advance(state)
                connection.execute('INSERT INTO batches(batch_id,payload) VALUES (?,?)',
                                   (batch['batch_id'],json.dumps(batch,allow_nan=False)))
                connection.execute('INSERT INTO state(id,payload) VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                                   (json.dumps(state,allow_nan=False),))
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        # State and raw batch commit together. Retry exports this same batch if CSV writing fails.
        folder = export_batch(batch,output)
        with connection:
            connection.execute('UPDATE batches SET exported=1 WHERE batch_id=?',(batch['batch_id'],))
    return {'mode':'simulated_telemetry','batch_id':batch['batch_id'],'simulated_timestamp':batch['timestamp'],
            'readings_generated':len(batch['telemetry']),'inspections_generated':len(batch['inspections']),
            'raw_batch_directory':str(folder),'operational_database_updated':False,
            'message':'One further simulated day generated; operational validation and ingestion are not connected yet.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-database',type=Path,default=Path('data/tyremind.db'))
    parser.add_argument('--history',type=Path,default=Path('data/generated'))
    parser.add_argument('--output',type=Path,default=Path('data/simulator'))
    parser.add_argument('--seed',type=int,default=43)
    args=parser.parse_args()
    print(json.dumps(generate_next_batch(args.source_database,args.history,args.output,args.seed),indent=2))


if __name__ == '__main__':
    main()
