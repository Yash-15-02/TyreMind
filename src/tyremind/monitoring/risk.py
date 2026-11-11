"""Deterministic, evidence-preserving demo risk classification after inference."""
import argparse
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tyremind.features.build import FEATURE_COLUMNS

LEVELS={'Normal':0,'Medium':1,'High':2,'Critical':3}


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (ValueError,TypeError):
        return False


def load_policy(path=Path('config/risk_rules.json')):
    policy=json.loads(Path(path).read_text())
    numeric=['pressure_medium_deviation','pressure_high_deviation','pressure_critical_deviation',
             'temperature_medium_c','temperature_high_c','temperature_critical_c',
             'observed_tread_high_mm','observed_tread_critical_mm','predicted_tread_high_mm',
             'max_inspection_age_days','pressure_decline_steps','pressure_decline_total_kpa',
             'max_trend_span_days','load_medium_ratio']
    for name in numeric:
        if isinstance(policy.get(name),bool) or not isinstance(policy.get(name),(int,float)) or not finite(policy.get(name)) or policy[name]<=0:
            raise ValueError(f'Invalid risk configuration: {name}')
    if not (policy['pressure_medium_deviation']<policy['pressure_high_deviation']<policy['pressure_critical_deviation']<1):
        raise ValueError('Pressure deviations must increase from Medium to Critical and remain below 1.')
    if not policy['temperature_medium_c']<policy['temperature_high_c']<policy['temperature_critical_c']:
        raise ValueError('Temperature limits must increase.')
    if policy['observed_tread_critical_mm']>=policy['observed_tread_high_mm']:
        raise ValueError('Critical tread depth must be below High tread depth.')
    if not isinstance(policy['pressure_decline_steps'],int) or policy['pressure_decline_steps']<2:
        raise ValueError('Pressure decline requires an integer number of at least two steps.')
    for field in ('version','notice'):
        if not isinstance(policy.get(field),str) or not policy[field]:
            raise ValueError(f'Risk policy requires {field}.')
    return policy


def assess_risk(evidence,policy):
    rules,issues=[],[]
    def trigger(rule_id,level,values):
        rules.append({'rule_id':rule_id,'level':level,'values':values})
    if evidence.get('is_stale'):
        return {'risk_level':None,'data_status':'stale','rules':[],
                'data_issues':['Latest valid reading is older than the assessment cutoff.']}
    pressure,reference=evidence.get('pressure_kpa'),evidence.get('reference_pressure_kpa')
    if finite(pressure) and finite(reference) and reference>0:
        deviation=float(pressure)/float(reference)-1
        loss=-deviation
        for level,key in [('Critical','pressure_critical_deviation'),('High','pressure_high_deviation'),('Medium','pressure_medium_deviation')]:
            if loss>=policy[key]-1e-12:
                trigger('pressure_deviation',level,dict(pressure_kpa=pressure,reference_pressure_kpa=reference,
                                                       deviation_ratio=deviation,threshold_ratio=policy[key]))
                break
    else:
        issues.append('Missing or invalid pressure/reference pressure.')
    temperature=evidence.get('temperature_c')
    if finite(temperature):
        for level,key in [('Critical','temperature_critical_c'),('High','temperature_high_c'),('Medium','temperature_medium_c')]:
            if temperature>=policy[key]:
                trigger('temperature_elevated',level,dict(temperature_c=temperature,threshold_c=policy[key]))
                break
    else:
        issues.append('Missing temperature.')
    history=evidence.get('pressure_history',[])
    needed=policy['pressure_decline_steps']+1
    if len(history)>=needed:
        recent=history[-needed:]
        timestamps=pd.to_datetime([row['timestamp'] for row in recent],utc=True)
        pressures=[row['pressure_kpa'] for row in recent]
        span=(timestamps[-1]-timestamps[0]).total_seconds()/86400
        if all(finite(value) for value in pressures) and 0<span<=policy['max_trend_span_days'] and timestamps.is_monotonic_increasing and timestamps.is_unique:
            total=pressures[0]-pressures[-1]
            if all(a>b for a,b in zip(pressures[:-1],pressures[1:])) and total>=policy['pressure_decline_total_kpa']:
                trigger('sustained_pressure_decline','High',dict(total_drop_kpa=total,steps=needed-1,
                        span_days=span,required_drop_kpa=policy['pressure_decline_total_kpa']))
        else:
            issues.append('Pressure trend history is invalid or too widely spaced.')
    else:
        issues.append('Insufficient pressure trend history.')
    tread,age=evidence.get('latest_measured_tread_mm'),evidence.get('days_since_inspection')
    if finite(tread) and tread>=0 and finite(age) and 0<=age<=policy['max_inspection_age_days']:
        if tread<=policy['observed_tread_critical_mm']:
            trigger('observed_tread_low','Critical',dict(measured_tread_mm=tread,inspection_age_days=age,
                                                        threshold_mm=policy['observed_tread_critical_mm']))
        elif tread<=policy['observed_tread_high_mm']:
            trigger('observed_tread_low','High',dict(measured_tread_mm=tread,inspection_age_days=age,
                                                    threshold_mm=policy['observed_tread_high_mm']))
    else:
        issues.append('Tread inspection is missing, invalid or older than the configured inspection age.')
    load=evidence.get('load_mean_7d')
    if finite(load):
        if load>policy['load_medium_ratio']:
            trigger('load_elevated','Medium',dict(load_mean_7d=load,threshold_ratio=policy['load_medium_ratio']))
    else:
        issues.append('Missing recent load history.')
    prediction=evidence.get('prediction')
    if prediction and prediction.get('status')=='scored':
        future=prediction.get('predicted_tread_depth_5000km_mm')
        if finite(future) and future>=0:
            if future<=policy['predicted_tread_high_mm']:
                trigger('predicted_tread_low','High',dict(predicted_tread_depth_5000km_mm=future,
                                                         threshold_mm=policy['predicted_tread_high_mm']))
        else:
            issues.append('Missing/invalid future tread prediction.')
        flag=prediction.get('anomaly_flag')
        if flag in (0,1) and finite(prediction.get('anomaly_score')) and finite(prediction.get('anomaly_threshold')):
            if flag==1:
                trigger('ml_anomaly','Medium',dict(anomaly_score=prediction['anomaly_score'],
                                                  threshold=prediction['anomaly_threshold']))
                concerning={'pressure_deviation','temperature_elevated','load_elevated'}
                if any(rule['rule_id'] in concerning for rule in rules):
                    trigger('anomaly_with_operating_condition','High',{'anomaly_flag':1})
        else:
            issues.append('Missing anomaly result.')
    else:
        issues.append('No matching scored prediction for the current feature snapshot.')
    level=max((rule['level'] for rule in rules),key=LEVELS.get,default=None)
    if level is None and not issues:
        level='Normal'
    return {'risk_level':level,'data_status':('partial' if level else 'insufficient_data') if issues else 'complete',
            'rules':rules,'data_issues':issues}


def monitoring_decision(assessment):
    """Record continuation explicitly; event dispatch remains a later step."""
    level=assessment['risk_level']
    if level in ('Normal','Medium'):
        action='CONTINUE_MONITORING'
        reason=f'{level} risk: results saved; continue scheduled monitoring without an investigation event.'
    elif level in ('High','Critical'):
        action='AWAIT_EVENT_HANDLER'
        reason=f'{level} risk: retain results for the next event-handling step; no event is created here.'
    else:
        action='DATA_REVIEW_REQUIRED'
        reason='Risk is unclassified because evidence is incomplete/stale; retain issues and continue collecting readings.'
    return {'next_action':action,'monitoring_continues':1,
            'data_review_required':int(assessment['data_status']!='complete'),
            'decision_reason':reason}


def canonical_inputs(row):
    return {name:float(row[name]) if finite(row[name]) else None for name in FEATURE_COLUMNS}


def evaluate_risks(database=Path('data/tyremind.db'),config=Path('config/risk_rules.json'),
                   output=Path('data/monitoring/risk'),wear_model='random_forest',model_hashes=None):
    policy=load_policy(config)
    policy_json=json.dumps(policy,sort_keys=True,allow_nan=False)
    policy_hash=hashlib.sha256(policy_json.encode()).hexdigest()
    database,output=Path(database),Path(output)
    if not database.is_file():
        raise FileNotFoundError('Operational database missing.')
    with closing(sqlite3.connect(database,timeout=30)) as connection:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('BEGIN IMMEDIATE')
        try:
            features=pd.read_sql_query('SELECT * FROM current_features ORDER BY tyre_id',connection)
            if features.empty:
                raise ValueError('Current features missing; refresh features first.')
            latest=connection.execute('SELECT MAX(timestamp) FROM telemetry').fetchone()[0]
            if pd.to_datetime(features.feature_as_of,utc=True).min()<pd.to_datetime(latest,utc=True):
                raise ValueError('Feature snapshot is older than stored telemetry; refresh first.')
            tyres={row[0]:row[1] for row in connection.execute('SELECT tyre_id,reference_pressure_kpa FROM tyres')}
            raw=pd.read_sql_query('SELECT tyre_id,timestamp,pressure_kpa,temperature_c FROM telemetry',connection)
            raw['parsed_timestamp']=pd.to_datetime(raw.timestamp,utc=True)
            predictions=[]
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone():
                prediction_frame=pd.read_sql_query('SELECT * FROM predictions WHERE wear_model=? ORDER BY predicted_at DESC,prediction_id DESC',connection,params=[wear_model])
                predictions=prediction_frame.astype(object).where(prediction_frame.notna(),None).to_dict('records')
            connection.execute('''CREATE TABLE IF NOT EXISTS risk_assessments (
                assessment_id TEXT PRIMARY KEY NOT NULL,tyre_id TEXT NOT NULL REFERENCES tyres(tyre_id),
                truck_id TEXT NOT NULL REFERENCES trucks(truck_id),timestamp TEXT NOT NULL,feature_as_of TEXT NOT NULL,
                risk_level TEXT CHECK(risk_level IN ('Normal','Medium','High','Critical')),
                data_status TEXT NOT NULL,rules_json TEXT NOT NULL,data_issues_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,policy_version TEXT NOT NULL,policy_sha256 TEXT NOT NULL,
                policy_json TEXT NOT NULL,prediction_id TEXT,assessed_at TEXT NOT NULL)''')
            connection.execute('CREATE INDEX IF NOT EXISTS idx_risk_tyre_time ON risk_assessments(tyre_id,timestamp)')
            connection.execute('''CREATE TABLE IF NOT EXISTS monitoring_decisions (
                assessment_id TEXT PRIMARY KEY NOT NULL REFERENCES risk_assessments(assessment_id),
                next_action TEXT NOT NULL CHECK(next_action IN ('CONTINUE_MONITORING','AWAIT_EVENT_HANDLER','DATA_REVIEW_REQUIRED')),
                monitoring_continues INTEGER NOT NULL CHECK(monitoring_continues=1),
                data_review_required INTEGER NOT NULL CHECK(data_review_required IN (0,1)),
                reason TEXT NOT NULL,recorded_at TEXT NOT NULL)''')
            rows,inserts,decision_inserts=[],0,0
            for _,feature in features.iterrows():
                history=raw.loc[(raw.tyre_id==feature.tyre_id)&(raw.parsed_timestamp<=pd.to_datetime(feature.timestamp,utc=True))].sort_values('parsed_timestamp')
                if history.empty:
                    raise ValueError('Feature row lacks source telemetry.')
                latest_reading=history.iloc[-1]
                inputs=canonical_inputs(feature)
                prediction=next((p for p in predictions if p['tyre_id']==feature.tyre_id and p['timestamp']==feature.timestamp
                                 and p['feature_as_of']==feature.feature_as_of and p['feature_version']==feature.feature_version
                                 and json.loads(p['input_features_json'])==inputs
                                 and (model_hashes is None or (p['wear_model_sha256']==model_hashes['wear'] and p['anomaly_model_sha256']==model_hashes['anomaly']))),None)
                evidence={'truck_id':feature.truck_id,'tyre_id':feature.tyre_id,'timestamp':feature.timestamp,
                          'feature_as_of':feature.feature_as_of,'is_stale':bool(feature.is_stale or pd.to_datetime(feature.timestamp,utc=True)<pd.to_datetime(feature.feature_as_of,utc=True)),
                          'pressure_kpa':float(latest_reading.pressure_kpa),'reference_pressure_kpa':tyres[feature.tyre_id],
                          'temperature_c':float(latest_reading.temperature_c),
                          'pressure_history':history.tail(policy['pressure_decline_steps']+1)[['timestamp','pressure_kpa']].to_dict('records'),
                          'latest_measured_tread_mm':inputs['latest_measured_tread_mm'],'days_since_inspection':inputs['days_since_inspection'],
                          'load_mean_7d':inputs['load_mean_7d'],
                          'prediction':prediction,'feature_values':inputs}
                assessment=assess_risk(evidence,policy)
                evidence_json=json.dumps(evidence,sort_keys=True,allow_nan=False)
                identity=hashlib.sha256((policy_hash+evidence_json).encode()).hexdigest()
                now=datetime.now(timezone.utc).isoformat()
                values=(identity,feature.tyre_id,feature.truck_id,feature.timestamp,feature.feature_as_of,
                        assessment['risk_level'],assessment['data_status'],json.dumps(assessment['rules']),
                        json.dumps(assessment['data_issues']),evidence_json,policy['version'],policy_hash,policy_json,
                        prediction['prediction_id'] if prediction else None,now)
                cursor=connection.execute('INSERT INTO risk_assessments VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(assessment_id) DO NOTHING',values)
                inserts+=cursor.rowcount
                decision=monitoring_decision(assessment)
                cursor=connection.execute('INSERT INTO monitoring_decisions VALUES (?,?,?,?,?,?) ON CONFLICT(assessment_id) DO NOTHING',
                    (identity,decision['next_action'],decision['monitoring_continues'],decision['data_review_required'],decision['decision_reason'],now))
                decision_inserts+=cursor.rowcount
                rows.append(dict(assessment_id=identity,tyre_id=feature.tyre_id,truck_id=feature.truck_id,
                                 timestamp=feature.timestamp,feature_as_of=feature.feature_as_of,
                                 risk_level=assessment['risk_level'],data_status=assessment['data_status'],
                                 rules_json=json.dumps(assessment['rules']),data_issues_json=json.dumps(assessment['data_issues']),
                                 policy_version=policy['version'],policy_sha256=policy_hash,
                                 prediction_id=prediction['prediction_id'] if prediction else None,**decision))
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    temporary=output/'latest_risk.csv.tmp'
    frame.to_csv(temporary,index=False)
    temporary.replace(output/'latest_risk.csv')
    report={'rows':len(rows),'risk_counts':{level:int(frame.risk_level.eq(level).sum()) for level in LEVELS},
            'unclassified_rows':int(frame.risk_level.isna().sum()),'data_status_counts':frame.data_status.value_counts().to_dict(),
            'inserted_assessments':inserts,'policy_version':policy['version'],'policy_sha256':policy_hash,
            'wear_model':wear_model,'csv_path':str(output/'latest_risk.csv'),
            'notice':policy['notice'],'events_created':False,
            'decision_counts':frame.next_action.value_counts().to_dict(),
            'inserted_decisions':decision_inserts,
            'continue_monitoring_rows':int(frame.next_action.eq('CONTINUE_MONITORING').sum()),
            'awaiting_event_handler_rows':int(frame.next_action.eq('AWAIT_EVENT_HANDLER').sum()),
            'data_review_rows':int(frame.data_review_required.sum()),'monitoring_continues':True}
    temporary=output/'risk_report.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n')
    temporary.replace(output/'risk_report.json')
    return report


def run_risk_cycle(source,history,simulator,seed,feature_output,prediction_output,models,config,output):
    from tyremind.monitoring.predict import run_prediction_cycle
    cycle=run_prediction_cycle(source,history,simulator,seed,feature_output,prediction_output,models)
    risk=evaluate_risks(source,config,output,models.wear_model,models.hashes)
    return {**cycle,'mode':'risk_levels_ready','risk_counts':risk['risk_counts'],
            'unclassified_risk_rows':risk['unclassified_rows'],'risk_csv':risk['csv_path'],
            'continue_monitoring_rows':risk['continue_monitoring_rows'],
            'awaiting_event_handler_rows':risk['awaiting_event_handler_rows'],
            'data_review_rows':risk['data_review_rows'],
            'message':'Results saved; Normal/Medium continue monitoring. High/Critical await the future event handler.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('data/tyremind.db'))
    parser.add_argument('--config',type=Path,default=Path('config/risk_rules.json'))
    parser.add_argument('--output',type=Path,default=Path('data/monitoring/risk'))
    parser.add_argument('--wear-model',choices=['random_forest','xgboost'],default='random_forest')
    args=parser.parse_args()
    print(json.dumps(evaluate_risks(args.database,args.config,args.output,args.wear_model),indent=2))


if __name__=='__main__':
    main()
