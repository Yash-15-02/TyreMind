"""Run persisted wear/anomaly models on current features without retraining."""
import argparse
import hashlib
import io
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tyremind.features.build import FEATURE_COLUMNS
from tyremind.ml.anomaly import ANOMALY_FEATURES, predict_anomalies
from tyremind.monitoring.features import FEATURE_VERSION


class SavedModels:
    def __init__(self,wear_model='random_forest',wear_artifact=None,
                 anomaly_artifact=Path('models/anomaly/isolation_forest.joblib')):
        if wear_model not in ('random_forest','xgboost'):
            raise ValueError('Choose random_forest or xgboost for wear prediction.')
        wear_artifact=Path(wear_artifact) if wear_artifact is not None else Path(f'models/wear/{wear_model}.joblib')
        anomaly_artifact=Path(anomaly_artifact)
        self.paths={'wear':str(wear_artifact),'anomaly':str(anomaly_artifact)}
        self.hashes={}
        artifacts={}
        for name,path in (('wear',wear_artifact),('anomaly',anomaly_artifact)):
            if not path.is_file():
                raise FileNotFoundError(f'Saved model missing: {path}. Complete local model training first.')
            payload=path.read_bytes()
            self.hashes[name]=hashlib.sha256(payload).hexdigest()
            artifacts[name]=joblib.load(io.BytesIO(payload))
        self.wear,self.anomaly=artifacts['wear'],artifacts['anomaly']
        self.wear_model=wear_model
        if self.wear['model_family']!=wear_model or self.wear['feature_columns']!=FEATURE_COLUMNS or self.wear['horizon_km']!=5000:
            raise ValueError('Wear artifact schema/model/horizon mismatch.')
        if self.anomaly['model_family']!='isolation_forest' or self.anomaly['feature_columns']!=ANOMALY_FEATURES:
            raise ValueError('Anomaly artifact schema/model mismatch.')
        if not np.isfinite(self.anomaly['threshold']):
            raise ValueError('Invalid saved anomaly threshold.')

    def score(self,frame):
        features=frame[FEATURE_COLUMNS].apply(pd.to_numeric,errors='raise')
        if np.isinf(features.to_numpy(dtype=float)).any():
            raise ValueError('Infinite input features cannot be scored.')
        if frame.feature_version.astype(str).ne(FEATURE_VERSION).any():
            raise ValueError('Current feature version differs from saved feature definitions.')
        cutoff=pd.to_datetime(frame.feature_as_of,utc=True)
        timestamp=pd.to_datetime(frame.timestamp,utc=True)
        if timestamp.isna().any() or cutoff.isna().any() or (timestamp>cutoff).any():
            raise ValueError('Invalid feature timestamps.')
        stale=frame.is_stale.ne(0) | timestamp.lt(cutoff)
        warmup=int(self.anomaly.get('warmup_min_observations_7d',3))
        insufficient=features.observations_7d.isna() | features.observations_7d.lt(max(3,warmup))
        ready=~stale & ~insufficient
        result=frame[['truck_id','tyre_id','timestamp','feature_as_of','feature_version']].copy()
        result['status']='scored'
        result.loc[insufficient,'status']='insufficient_history'
        result.loc[stale,'status']='stale_features'
        result['predicted_tread_depth_5000km_mm']=np.nan
        result['anomaly_score']=np.nan
        result['anomaly_flag']=pd.Series(pd.NA,index=result.index,dtype='Int64')
        if ready.any():
            wear=np.asarray(self.wear['pipeline'].predict(features.loc[ready]),dtype=float)
            scores,flags=predict_anomalies(self.anomaly,features.loc[ready])
            if not np.isfinite(wear).all() or not np.isfinite(scores).all():
                raise ValueError('Saved model returned nonfinite outputs.')
            result.loc[ready,'predicted_tread_depth_5000km_mm']=wear
            result.loc[ready,'anomaly_score']=scores
            result.loc[ready,'anomaly_flag']=flags.astype(int)
        return result,features


def run_saved_models(database=Path('data/tyremind.db'),output=Path('data/monitoring/predictions'),models=None,
                     wear_model='random_forest',wear_artifact=None,anomaly_artifact=Path('models/anomaly/isolation_forest.joblib')):
    models=models if models is not None else SavedModels(wear_model,wear_artifact,anomaly_artifact)
    database,output=Path(database),Path(output)
    if not database.is_file():
        raise FileNotFoundError('Operational database missing.')
    with closing(sqlite3.connect(database,timeout=30)) as connection:
        connection.execute('PRAGMA foreign_keys=ON')
        connection.execute('BEGIN IMMEDIATE')
        try:
            current=pd.read_sql_query('SELECT * FROM current_features ORDER BY tyre_id',connection)
            if current.empty:
                raise ValueError('No current feature rows; refresh features first.')
            latest=connection.execute('SELECT MAX(timestamp) FROM telemetry').fetchone()[0]
            if latest and pd.to_datetime(current.feature_as_of,utc=True).min()<pd.to_datetime(latest,utc=True):
                raise ValueError('Features are older than stored telemetry; refresh current features first.')
            scored,features=models.score(current)
            connection.execute('''CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY NOT NULL,
                truck_id TEXT NOT NULL REFERENCES trucks(truck_id),
                tyre_id TEXT NOT NULL REFERENCES tyres(tyre_id),timestamp TEXT NOT NULL,
                feature_as_of TEXT NOT NULL,feature_version TEXT NOT NULL,input_features_json TEXT NOT NULL,
                wear_model TEXT NOT NULL,wear_model_sha256 TEXT NOT NULL,anomaly_model_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,predicted_tread_depth_5000km_mm REAL,anomaly_score REAL,
                anomaly_flag INTEGER CHECK(anomaly_flag IN (0,1)),anomaly_threshold REAL NOT NULL,
                predicted_at TEXT NOT NULL,
                FOREIGN KEY(tyre_id,truck_id) REFERENCES tyres(tyre_id,truck_id))''')
            connection.execute('CREATE INDEX IF NOT EXISTS idx_predictions_tyre_time ON predictions(tyre_id,timestamp)')
            predicted_at=datetime.now(timezone.utc).isoformat()
            inserts=0
            identifiers=[]
            for index,row in scored.iterrows():
                inputs={name:None if pd.isna(value) else float(value) for name,value in features.loc[index].items()}
                inputs_json=json.dumps(inputs,sort_keys=True,allow_nan=False)
                identity={'tyre_id':row.tyre_id,'timestamp':row.timestamp,'as_of':row.feature_as_of,
                          'feature_version':row.feature_version,'inputs':inputs,'status':row.status,
                          'wear_model_sha256':models.hashes['wear'],'anomaly_model_sha256':models.hashes['anomaly']}
                identifier=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
                identifiers.append(identifier)
                values=(identifier,row.truck_id,row.tyre_id,row.timestamp,row.feature_as_of,row.feature_version,inputs_json,
                        models.wear_model,models.hashes['wear'],models.hashes['anomaly'],row.status,
                        None if pd.isna(row.predicted_tread_depth_5000km_mm) else float(row.predicted_tread_depth_5000km_mm),
                        None if pd.isna(row.anomaly_score) else float(row.anomaly_score),
                        None if pd.isna(row.anomaly_flag) else int(row.anomaly_flag),float(models.anomaly['threshold']),predicted_at)
                cursor=connection.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(prediction_id) DO NOTHING',values)
                inserts+=cursor.rowcount
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    scored['prediction_id']=identifiers
    scored['wear_model']=models.wear_model
    scored['anomaly_threshold']=float(models.anomaly['threshold'])
    scored['wear_model_sha256']=models.hashes['wear']
    scored['anomaly_model_sha256']=models.hashes['anomaly']
    output.mkdir(parents=True,exist_ok=True)
    temporary=output/'latest_predictions.csv.tmp'
    scored.to_csv(temporary,index=False)
    temporary.replace(output/'latest_predictions.csv')
    report={'rows':len(scored),'scored_rows':int(scored.status.eq('scored').sum()),
            'skipped_rows':int(scored.status.ne('scored').sum()),'status_counts':scored.status.value_counts().to_dict(),
            'anomaly_flags':int(scored.anomaly_flag.fillna(0).sum()),'inserted_prediction_rows':inserts,
            'wear_model':models.wear_model,'wear_horizon_km':5000,'anomaly_model':'isolation_forest',
            'anomaly_threshold':float(models.anomaly['threshold']),'model_hashes':models.hashes,
            'feature_version':FEATURE_VERSION,'predicted_at':predicted_at,'retrained':False,
            'csv_path':str(output/'latest_predictions.csv'),
            'csv_sha256':hashlib.sha256((output/'latest_predictions.csv').read_bytes()).hexdigest(),
            'note':'Synthetic-data inference. Anomaly flags are not diagnoses or risk levels; held-out recall is low.'}
    temporary=output/'prediction_report.json.tmp'
    temporary.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    temporary.replace(output/'prediction_report.json')
    return report


def run_prediction_cycle(source,history,simulator,seed,feature_output,prediction_output,models):
    from tyremind.monitoring.features import run_feature_cycle
    cycle=run_feature_cycle(source,history,simulator,seed,feature_output)
    prediction=run_saved_models(source,prediction_output,models=models)
    return {**cycle,'mode':'ml_predictions_ready','wear_model':prediction['wear_model'],
            'prediction_rows':prediction['scored_rows'],'skipped_prediction_rows':prediction['skipped_rows'],
            'anomaly_flags':prediction['anomaly_flags'],'prediction_csv':prediction['csv_path'],
            'message':'Saved wear and anomaly models ran; risk decisions and investigation events are not connected yet.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path('data/tyremind.db'))
    parser.add_argument('--output',type=Path,default=Path('data/monitoring/predictions'))
    parser.add_argument('--wear-model',choices=['random_forest','xgboost'],default='random_forest')
    parser.add_argument('--wear-artifact',type=Path)
    parser.add_argument('--anomaly-artifact',type=Path,default=Path('models/anomaly/isolation_forest.joblib'))
    args=parser.parse_args()
    print(json.dumps(run_saved_models(args.database,args.output,wear_model=args.wear_model,
                                     wear_artifact=args.wear_artifact,anomaly_artifact=args.anomaly_artifact),indent=2))


if __name__=='__main__':
    main()
