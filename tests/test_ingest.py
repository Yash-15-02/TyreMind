"""Operational ingestion integrity, quarantine, retry and transaction tests."""
import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from tyremind.data.generate import Config,write_dataset
from tyremind.data.preprocess import preprocess
from tyremind.database.load import load_database
from tyremind.monitoring.simulator import generate_next_batch
from tyremind.monitoring.ingest import ingest_batch,ingest_pending,run_ingestion_cycle


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.history=self.root/'history'
        self.database=self.root/'fleet.db'
        self.simulator=self.root/'simulator'
        write_dataset(Config(trucks=2,days=30),self.history)
        preprocess(self.history,self.root/'processed')
        load_database(self.root/'processed/clean',self.database)
        generate_next_batch(self.database,self.history,self.simulator)
        with sqlite3.connect(self.simulator/'simulator.db') as c:
            self.raw=json.loads(c.execute('SELECT payload FROM batches WHERE batch_id=1').fetchone()[0])

    def count(self,table):
        with sqlite3.connect(self.database) as c:
            return c.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]

    def test_pending_backlog_and_repeat_are_duplicate_free(self):
        generate_next_batch(self.database,self.history,self.simulator)
        first=ingest_pending(self.simulator,self.database)
        second=ingest_pending(self.simulator,self.database)
        self.assertEqual(first['telemetry_inserted'],24)
        self.assertEqual(second['telemetry_inserted'],0)
        self.assertEqual(second['processed_batches'],0)
        self.assertEqual(self.count('telemetry'),384)
        self.assertEqual(self.count('ingestion_batches'),2)
        with sqlite3.connect(self.database) as c:
            self.assertEqual(c.execute('PRAGMA foreign_key_check').fetchall(),[])
            self.assertNotIn('simulation_truth', [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")])

    def test_invalid_readings_quarantined_but_fault_values_retained(self):
        batch=copy.deepcopy(self.raw)
        batch['telemetry'][0]['pressure_kpa']='broken-sensor'
        batch['telemetry'][1]['tyre_id']='UNKNOWN'
        batch['telemetry'][2]['daily_distance_km']=-10
        batch['telemetry'][3]['pressure_kpa']=100
        batch['telemetry'][4]['temperature_c']=120
        report=ingest_batch(self.database,'test',batch)
        self.assertEqual(report['tables']['telemetry']['inserted_rows'],9)
        self.assertEqual(report['tables']['telemetry']['rejected_rows'],3)
        self.assertEqual(self.count('ingestion_rejections'),3)
        with sqlite3.connect(self.database) as c:
            self.assertEqual(c.execute('SELECT COUNT(*) FROM telemetry WHERE pressure_kpa=100').fetchone()[0],1)
            self.assertIn('broken-sensor', c.execute('SELECT original_json FROM ingestion_rejections WHERE source_row=2').fetchone()[0])

    def test_mileage_regression_checked_against_stored_history(self):
        batch=copy.deepcopy(self.raw)
        batch['telemetry'][0]['cumulative_mileage_km']=1
        report=ingest_batch(self.database,'test',batch)
        self.assertEqual(report['tables']['telemetry']['rejected_rows'],1)
        with sqlite3.connect(self.database) as c:
            self.assertEqual(c.execute('SELECT reasons FROM ingestion_rejections').fetchone()[0],'mileage_below_previous_record')

    def test_conflicting_existing_key_cannot_overwrite(self):
        ingest_batch(self.database,'original',self.raw)
        changed=copy.deepcopy(self.raw)
        changed['telemetry'][0]['pressure_kpa']+=10
        changed['batch_id']=2
        result=ingest_batch(self.database,'original',changed)
        self.assertEqual(result['tables']['telemetry']['inserted_rows'],0)
        self.assertEqual(result['tables']['telemetry']['unchanged_rows'],11)
        self.assertEqual(result['tables']['telemetry']['rejected_rows'],1)
        with self.assertRaisesRegex(ValueError,'content changed'):
            ingest_batch(self.database,'original',dict(self.raw,telemetry=changed['telemetry']))
        self.assertEqual(self.count('telemetry'),372)

    def test_schema_failure_rolls_back_earlier_telemetry_inserts(self):
        malformed=copy.deepcopy(self.raw)
        malformed['inspections']=[{'tyre_id':'TYR-0001'}]
        with self.assertRaisesRegex(ValueError,'missing required columns'):
            ingest_batch(self.database,'test',malformed)
        self.assertEqual(self.count('telemetry'),360)
        self.assertEqual(ingest_batch(self.database,'test',self.raw)['tables']['telemetry']['inserted_rows'],12)

    def test_duplicate_rows_quarantined_and_source_rows_preserved(self):
        batch=copy.deepcopy(self.raw)
        batch['telemetry'].append(copy.deepcopy(batch['telemetry'][0]))
        result=ingest_batch(self.database,'test',batch)
        self.assertEqual(result['tables']['telemetry']['inserted_rows'],12)
        self.assertEqual(result['tables']['telemetry']['rejected_rows'],1)
        with sqlite3.connect(self.database) as c:
            self.assertEqual(c.execute('SELECT source_row,reasons FROM ingestion_rejections').fetchone(),(14,'exact_duplicate'))

    def test_complete_cycle_drains_backlog_before_new_batch(self):
        result=run_ingestion_cycle(self.database,self.history,self.simulator)
        self.assertTrue(result['operational_database_updated'])
        self.assertEqual(result['ingested_batches'],2)
        self.assertEqual(result['telemetry_inserted'],24)
        self.assertEqual(self.count('telemetry'),384)

    def test_inspection_ingestion_and_out_of_order_mileage(self):
        # Store a later timestamp first, then accept a consistent reading between it and history.
        later=copy.deepcopy(self.raw)
        later['telemetry']=later['telemetry'][:1]
        later['telemetry'][0]['timestamp']='2026-02-02T00:00:00+00:00'
        later['telemetry'][0]['cumulative_mileage_km']+=500
        later['inspections']=[dict(inspection_id='LIVE-TEST',tyre_id='TYR-0001',
                                    timestamp='2026-02-02T00:00:00+00:00',tread_depth_mm=12)]
        ingest_batch(self.database,'later',later)
        middle=copy.deepcopy(self.raw)
        middle['telemetry']=middle['telemetry'][:1]
        result=ingest_batch(self.database,'middle',middle)
        self.assertEqual(result['tables']['telemetry']['inserted_rows'],1)
        bad=copy.deepcopy(middle)
        bad['telemetry'][0]['timestamp']='2026-02-01T00:00:00+00:00'
        bad['telemetry'][0]['cumulative_mileage_km']+=1000
        result=ingest_batch(self.database,'bad',bad)
        self.assertEqual(result['tables']['telemetry']['rejected_rows'],1)
        with sqlite3.connect(self.database) as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM inspections WHERE inspection_id='LIVE-TEST'").fetchone()[0],1)


if __name__=='__main__':
    unittest.main()
