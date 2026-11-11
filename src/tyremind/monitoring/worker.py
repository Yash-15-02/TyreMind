"""Schedule fleet telemetry, model inference and deterministic risk assessment."""
import argparse
import json
import logging
import math
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.base import SchedulerNotRunningError


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def configure_logging(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('tyremind.monitoring')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter('%(asctime)sZ %(levelname)s %(message)s')
    formatter.converter = time.gmtime
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logfile = RotatingFileHandler(directory / 'worker.log', maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    logfile.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(logfile)
    return logger


def run_monitoring_cycle():
    """Step 1 heartbeat only. Later steps replace this with the fleet pipeline."""
    return {'mode': 'scheduler_heartbeat', 'tyres_processed': 0,
            'message': 'Cycle triggered successfully; telemetry and inference are not connected yet.'}


class MonitoringWorker:
    def __init__(self, interval_seconds=60, directory=Path('data/monitoring'), cycle=None, max_cycles=None):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError('Interval must be positive and finite.')
        if max_cycles is not None and max_cycles < 1:
            raise ValueError('max_cycles must be positive.')
        self.interval_seconds = interval_seconds
        self.directory = Path(directory)
        self.logger = configure_logging(self.directory)
        self.cycle = cycle if cycle is not None else run_monitoring_cycle
        self.max_cycles = max_cycles
        self.scheduler = BlockingScheduler(timezone='UTC')
        self.lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.run_id = uuid.uuid4().hex
        self.status = {'run_id': self.run_id, 'pid': os.getpid(), 'state': 'created',
                       'interval_seconds': interval_seconds, 'started_at': utc_now(),
                       'cycles_started': 0, 'cycles_succeeded': 0, 'cycles_failed': 0}

    def close_logging(self):
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def write_status(self, **changes):
        self.status.update(changes)
        self.status['updated_at'] = utc_now()
        # Caller holds lock while scheduled job runs; stop waits for it to finish.
        temporary = self.directory / f'worker_status.{self.run_id}.tmp'
        temporary.write_text(json.dumps(self.status, indent=2) + '\n', encoding='utf-8')
        temporary.replace(self.directory / 'worker_status.json')

    def execute_cycle(self):
        if self.stop_requested.is_set():
            return
        if not self.lock.acquire(blocking=False):
            self.logger.warning('Overlapping cycle skipped')
            return
        started = time.monotonic()
        try:
            self.status['cycles_started'] += 1
            count = self.status['cycles_started']
            self.write_status(state='cycle_running', last_cycle_started_at=utc_now())
            self.logger.info('Cycle %s started', count)
            try:
                result = self.cycle()
                # Verify result can be persisted before marking the cycle successful.
                json.dumps(result, allow_nan=False)
            except Exception as error:
                self.status['cycles_failed'] += 1
                self.write_status(state='waiting', last_error=str(error), last_cycle_status='failed',
                                  last_cycle_finished_at=utc_now(), last_duration_seconds=time.monotonic() - started)
                self.logger.exception('Cycle %s failed; the next scheduled cycle remains enabled', count)
                raise  # APScheduler records the failed job and continues scheduling.
            else:
                self.status['cycles_succeeded'] += 1
                self.write_status(state='waiting', last_error=None, last_cycle_status='completed',
                                  last_cycle_finished_at=utc_now(), last_duration_seconds=time.monotonic() - started,
                                  last_result=result)
                self.logger.info('Cycle %s completed in %.3fs: %s', count, time.monotonic() - started, result)
        finally:
            self.lock.release()
            if self.max_cycles is not None and self.status['cycles_started'] >= self.max_cycles:
                self.stop_requested.set()
                try:
                    self.scheduler.shutdown(wait=False)
                except SchedulerNotRunningError:
                    pass

    def request_stop(self, signum=None, frame=None):
        self.logger.info('Stop requested; waiting for any running cycle to finish')
        self.stop_requested.set()
        try:
            self.scheduler.shutdown(wait=True)
        except SchedulerNotRunningError:
            pass

    def start(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Start the worker from the main thread.')
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for sig in handlers:
                signal.signal(sig, self.request_stop)
            self.scheduler.add_job(self.execute_cycle, 'interval', seconds=self.interval_seconds,
                                   id='fleet_monitoring', max_instances=1, coalesce=True, misfire_grace_time=30)
            self.write_status(state='waiting')
            self.logger.info('Worker started: interval=%ss; first cycle runs after one interval. Press Ctrl+C to stop.',
                             self.interval_seconds)
            self.scheduler.start()
        finally:
            self.request_stop()
            # max_cycles may shut down the scheduler from its job; wait for that job's final writes.
            with self.lock:
                self.write_status(state='stopped', stopped_at=utc_now())
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            self.logger.info('Worker stopped: %s succeeded, %s failed',
                             self.status['cycles_succeeded'], self.status['cycles_failed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval-seconds', type=float, default=60)
    parser.add_argument('--output', type=Path, default=Path('data/monitoring'))
    parser.add_argument('--run-once', action='store_true', help='Check one cycle immediately, then exit.')
    parser.add_argument('--max-cycles', type=int, help='Stop automatically after this many cycle attempts.')
    parser.add_argument('--heartbeat-only', action='store_true', help='Run the Step 1 timer check without generating readings.')
    parser.add_argument('--source-database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--history', type=Path, default=Path('data/generated'))
    parser.add_argument('--simulator-output', type=Path, default=Path('data/simulator'))
    parser.add_argument('--simulation-seed', type=int, default=43)
    parser.add_argument('--feature-output', type=Path, default=Path('data/monitoring/features'))
    parser.add_argument('--features-only', action='store_true', help='Stop the cycle after feature calculation, without loading models.')
    parser.add_argument('--wear-model', choices=['random_forest','xgboost'], default='random_forest')
    parser.add_argument('--wear-artifact', type=Path)
    parser.add_argument('--anomaly-artifact', type=Path, default=Path('models/anomaly/isolation_forest.joblib'))
    parser.add_argument('--prediction-output', type=Path, default=Path('data/monitoring/predictions'))
    parser.add_argument('--prediction-only', action='store_true', help='Stop after saved-model inference, before risk rules.')
    parser.add_argument('--risk-config', type=Path, default=Path('config/risk_rules.json'))
    parser.add_argument('--risk-output', type=Path, default=Path('data/monitoring/risk'))
    parser.add_argument('--risk-only', action='store_true', help='Stop after risk assessment, before event routing.')
    parser.add_argument('--event-output', type=Path, default=Path('data/monitoring/events'))
    args = parser.parse_args()
    cycle = None
    if not args.heartbeat_only:
        if args.features_only:
            from tyremind.monitoring.features import run_feature_cycle
            cycle = lambda: run_feature_cycle(args.source_database, args.history, args.simulator_output, args.simulation_seed, args.feature_output)
        else:
            from tyremind.monitoring.predict import SavedModels, run_prediction_cycle
            models = SavedModels(args.wear_model,args.wear_artifact,args.anomaly_artifact)
            if args.prediction_only:
                cycle = lambda: run_prediction_cycle(args.source_database,args.history,args.simulator_output,args.simulation_seed,
                                                      args.feature_output,args.prediction_output,models)
            else:
                from tyremind.monitoring.risk import load_policy, run_risk_cycle
                load_policy(args.risk_config)
                def cycle():
                    risk = run_risk_cycle(args.source_database,args.history,args.simulator_output,args.simulation_seed,
                                           args.feature_output,args.prediction_output,models,args.risk_config,args.risk_output)
                    if args.risk_only:
                        return risk
                    from tyremind.monitoring.events import process_risk_events
                    events = process_risk_events(args.source_database,args.event_output)
                    return {**risk,'mode':'risk_events_ready','open_risk_events':events['open_events'],
                            'events_created':events['events_created'],'events_updated':events['events_updated'],
                            'investigations_queued':events['investigations_queued'],
                            'pending_investigations':events['pending_investigations'],
                            'message':'Events deduplicated and investigations queued; no AI agents executed yet.'}
    worker = MonitoringWorker(args.interval_seconds, args.output, cycle=cycle, max_cycles=args.max_cycles)
    try:
        if args.run_once:
            try:
                worker.execute_cycle()
            finally:
                worker.write_status(state='stopped', stopped_at=utc_now())
        else:
            worker.start()
        if worker.status['cycles_failed']:
            raise SystemExit(1)
    finally:
        worker.close_logging()


if __name__ == '__main__':
    main()
