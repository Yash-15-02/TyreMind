"""Real scheduler ticks, overlap prevention, recovery and graceful CLI exit."""
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('apscheduler'), 'Install base project dependencies')
class WorkerTests(unittest.TestCase):
    def worker(self, **kwargs):
        from tyremind.monitoring.worker import MonitoringWorker
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        worker = MonitoringWorker(directory=Path(temporary.name), **kwargs)
        self.addCleanup(worker.close_logging)
        return worker

    def test_once_persists_heartbeat(self):
        worker = self.worker()
        worker.execute_cycle()
        status = json.loads((worker.directory / 'worker_status.json').read_text())
        self.assertEqual(status['cycles_succeeded'], 1)
        self.assertEqual(status['last_result']['tyres_processed'], 0)
        self.assertEqual(status['last_result']['mode'], 'scheduler_heartbeat')
        self.assertTrue((worker.directory / 'worker.log').exists())

    def test_invalid_intervals_rejected(self):
        from tyremind.monitoring.worker import MonitoringWorker
        for interval in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                MonitoringWorker(interval_seconds=interval)

    def test_overlap_is_skipped(self):
        started, release = threading.Event(), threading.Event()
        def cycle():
            started.set()
            if not release.wait(3):
                raise RuntimeError('Test timed out')
            return {'ok': True}
        worker = self.worker(cycle=cycle)
        thread = threading.Thread(target=worker.execute_cycle)
        thread.start()
        try:
            self.assertTrue(started.wait(2))
            worker.execute_cycle()
            self.assertEqual(worker.status['cycles_started'], 1)
        finally:
            release.set()
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker.status['cycles_succeeded'], 1)

    def test_cycle_failure_does_not_disable_future_cycle(self):
        attempts = []
        def cycle():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError('Injected failure')
            return {'ok': True}
        worker = self.worker(cycle=cycle)
        with self.assertRaisesRegex(RuntimeError, 'Injected failure'):
            worker.execute_cycle()
        worker.execute_cycle()
        self.assertEqual(worker.status['cycles_failed'], 1)
        self.assertEqual(worker.status['cycles_succeeded'], 1)
        self.assertIsNone(worker.status['last_error'])

    def test_real_interval_scheduler_stops_after_two_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-m', 'tyremind.monitoring.worker',
                                     '--interval-seconds', '0.15', '--max-cycles', '2', '--output', directory],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            status = json.loads((Path(directory) / 'worker_status.json').read_text())
            self.assertEqual(status['state'], 'stopped')
            self.assertEqual(status['cycles_succeeded'], 2)
            self.assertEqual(status['cycles_failed'], 0)

    @unittest.skipIf(os.name == 'nt', 'POSIX termination signal test; Windows uses console Ctrl+C')
    def test_shutdown_signal_records_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen([sys.executable, '-m', 'tyremind.monitoring.worker',
                                        '--interval-seconds', '60', '--output', directory],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 8
                statusfile = Path(directory) / 'worker_status.json'
                while not statusfile.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue(statusfile.exists())
                # Status file is written just before scheduler.start(); wait for startup log.
                time.sleep(.1)
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=8), 0)
                self.assertEqual(json.loads(statusfile.read_text())['state'], 'stopped')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    unittest.main()
