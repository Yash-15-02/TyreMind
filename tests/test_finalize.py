"""Finalization must reject changed inputs and mismatched recorded metrics."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'Install .[ml] for finalization checks')
class FinalizationTests(unittest.TestCase):
    def test_changed_inputs_rejected(self):
        from tyremind.ml.finalize import verify_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'features.csv'
            source.write_bytes(b'original')
            report = {'input_sha256': {'features.csv': hashlib.sha256(b'original').hexdigest()}}
            verify_inputs(report, root)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Input changed'):
                verify_inputs(report, root)

    def test_metric_mismatch_rejected(self):
        from tyremind.ml.finalize import verify_metrics
        verify_metrics({'mae': .1}, {'mae': .1}, ['mae'])
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            verify_metrics({'mae': .2}, {'mae': .1}, ['mae'])


if __name__ == '__main__':
    unittest.main()
