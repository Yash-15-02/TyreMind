"""Validate compressed PDF collection and failed-download reporting offline."""
import gzip
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tyremind.rag.collect import collect


class CollectionTests(unittest.TestCase):
    def run_download(self, payload):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        catalog = root / 'sources.json'
        catalog.write_text(json.dumps({'sources': [{'id': 'sample', 'filename': 'test.pdf',
                                                     'url': 'https://example.com/test.pdf'}]}))
        response = io.BytesIO(payload)
        response.headers = {'Content-Type': 'application/pdf'}
        response.geturl = lambda: 'https://example.com/test.pdf'
        with patch('tyremind.rag.collect.urlopen', return_value=response), redirect_stdout(io.StringIO()):
            result = collect(catalog, root / 'raw')
        return result, root / 'raw'

    def test_plain_pdf(self):
        payload = b'%PDF-1.7\nexample\n%%EOF'
        result, output = self.run_download(payload)
        self.assertEqual(result['downloaded'], 1)
        self.assertEqual((output / 'test.pdf').read_bytes(), payload)
        self.assertEqual(len(result['documents'][0]['sha256']), 64)

    def test_compressed_pdf(self):
        payload = b'%PDF-1.7\nexample\n%%EOF'
        result, output = self.run_download(gzip.compress(payload))
        self.assertEqual(result['downloaded'], 1)
        self.assertEqual((output / 'test.pdf').read_bytes(), payload)

    def test_html_rejected(self):
        result, output = self.run_download(b'<html>Access denied</html>')
        self.assertEqual(result['failed'], 1)
        self.assertFalse((output / 'test.pdf').exists())
        self.assertIn('error', result['documents'][0])


if __name__ == '__main__':
    unittest.main()
