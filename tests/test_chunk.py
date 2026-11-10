"""Passage coverage, provenance, policies and changed-source checks."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tyremind.rag.chunk import chunk_documents, normalize_text, split_page


class ChunkTests(unittest.TestCase):
    def test_overlap_offsets_and_coverage(self):
        text = ' '.join(f'word{i}' for i in range(75))
        fragments = list(split_page(text, max_words=20, overlap=5))
        self.assertEqual(fragments[0]['text'].split()[-5:], fragments[1]['text'].split()[:5])
        coverage = set()
        for fragment in fragments:
            self.assertEqual(fragment['text'], text[fragment['char_start']:fragment['char_end']])
            self.assertLessEqual(fragment['word_count'], 20)
            coverage.update(fragment['text'].split())
        self.assertEqual(coverage, set(text.split()))
        self.assertEqual(list(split_page('')), [])
        with self.assertRaises(ValueError):
            list(split_page(text, 20, 20))

    def test_normalization(self):
        self.assertEqual(normalize_text('  ﬁnite\t pressure\x00\n  next  line '), 'finite pressure\nnext line')

    def setup_sources(self, root):
        raw = root / 'raw'
        raw.mkdir()
        payload = b'fixture-pdf-content'
        (raw / 'test.pdf').write_bytes(payload)
        source = dict(id='manual', title='Test manual', publisher='Test publisher', filename='test.pdf',
                      url='https://example.com/manual.pdf', landing_url='https://example.com',
                      edition='test', region='test', enabled=True, expected_pages=2,
                      verified_sha256=hashlib.sha256(payload).hexdigest(),
                      chunk_sections=[dict(pdf_pages=[2,2], section='Inspection', printed_page_offset=-1)])
        disabled = dict(id='fmcsa', enabled=False)  # Missing file must not matter.
        catalog = root / 'sources.json'
        catalog.write_text(json.dumps({'sources': [source, disabled]}))
        reader = SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: 'Warranty excluded'),
                                       SimpleNamespace(extract_text=lambda: 'Inspect tyre pressure. Record findings.')])
        return catalog, raw, reader

    def test_provenance_section_filter_and_repeatability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, raw, reader = self.setup_sources(root)
            output = root / 'processed'
            with patch('tyremind.rag.chunk.PdfReader', return_value=reader):
                report = chunk_documents(catalog, raw, output)
                before = (output / 'chunks.jsonl').read_bytes()
                chunk_documents(catalog, raw, output)
            self.assertEqual(report['chunk_count'], 1)
            self.assertEqual(report['page_count'], 2)
            self.assertEqual(before, (output / 'chunks.jsonl').read_bytes())
            chunk = json.loads(before)
            self.assertEqual(chunk['pdf_page'], 2)
            self.assertEqual(chunk['printed_page'], '1')
            self.assertEqual(chunk['section'], 'Inspection')
            self.assertEqual(chunk['citation_url'], 'https://example.com/manual.pdf#page=2')
            pages = [json.loads(line) for line in (output / 'pages.jsonl').read_text().splitlines()]
            self.assertEqual(chunk['text'], pages[1]['text'][chunk['char_start']:chunk['char_end']])
            self.assertNotIn('Warranty', chunk['text'])
            self.assertEqual(report['documents'][1]['status'], 'disabled')

    def test_changed_pdf_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, raw, _ = self.setup_sources(root)
            (raw / 'test.pdf').write_bytes(b'changed-version')
            with self.assertRaisesRegex(ValueError, 'differs'):
                chunk_documents(catalog, raw, root / 'processed')
            self.assertFalse((root / 'processed').exists())


if __name__ == '__main__':
    unittest.main()
