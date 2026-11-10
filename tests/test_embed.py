"""Offline tests for embedding safety, provenance and output alignment."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tyremind.rag.embed import read_chunks, token_safe_segments, encode_records, export_embeddings


class FakeTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split()) + 2))


class FakeModel:
    tokenizer = FakeTokenizer()
    max_seq_length = 10

    def encode(self, texts, **kwargs):
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            vectors[i, 0] = 1
            vectors[i, 1] = len(text.split()) / 10
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


def record(identity, text):
    return dict(chunk_id=identity, text=text, title='Test manual', source_url='https://example.com/manual.pdf',
                pdf_page=7, document_id='manual', source_sha256='test',
                text_sha256=hashlib.sha256(text.encode()).hexdigest())


class EmbeddingTests(unittest.TestCase):
    def test_token_segments_cover_all_words(self):
        text = ' '.join(f'word{i}' for i in range(23))
        segments = token_safe_segments(text, FakeTokenizer(), 10)
        self.assertEqual(' '.join(s['text'] for s in segments), text)
        self.assertTrue(all(s['token_count'] <= 10 for s in segments))
        self.assertEqual(sum(s['word_count'] for s in segments), 23)

    def test_shape_normalization_and_order(self):
        records = [record('a', 'one two'), record('b', ' '.join(['long'] * 25))]
        vectors, counts, pieces = encode_records(records, FakeModel())
        self.assertEqual(vectors.shape, (2, 384))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
        self.assertEqual(counts, [1, 4])
        self.assertEqual(pieces, 5)
        self.assertTrue(np.isfinite(vectors).all())

    def test_export_keeps_source_metadata_and_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'chunks.jsonl'
            rows = [record('a', 'pressure loss'), record('b', 'temperature and load')]
            source.write_text('\n'.join(json.dumps(r) for r in rows))
            report = export_embeddings(source, root / 'out', model=FakeModel())
            vectors = np.load(root / 'out/embeddings.npy', allow_pickle=False)
            metadata = [json.loads(line) for line in (root / 'out/chunk_metadata.jsonl').read_text().splitlines()]
            self.assertEqual(vectors.shape, (2, 384))
            self.assertEqual([r['chunk_id'] for r in metadata], ['a', 'b'])
            self.assertEqual([r['embedding_row'] for r in metadata], [0, 1])
            self.assertEqual(metadata[0]['pdf_page'], 7)
            self.assertEqual(report['metadata_sha256'], hashlib.sha256(
                (root / 'out/chunk_metadata.jsonl').read_bytes()).hexdigest())

    def test_corrupted_and_duplicate_chunks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'chunks.jsonl'
            row = record('a', 'pressure')
            path.write_text(json.dumps(row) + '\n' + json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_chunks(path)
            row['text'] = 'modified'
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_chunks(path)
            path.write_text('')
            with self.assertRaisesRegex(ValueError, 'No chunks'):
                read_chunks(path)

    def test_nonfinite_model_output_rejected(self):
        class InvalidModel(FakeModel):
            def encode(self, texts, **kwargs):
                return np.full((len(texts), 384), np.nan)
        with self.assertRaisesRegex(ValueError, 'invalid'):
            encode_records([record('a', 'pressure')], InvalidModel())


if __name__ == '__main__':
    unittest.main()
