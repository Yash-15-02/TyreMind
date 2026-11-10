"""Artifact validation plus real local Chroma persistence/reimport tests."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tyremind.rag.embed import MODEL_NAME, MODEL_REVISION
from tyremind.rag.store import read_embedding_export, store_embeddings


def fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    vectors = np.eye(3, 384, dtype=np.float32)
    np.save(root / 'embeddings.npy', vectors, allow_pickle=False)
    rows = []
    for index, text in enumerate(['pressure inspection', 'alignment wear', 'load temperature']):
        rows.append(dict(chunk_id=f'chunk-{index}', document_id='test', text=text, title='Test manual',
                         publisher='Test', source_url='https://example.com/manual.pdf', pdf_page=index + 1,
                         printed_page=None, source_sha256='test-source', embedding_row=index,
                         text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    (root / 'chunk_metadata.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    report = dict(model=MODEL_NAME, model_revision=MODEL_REVISION, chunks=3, dimensions=384, normalized=True)
    update_report(root, report)
    return rows, vectors, report


def update_report(root, report):
    for filename, key in [('embeddings.npy', 'embeddings_sha256'), ('chunk_metadata.jsonl', 'metadata_sha256')]:
        report[key] = hashlib.sha256((root / filename).read_bytes()).hexdigest()
    (root / 'embedding_report.json').write_text(json.dumps(report))


class StoreValidationTests(unittest.TestCase):
    def test_corrupt_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            (root / 'chunk_metadata.jsonl').write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                read_embedding_export(root)

    def test_wrong_order_and_invalid_vectors_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, vectors, report = fixture(root)
            rows[0]['embedding_row'] = 2
            (root / 'chunk_metadata.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            update_report(root, report)
            with self.assertRaisesRegex(ValueError, 'order'):
                read_embedding_export(root)
            rows[0]['embedding_row'] = 0
            (root / 'chunk_metadata.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            vectors[0] = 0
            np.save(root / 'embeddings.npy', vectors)
            update_report(root, report)
            with self.assertRaisesRegex(ValueError, 'unit length'):
                read_embedding_export(root)


@unittest.skipUnless(importlib.util.find_spec('chromadb'), 'Install .[rag] for Chroma integration tests')
class ChromaTests(unittest.TestCase):
    def test_persistent_storage_query_and_repeat_import(self):
        from chromadb import PersistentClient
        from chromadb.config import Settings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows, vectors, _ = fixture(root / 'embeddings')
            first = store_embeddings(root / 'embeddings', root / 'chroma')
            second = store_embeddings(root / 'embeddings', root / 'chroma')
            self.assertEqual(first['inserted_records'], 3)
            self.assertEqual(second['inserted_records'], 0)
            self.assertEqual(first['collection'], second['collection'])
            client = PersistentClient(path=str(root / 'chroma'), settings=Settings(anonymized_telemetry=False))
            collection = client.get_collection(first['collection'], embedding_function=None)
            found = collection.query(query_embeddings=[vectors[0].tolist()], n_results=1,
                                     include=['documents', 'metadatas'])
            self.assertEqual(found['ids'][0], ['chunk-0'])
            self.assertEqual(found['metadatas'][0][0]['pdf_page'], 1)
            self.assertEqual(found['documents'][0][0], rows[0]['text'])
            client.delete_collection(first['collection'])
            client.close()

    def test_invalid_input_leaves_active_collection_unchanged(self):
        from chromadb import PersistentClient
        from chromadb.config import Settings
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root / 'embeddings')
            first = store_embeddings(root / 'embeddings', root / 'chroma')
            pointer = (root / 'chroma/active_index.json').read_bytes()
            (root / 'embeddings/embeddings.npy').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                store_embeddings(root / 'embeddings', root / 'chroma')
            self.assertEqual(pointer, (root / 'chroma/active_index.json').read_bytes())
            client = PersistentClient(path=str(root / 'chroma'), settings=Settings(anonymized_telemetry=False))
            self.assertEqual(client.get_collection(first['collection'], embedding_function=None).count(), 3)
            client.delete_collection(first['collection'])
            client.close()


if __name__ == '__main__':
    unittest.main()
