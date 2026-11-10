"""Store verified embeddings and citations in persistent local ChromaDB."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tyremind.rag.embed import MODEL_NAME, MODEL_REVISION, read_chunks


def read_embedding_export(directory):
    directory = Path(directory)
    report = json.loads((directory / 'embedding_report.json').read_text(encoding='utf-8'))
    for filename, key in [('embeddings.npy', 'embeddings_sha256'),
                          ('chunk_metadata.jsonl', 'metadata_sha256')]:
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != report[key]:
            raise ValueError(f'{filename}: checksum mismatch; rerun embedding generation.')
    if report['model'] != MODEL_NAME or report['model_revision'] != MODEL_REVISION:
        raise ValueError('Embedding model/revision differs from the configured query model.')
    records = read_chunks(directory / 'chunk_metadata.jsonl')
    vectors = np.load(directory / 'embeddings.npy', allow_pickle=False)
    if vectors.shape != (len(records), 384) or report['chunks'] != len(records) or report['dimensions'] != 384:
        raise ValueError('Embedding count or dimensions do not match metadata.')
    if report.get('normalized') is not True or not np.isfinite(vectors).all():
        raise ValueError('Embeddings must be finite normalized vectors.')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError('Embedding vectors are not unit length.')
    if [record.get('embedding_row') for record in records] != list(range(len(records))):
        raise ValueError('Metadata row order does not match embeddings.')
    return records, vectors, report


def scalar_metadata(record):
    # Chroma metadata has no null values. Original null fields remain in the export.
    return {key: value for key, value in record.items()
            if key != 'text' and isinstance(value, (str, int, float, bool))}


def verify_collection(collection, records, vectors):
    if collection.count() != len(records):
        raise ValueError('Stored collection has an unexpected record count.')
    for start in range(0, len(records), 100):
        batch = records[start:start + 100]
        found = collection.get(ids=[row['chunk_id'] for row in batch],
                               include=['documents', 'metadatas', 'embeddings'])
        positions = {identity: index for index, identity in enumerate(found['ids'])}
        for offset, record in enumerate(batch):
            if record['chunk_id'] not in positions:
                raise ValueError('Stored chunk is missing.')
            index = positions[record['chunk_id']]
            if found['documents'][index] != record['text'] or found['metadatas'][index] != scalar_metadata(record):
                raise ValueError('Stored text/source metadata differs from the embedding export.')
            if not np.allclose(found['embeddings'][index], vectors[start + offset], atol=1e-6):
                raise ValueError('Stored vector differs from the embedding export.')


def store_embeddings(embeddings=Path('data/knowledge/embeddings'),
                     database=Path('data/knowledge/chroma'), batch_size=100):
    if not 1 <= batch_size <= 500:
        raise ValueError('Use a batch size between 1 and 500.')
    records, vectors, report = read_embedding_export(embeddings)
    from chromadb import PersistentClient
    from chromadb.config import Settings
    database = Path(database)
    database.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(database), settings=Settings(anonymized_telemetry=False))
    try:
        # New source/vector snapshots get new collections; publish active pointer only after verification.
        fingerprint = hashlib.sha256(json.dumps({key: report[key] for key in
            ('embeddings_sha256', 'metadata_sha256', 'model', 'model_revision')}, sort_keys=True).encode()).hexdigest()
        name = f'tyremind_{fingerprint[:24]}'
        collection = client.get_or_create_collection(
            name=name, embedding_function=None, configuration={'hnsw': {'space': 'cosine'}},
            metadata={'snapshot_sha256': fingerprint, 'model': MODEL_NAME, 'model_revision': MODEL_REVISION})
        if collection.metadata.get('snapshot_sha256') != fingerprint:
            raise ValueError('Existing collection fingerprint mismatch.')
        existing_ids = set(collection.get(include=[])['ids'])
        expected_ids = {record['chunk_id'] for record in records}
        if existing_ids - expected_ids:
            raise ValueError('Unexpected records in snapshot collection.')
        inserted = len(expected_ids - existing_ids)
        # Upsert resumes incomplete snapshots and makes repeat imports duplicate-free.
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            collection.upsert(ids=[row['chunk_id'] for row in batch],
                              embeddings=vectors[start:start + batch_size].tolist(),
                              documents=[row['text'] for row in batch],
                              metadatas=[scalar_metadata(row) for row in batch])
        verify_collection(collection, records, vectors)
        status = {'database': str(database), 'collection': name, 'snapshot_sha256': fingerprint,
                  'records': collection.count(), 'dimensions': 384, 'distance_metric': 'cosine',
                  'model': MODEL_NAME, 'model_revision': MODEL_REVISION,
                  'embeddings_sha256': report['embeddings_sha256'], 'metadata_sha256': report['metadata_sha256'],
                  'inserted_records': inserted, 'existing_records': len(expected_ids & existing_ids),
                  'verification': 'All IDs, vectors, passage texts and scalar source metadata matched.'}
        temporary = database / 'active_index.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        temporary.replace(database / 'active_index.json')
        return status
    finally:
        client.close()



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/knowledge/embeddings'))
    parser.add_argument('--database', type=Path, default=Path('data/knowledge/chroma'))
    parser.add_argument('--batch-size', type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(store_embeddings(args.input, args.database, args.batch_size), indent=2))


if __name__ == '__main__':
    main()
