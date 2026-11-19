"""Retrieve cited Phase 1 passages for one investigation's triggered rules."""
import argparse
import json
from pathlib import Path

from tyremind.agents.evidence import load_evidence
from tyremind.agents.supervisor import SupervisorCheckpoint, digest, export_json
from tyremind.monitoring.events import transaction
from tyremind.rag.embed import MODEL_NAME, MODEL_REVISION


def retrieve(index_dir, rules):
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    pointer = index_dir / 'active_index.json'
    if not pointer.is_file():
        raise FileNotFoundError('Phase 1 active index missing. Run python -m tyremind.rag.store first.')
    index = json.loads(pointer.read_text(encoding='utf-8'))
    if (index['model'], index['model_revision']) != (MODEL_NAME, MODEL_REVISION):
        raise ValueError('The active index requires a different embedding model.')
    client = chromadb.PersistentClient(path=str(index_dir), settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(index['collection'], embedding_function=None)
        if collection.metadata['snapshot_sha256'] != index['snapshot_sha256'] or collection.count() != index['records'] or not index['records']:
            raise ValueError('The active Chroma collection does not match its Phase 1 index.')
        topics = sorted({rule['rule_id'].replace('_', ' ') for rule in rules})
        query = 'Truck tire inspection and maintenance guidance for ' + ', '.join(topics)
        model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device='cpu', trust_remote_code=False)
        vector = model.encode([query], normalize_embeddings=True)
        result = collection.query(query_embeddings=vector.tolist(), n_results=min(3, index['records']),
                                  include=['metadatas', 'documents', 'distances'])
        passages = [dict(chunk_id=key, text=text, source=metadata, cosine_distance=distance)
                    for key, text, metadata, distance in zip(result['ids'][0], result['documents'][0],
                        result['metadatas'][0], result['distances'][0], strict=True)]
        return dict(query=query, collection=index['collection'], snapshot_sha256=index['snapshot_sha256'],
                    passages=passages)
    finally:
        client.close()


def load_knowledge(database=Path('data/tyremind.db'), consumer_id='local-phase3',
                   index_dir=Path('data/knowledge/chroma'), root=Path('data/phase3')):
    evidence = load_evidence(database, consumer_id, root)
    if evidence['action'] == 'NO_WORK':
        return evidence
    identity = Path(evidence['evidence_path']).stem
    with transaction(database) as connection:
        row = connection.execute('SELECT state_json,state_sha256 FROM phase3_supervisor_states WHERE investigation_id=?',
                                 (identity,)).fetchone()
        if digest(row['state_json']) != row['state_sha256']:
            raise ValueError('Supervisor checkpoint checksum mismatch.')
        case = SupervisorCheckpoint.model_validate_json(row['state_json']).investigation
        connection.execute('''CREATE TABLE IF NOT EXISTS phase3_knowledge (
            investigation_id TEXT PRIMARY KEY REFERENCES investigations(investigation_id), payload TEXT NOT NULL)''')
        saved = connection.execute('SELECT payload FROM phase3_knowledge WHERE investigation_id=?', (identity,)).fetchone()
    if saved:
        packet = json.loads(saved['payload'])
        action = 'RESUMED'
    else:
        packet = retrieve(index_dir, case.source_handoff['triggered_rules'])
        packet.update(investigation_id=identity, tyre_id=case.tyre_id, handoff_id=case.handoff_id)
        with transaction(database) as connection:
            owner = connection.execute("""SELECT 1 FROM investigations j JOIN phase3_receipts r USING(investigation_id)
                WHERE investigation_id=? AND j.status='RUNNING' AND r.consumer_id=? AND r.claim_token=j.claim_token""",
                (identity, consumer_id)).fetchone()
            if not owner:
                raise ValueError('Investigation ownership changed during retrieval.')
            inserted = connection.execute('INSERT OR IGNORE INTO phase3_knowledge VALUES (?,?)',
                                          (identity, json.dumps(packet, allow_nan=False))).rowcount
            packet = json.loads(connection.execute('SELECT payload FROM phase3_knowledge WHERE investigation_id=?',
                                                   (identity,)).fetchone()[0])
            action = 'RETRIEVED' if inserted else 'RESUMED'
    path = root/'knowledge'/f'{identity}.json'
    export_json(path, packet)
    return dict(action=action, tyre_id=packet['tyre_id'], query=packet['query'], output=str(path),
                citations=[p['source']['citation_url'] for p in packet['passages']])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, default=Path('data/tyremind.db'))
    parser.add_argument('--consumer-id', default='local-phase3')
    parser.add_argument('--index-dir', type=Path, default=Path('data/knowledge/chroma'))
    print(json.dumps(load_knowledge(**vars(parser.parse_args())), indent=2))
