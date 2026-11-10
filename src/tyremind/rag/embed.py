"""Generate local, source-linked passage embeddings without a vector database."""
import argparse
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import numpy as np

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def read_chunks(path):
    records, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        required = ('chunk_id', 'text', 'title', 'source_url', 'pdf_page', 'document_id',
                    'source_sha256', 'text_sha256')
        if any(key not in record for key in required):
            raise ValueError(f'Chunk line {number} lacks required provenance fields.')
        if not isinstance(record['text'], str) or not record['text'].strip():
            raise ValueError(f'Chunk line {number} has empty text.')
        if record['chunk_id'] in seen:
            raise ValueError('Duplicate chunk IDs.')
        if hashlib.sha256(record['text'].encode()).hexdigest() != record['text_sha256']:
            raise ValueError(f'Chunk text checksum mismatch at line {number}. Rerun chunking.')
        seen.add(record['chunk_id'])
        records.append(record)
    if not records:
        raise ValueError('No chunks to embed.')
    return records


def token_safe_segments(text, tokenizer, token_limit):
    """Split only when needed; every source word is encoded without truncation."""
    words = list(re.finditer(r'\S+', text))
    start = 0
    segments = []
    while start < len(words):
        low, high, best = start + 1, len(words), None
        while low <= high:
            end = (low + high) // 2
            fragment = text[words[start].start():words[end - 1].end()]
            count = len(tokenizer.encode(fragment, add_special_tokens=True, truncation=False, verbose=False))
            if count <= token_limit:
                best = (end, fragment, count)
                low = end + 1
            else:
                high = end - 1
        if best is None:
            raise ValueError('A source word exceeds the model token limit.')
        end, fragment, count = best
        segments.append({'text': fragment, 'word_count': end - start, 'token_count': count})
        start = end
    return segments


def encode_records(records, model, batch_size=16):
    pieces, owners, weights, counts = [], [], [], []
    for row, record in enumerate(records):
        segments = token_safe_segments(record['text'], model.tokenizer, model.max_seq_length)
        counts.append(len(segments))
        for segment in segments:
            pieces.append(segment['text'])
            owners.append(row)
            weights.append(segment['word_count'])
    encoded = np.asarray(model.encode(pieces, batch_size=batch_size, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    if encoded.ndim != 2 or encoded.shape[0] != len(pieces) or not np.isfinite(encoded).all():
        raise ValueError('Model returned invalid segment vectors.')
    vectors = np.zeros((len(records), encoded.shape[1]), dtype=np.float32)
    for owner, weight, vector in zip(owners, weights, encoded):
        vectors[owner] += weight * vector
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError('Model returned zero/invalid passage vectors.')
    vectors /= norms
    return vectors, counts, len(pieces)


def export_embeddings(chunks=Path('data/knowledge/processed/chunks.jsonl'),
                      output=Path('data/knowledge/embeddings'), batch_size=16, model=None):
    if batch_size < 1:
        raise ValueError('Batch size must be positive.')
    chunks, output = Path(chunks), Path(output)
    if chunks.resolve().parent == output.resolve():
        raise ValueError('Embedding output must be separate from chunk input.')
    records = read_chunks(chunks)
    # Lazy import allows validation and unit tests without downloading model weights.
    if model is None:
        from sentence_transformers import SentenceTransformer
        print(f'Loading {MODEL_NAME} on CPU; first run downloads model files.', flush=True)
        model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device='cpu', trust_remote_code=False)
    vectors, counts, total_segments = encode_records(records, model, batch_size)
    if vectors.shape[1] != 384:
        raise ValueError('Expected 384-dimensional MiniLM embeddings.')
    output.mkdir(parents=True, exist_ok=True)
    vector_file = output / 'embeddings.npy'
    metadata_file = output / 'chunk_metadata.jsonl'
    # Write report last, with output checksums, so downstream consumers can verify a consistent export.
    temporary = output / 'embeddings.npy.tmp'
    with temporary.open('wb') as handle:
        np.save(handle, vectors, allow_pickle=False)
    temporary.replace(vector_file)
    temporary = output / 'chunk_metadata.jsonl.tmp'
    with temporary.open('w', encoding='utf-8') as handle:
        for row, (record, count) in enumerate(zip(records, counts)):
            handle.write(json.dumps({**record, 'embedding_row': row, 'embedding_segments': count},
                                    ensure_ascii=False) + '\n')
    temporary.replace(metadata_file)
    try:
        version = importlib.metadata.version('sentence-transformers')
    except importlib.metadata.PackageNotFoundError:
        version = 'not installed; injected test encoder'
    report = {'embedding_version': '1.0', 'model': MODEL_NAME, 'model_revision': MODEL_REVISION,
              'sentence_transformers_version': version, 'dimensions': 384,
              'chunks': len(records), 'encoded_segments': total_segments,
              'chunks_split_for_token_limit': sum(count > 1 for count in counts),
              'model_token_limit': int(model.max_seq_length), 'normalized': True,
              'dtype': str(vectors.dtype), 'device': 'cpu',
              'aggregation': 'Word-count-weighted mean of normalized segments, then L2 normalize.',
              'input_sha256': hashlib.sha256(chunks.read_bytes()).hexdigest(),
              'embeddings_sha256': hashlib.sha256(vector_file.read_bytes()).hexdigest(),
              'metadata_sha256': hashlib.sha256(metadata_file.read_bytes()).hexdigest(),
              'note': 'Local embedding artifacts only; no ChromaDB index or LLM call.'}
    (output / 'embedding_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/knowledge/processed/chunks.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('data/knowledge/embeddings'))
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(export_embeddings(args.input, args.output, args.batch_size), indent=2))


if __name__ == '__main__':
    main()
