# Local passage embeddings

## Run

From the cloned project folder, run commands one at a time:

```powershell
git pull
python -m pip install -e ".[rag]"
python -m tyremind.rag.embed
python -m unittest discover -s tests -v
```

The optional `rag` dependency group installs sentence-transformers and its dependencies, including PyTorch. The first installation can be large. This implementation uses CPU, so a GPU is not required. The first embedding run downloads model files from Hugging Face and needs internet; model files are cached locally for later use. No API key or paid embedding service is required. Existing source PDFs/passages remain local. Only model assets are downloaded; passage text is encoded locally rather than sent to an embedding API.

Input: `data/knowledge/processed/chunks.jsonl` from the previous step. If absent, run `python -m tyremind.rag.chunk` first. The base dependencies now include fonttools to improve embedded-font decoding. Existing models or databases are not modified.

## Model and token handling

Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It produces 384-dimensional embeddings. This is a pretrained general-purpose embedding model, not a tyre-domain model trained by TyreMind.

Word counts and model token counts differ. Each passage is checked with the model tokenizer against its 256-token limit, including special tokens. Oversized passages are divided into contiguous, nonoverlapping internal segments on whitespace boundaries. Each segment is encoded, then normalized vectors are combined using a word-count-weighted mean and normalized again. This retains every word without silently truncating long passages. It is a documented approximation to a whole-passage representation, not semantic rechunking. The original passage, ID and citations remain unchanged.

The verified corpus generated 82 passage vectors from 83 internal segments; one passage exceeded the token limit. Exact values can vary slightly across hardware/library versions. Model revision and library version are recorded.

## Output

`data/knowledge/embeddings/` contains:

| File | Contents |
|---|---|
| embeddings.npy | Float32 matrix; one normalized 384-value vector per chunk |
| chunk_metadata.jsonl | Original chunk text and references, embedding row index and segment count |
| embedding_report.json | Model/revision, shape, normalization, token handling and input/output checksums |

Vector row N corresponds to metadata `embedding_row=N`. Unique chunk IDs and text checksums are validated before model loading. Invalid/zero vectors fail the run. Source PDF checksums, titles, URLs and page references are carried through unchanged. Output is replaced on rerun and ignored by Git under `data/knowledge/`.

The report is written last and records checksums for both artifacts. A downstream consumer must verify these before reading to catch stale or interrupted exports. A failed run can leave older output; check command success and checksums before indexing.

To check matrix shape:

```powershell
python -c "import numpy as np; v=np.load('data/knowledge/embeddings/embeddings.npy', allow_pickle=False); print(v.shape)"
```

Expected for the current corpus: `(82, 384)`. Custom batch size can reduce memory use:

```powershell
python -m tyremind.rag.embed --batch-size 4
```

## Verification and next step

The complete suite has 38 tests. The embedding-specific unit tests use a small fake encoder so normal tests do not download weights; these verify alignment, normalization, token coverage and rejection of corrupted inputs. A separate actual MiniLM run generated the full 82 × 384 matrix and checked finite unit-length vectors.

ChromaDB storage, a retrieval interface and LLM answers are not implemented in this step. Later queries must use the same model/revision and normalization as the stored vectors. Numerical table exclusions from chunking remain in effect.
