# Persistent local ChromaDB storage

This is the actual ChromaDB vector database, running in embedded persistent mode. No Chroma Cloud/Pinecone account, API key, hosted database or separate server is required. Data is saved on the machine running the command.

## Run

From the cloned project folder, run one command at a time:

```powershell
git pull
python -m pip install -e ".[rag]"
python -m tyremind.rag.store
python -m unittest discover -s tests -v
```

Requires the completed embedding export in `data/knowledge/embeddings/`. It reuses those vectors; it does not load sentence-transformers, download a new embedding model or call an LLM. Installing the Chroma dependency needs internet the first time. Storage uses precomputed embeddings with Chroma's automatic embedding function disabled. Anonymous telemetry is disabled in client settings.

## Stored data

Default persistent folder: `data/knowledge/chroma/`. It contains Chroma's SQLite catalog and vector-index files, plus `active_index.json` identifying the current verified collection. These are local generated files excluded from Git. Chroma's internal SQLite database is separate from TyreMind's operational `data/tyremind.db`.

Each record contains a unique chunk ID, its 384-dimensional vector, the original passage text and scalar source metadata (title, publisher, source URL, PDF page, edition, section, checksums and other available references). Null metadata fields are omitted because Chroma metadata does not accept null values; the original JSONL export is unchanged.

The collection uses cosine distance. A distance is not a calibrated confidence or probability. Future questions must be embedded with the same model/revision and normalization before querying. A general retrieval command/agent tool is a later step.

## Verification and repeat imports

Before any database mutation, the importer verifies the artifact checksums, model identity/revision, dimensions, row alignment, unique IDs, passage text checksums, finite vectors and unit norms. It then stores the records and reads them back to verify every ID, vector, document and scalar metadata entry.

A successful first import reports 82 records and 82 inserted records. Reimporting the same export reports 82 records, zero inserted and 82 existing, without duplicates. A hash of the vector/text export identifies each collection; new exports create separate collections. The active pointer is replaced only after verification succeeds. Old snapshots are retained, not automatically deleted. Do not run multiple importers concurrently in this local MVP. A failed new snapshot can leave an inactive partial collection, but does not publish it as active.

The importer closes its client in all cases. Persistence does not require a separate `persist()` call. Closing the terminal does not erase the database.

## Inspect the result

```powershell
Get-Content .\data\knowledge\chroma\active_index.json
explorer .\data\knowledge\chroma
```

Collection names include a content fingerprint and will differ across machines if their extraction/embeddings differ. Use `active_index.json`, not a hardcoded collection name. Do not manually edit internal database/index files.

The full suite has 42 tests; the two real-Chroma integration tests run when the optional `rag` dependencies are installed. They verify persistent reads, vector queries, source references, repeat import behavior and preservation of the active index when inputs are corrupted. Unit tests use artificial vectors; the actual 82-record corpus was separately imported and verified in development.
