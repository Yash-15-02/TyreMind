# PDF text extraction and source-preserving chunking

## Run on Windows or other platforms

From the cloned project root, run one command at a time:

```bash
git pull
python -m pip install -e .
python -m tyremind.rag.chunk
python -m unittest discover -s tests -v
```

Existing downloaded manufacturer PDFs are reused. No API key or LLM is needed. `pypdf` is now a project dependency and is installed by the pip command. FMCSA is disabled in the catalog and ignored even if an old file or failed manifest entry exists. The downloader also skips disabled sources on future runs.

If raw PDFs are missing, first run `python -m tyremind.rag.collect`; the revised expected collection is 3 downloads and zero failures.

## Outputs

All extracted text stays local under `data/knowledge/processed/` and is excluded from Git:

- `pages.jsonl`: one record per PDF page, normalized text plus source metadata; all 224 manufacturer-document pages. This is an extraction archive, not the retrieval corpus.
- `chunks.jsonl`: selected page-bounded passages, up to 180 whitespace-delimited words each, with 30-word overlap. Nearby sentence endings are preferred when possible. A short final chunk can contain fewer words.
- `chunk_report.json`: source-specific page/chunk counts, exclusions and empty-page reporting.

Default verification produced 82 chunks: 70 Michelin service-manual chunks and 12 Bridgestone safety-manual chunks. Text extraction/chunk counts can vary with pypdf versions. The full suite now has 33 tests.

## Scope and source references

The source catalog explicitly selects 1-based PDF page ranges: Michelin 55–57, 59–63, 69–80 and 85–86; Bridgestone 6–7. These cover pressure, inspection, monitoring, alignment/wear and brake-related heat. The pipeline avoids automatically indexing entire manuals, warranty administration and mounting instructions. It does not infer a diagnosis from these documents.

The Michelin inflation-table PDF is extracted into the page archive but produces zero retrieval chunks until its tabular layout is visually verified. Other selected manual pages can also contain diagrams or tables: page text does not reliably preserve their spatial relationships, and numerical recommendations still require source-page verification. Page-bounded chunks may omit adjacent-page context; future retrieval should fetch neighboring chunks/pages when needed and preserve warnings.

Each passage records:

- Stable chunk ID and document ID.
- Document title, publisher, edition, region and original filename.
- Original source URL, landing-page URL and a `#page=N` citation link.
- 1-based PDF page and broad catalog section.
- Printed page where verified by catalog offset (Michelin printed page = PDF page minus 5). Otherwise null rather than guessed.
- Source-file SHA-256, text SHA-256 and page chunk number.
- Exact start-inclusive/end-exclusive character offsets into the normalized page archive.

Normalization expands Unicode ligatures, removes null characters and tidies whitespace; it does not summarize or paraphrase source text. Chunks never cross PDF pages, preserving unambiguous citation locations. Original publisher PDFs are unchanged.

## Version checks and failures

Catalog PDF checksums and page counts pin the reviewed source versions, preventing a publisher update from silently shifting curated page references. A mismatch, missing enabled PDF, invalid page range or parsing error stops the run before new output is written. Previously generated output can remain after a failed rerun; check command success before continuing. Empty extracted pages are reported, not OCR'd. No passages at all is an error. Outputs are replaced on a successful rerun; they are not appended.

This step does not generate embeddings, populate ChromaDB or call an LLM. The later embedding pipeline must read `chunks.jsonl`, not the complete page archive, to honor source-selection rules.
