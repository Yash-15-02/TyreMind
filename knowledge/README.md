# TyreMind technical reference collection

Three manufacturer PDFs are enabled. FMCSA is disabled at the user's request after HTTP 403; its catalog entry remains for provenance. `sources.json` records titles, publishers, editions, regions, URLs, topics and section hints. Source-preserving chunking and local embeddings are implemented; ChromaDB and agent retrieval are later steps. See [chunking instructions](../docs/chunking.md).

| Source | Intended role |
|---|---|
| [MICHELIN Truck Tire Service Manual, 2023](https://business.michelinman.com/tips-advice/resources) | Primary maintenance, pressure, wear and damage reference |
| [MICHELIN Load and Inflation Table Standard](https://business.michelinman.com/tips-advice/resources) | Product/size/load-specific supplement; visually verify tables before indexing |
| [Bridgestone Truck Tire Limited Warranty & Safety Manual](https://commercial.bridgestone.com/content/dam/bcs-sites/bridgestone-commercial/images/warranty/bridgestone-us-englishtbr-limited-warranty-and-safety-manual-v2.pdf) | Secondary inspection/maintenance reference; exclude warranty material |
| [FMCSA USDOT Tire Advisory Card, June 2016](https://www.fmcsa.dot.gov/sites/fmcsa.dot.gov/files/docs/USDOT_1168_1097_TireAdvisory_508.pdf) | Concise commercial-vehicle inspection checklist |

## Download on your computer

From the repository root:

```bash
python -m tyremind.rag.collect
```

Original PDFs and `collection_manifest.json` are saved in `data/knowledge/raw/`. No new Python dependencies or API keys are needed. Downloads require network access; source URLs can change. Failures are recorded and return a nonzero command exit status. Repeated runs fetch and replace successfully downloaded files; a failed fetch leaves any existing file untouched, but its manifest status is failed for the current attempt. Check the manifest before indexing.

The collector validates the PDF signature/end marker, handles gzip-compressed delivery, limits download/decompressed size to 25 MiB, and records resolved URLs, byte counts, SHA-256 checksums and download timestamps. The collection was additionally opened and text extraction checked in the development environment; the collection command itself does not require or run a full PDF parser.

Original publisher PDFs remain local and are excluded from Git. The repository stores the source catalog and collection code, not republished manuals or extracted full text. Source access does not grant an open redistribution license.

## Retrieval preparation rules for the next step

- Preserve document ID, title, publisher, URL, edition and 1-based PDF page in every extracted passage. Keep printed page labels distinct from PDF page numbers.
- Start with relevant pressure, inspection, wear and maintenance sections from the Michelin manual and the FMCSA checklist. Preserve warnings and context. Do not automatically index every page, warranty provision, mounting or repair procedure.
- Inspect table extraction manually before using inflation values. Product-specific tables are not universal operating thresholds.
- Keep manufacturer and region metadata attached; do not attribute Bridgestone instructions to Michelin or treat U.S. material as Indian regulations.
- The FMCSA advisory is dated 2016; it is not proof of current regulatory compliance. Manufacturer editions are reference snapshots, not a claim that these are the latest publications.
- Documents ground explanations; they do not validate our simulator coefficients, certify safety or authorize maintenance without human review.
