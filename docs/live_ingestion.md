# Phase 2, Step 3: validate new readings and store them in SQLite

## Run on Windows

Stop any running worker with Ctrl+C. From the cloned project folder, run commands separately:

```powershell
git pull
python -m pip install -e .
python -m tyremind.monitoring.ingest
```

The ingestion command drains existing simulator batches without generating another day. A repeat run reports zero processed batches and zero inserts. If there are no simulator batches, it reports zero; it does not create readings. The operational database from Phase 1 must already exist.

Run one complete generation/validation/storage cycle immediately:

```powershell
python -m tyremind.monitoring.worker --run-once
```

Normal scheduling uses the same pipeline every 60 seconds:

```powershell
python -m tyremind.monitoring.worker
```

Press Ctrl+C to stop. Do not run simultaneous workers for this MVP. In the first cycle after updating, all previously generated waiting batches are ingested before another day is generated. Therefore inserted rows may exceed the 120 freshly generated readings. Later default cycles usually insert 120 readings each. Periodic inspections are ingested when emitted. Bad rows can reduce inserted counts.

## Actual pipeline

```text
Drain earlier raw batches
    → Generate/export one further daily batch
    → Pandas validation
    → Check stored identities, duplicate keys and mileage neighbors
    → Accepted rows + quarantine + audit ledger commit together
```

The authoritative raw source is `data/simulator/simulator.db`'s exported batch queue. This queue stores the same records as raw CSV exports. Editing an exported CSV does not edit the queued input. Hidden `simulation_truth` is never read into operational records; only telemetry and measured inspections are used.

The importer reuses Step 2's Pandas schemas, timestamp/numeric conversion, finite/nonnegative-value checks, missing-value detection, duplicate/conflicting-key checks, tyre/truck identity and installation checks. No imputation is performed. Low pressure and high temperature alone are not rejected: they can be valid fault evidence.

Additional live checks compare cumulative mileage against both earlier and later stored observations, so consistent out-of-order readings can be accepted while mileage regressions are quarantined. Existing identical keys are skipped; existing changed keys are quarantined and never overwrite stored readings. This assumes one installation per tyre, as in the MVP. Replacement lifecycle handling remains future work.

## Operational database and audit tables

Valid readings and inspections are added to existing `data/tyremind.db` tables, preserving the historical rows. Two audit tables are added without deleting/migrating the earlier tables:

| Table | Purpose |
|---|---|
| ingestion_batches | Source/batch identity, input checksum, timestamp and JSON report |
| ingestion_rejections | Original rejected fields, table name, source row and rejection reasons |

Source identity is a deterministic fingerprint of the historical simulation and continuation seed, not merely a folder name. Replaying the same seeded stream does not create new operational identities. The operational content checksum covers telemetry and inspections; hidden truth does not affect ingestion identity or predictions.

Per-batch inserts, rejection records and ledger commit in one SQLite transaction with foreign keys enabled. Retry after interruption is safe: either the batch committed and is skipped, or it rolled back and can be retried. Each batch is atomic; earlier successfully ingested batches remain committed if a later batch fails. Missing columns/malformed payloads or SQLite errors roll back that batch. Row-level quality problems quarantine only affected rows and commit valid peers. Exact within-batch duplicates are removed and recorded in quarantine.

If content is changed under an already committed source/batch identity, ingestion fails rather than silently revising history. Quarantined rows are considered reviewed by this ingestion attempt and do not automatically retry; corrections need a new explicit input identity/workflow. Reports distinguish inserts, unchanged records, rejected rows and stored totals. Source row 2 corresponds to the first raw data record (CSV header is row 1).

Raw simulator checkpoints and original CSVs are unchanged by ingestion. ML models, RAG files and historical training feature exports are unchanged. Do not delete/reinitialize the simulator checkpoint against a database already extended with live rows; preserve it so generation continues from the proper state.

## Inspect storage

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/tyremind.db'); print('Telemetry rows:', c.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]); print('Ingested batches:', c.execute('SELECT COUNT(*) FROM ingestion_batches').fetchone()[0]); print('Rejected rows:', c.execute('SELECT COUNT(*) FROM ingestion_rejections').fetchone()[0]); c.close()"
```

Default history starts with 21,600 telemetry rows. With N valid daily batches and no rejected/duplicate readings, the count is `21,600 + 120 × N`. Existing historical rows are not removed. Worker status in `data/monitoring/worker_status.json` now records `operational_database_updated: true`, inserted counts and rejection counts for its last cycle.

No external SQLite server, API key or extra Python dependency is needed. The source simulator database and operational database are distinct local files.

## Verification and limits

```powershell
python -m unittest discover -s tests -v
```

The full suite has 78 tests when prior ML/RAG dependencies are available; Windows skips the POSIX signal test. New tests cover backlog import/replay, preservation of abnormal but valid readings, corrupted-value quarantine with original fields, cross-history mileage, key conflicts, schema-failure rollback, duplicate tracing, measured inspections, consistent late records and the full worker path.

The default development check imported four waiting batches (480 readings plus 120 inspections), then generated/ingested another 120 readings, resulting in 22,200 telemetry rows. A repeat ingestion inserted zero records and foreign key checks passed. Your totals depend on how many simulator cycles you have run.

**This step ends at validated storage.** Live feature calculation, saved-model inference, risk levels and investigation events remain later steps. The current importer scans stored batch identities/checksums for correctness and is appropriate for a local demo, not a high-volume distributed ingestion service.
