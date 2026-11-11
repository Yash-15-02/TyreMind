# Phase 2, Step 4: current feature calculation

The default worker now includes saved-model inference (Step 5). Use `--features-only` to stop after this guide's feature stage; the standalone feature command remains unchanged. See [saved inference](saved_inference.md).

## Run on Windows

Stop a running worker with Ctrl+C before updating. From the cloned project folder, run commands individually:

```powershell
git pull
python -m pip install -e .
python -m tyremind.monitoring.features
```

This standalone command reads the already stored operational data and refreshes features only. It does not simulate another day or retrain a model.

To generate, validate, ingest and calculate features together once:

```powershell
python -m tyremind.monitoring.worker --run-once
```

Automatic cycles, every 60 seconds:

```powershell
python -m tyremind.monitoring.worker
```

Press Ctrl+C to stop. Expected default cycle: 120 new readings ingested, `feature_rows: 120`, `stale_feature_rows: 0`, mode `current_features_ready`. Your simulated date and batch number depend on previous runs. Timer-only checks still use `--heartbeat-only`.

## How features are calculated

The code reads telemetry, tyre metadata, measured inspections and maintenance events from SQLite and calls the exact `build_features()` function used during training. No feature formula is copied into a separate live implementation. It processes history, then keeps the latest feature row per tyre.

- `diff()` and elapsed time calculate pressure change per day.
- Actual trailing seven-day windows calculate pressure/temperature/load statistics and distance sums.
- `merge_asof(direction='backward')` attaches only inspections/maintenance available at a reading's timestamp.
- Two available inspections and associated historical mileage determine observed wear per 1,000 km.
- Observation count, age and history-availability flags preserve context.

All 19 numeric features remain in the training-defined order. Tyre/truck identifiers, timestamps, feature version and freshness fields are metadata, not additional model predictors. Missing values remain missing; SQLite stores them as NULL and the CSV leaves blanks. There is no live imputation, normalization, PCA, feature selection or model fitting in this step. Saved model pipelines will perform their own fitted preprocessing at inference time.

The full stored history is recomputed for correctness in this local MVP. New batch rows alone are insufficient for pressure trends, historical inspections and maintenance context. A larger deployment will need incremental/windowed computation.

## Time and freshness

The standalone command uses the newest stored telemetry timestamp as its cutoff. The worker uses the generated cycle's simulated timestamp. All telemetry, inspections and maintenance after the cutoff are filtered out before feature building. Exact-time measurements are assumed available at their recorded event timestamp, as in the original feature pipeline; real delayed records would require availability-time handling.

One row is produced for each tyre with observations by the cutoff. If a tyre's newest valid reading is older than the cutoff (for example its new reading was rejected), `is_stale=1`. Its old feature row is not mislabeled as fresh. If every new reading is rejected, the worker still reports the old timestamps/stale flags rather than fabricated current values. A future inference step must respect freshness and warmup before scoring.

The SQLite snapshot cannot move backward in cutoff time. Retrospective experiments should use the pure calculation function or a separate database, not overwrite current operational state. The V1 single-installation tyre lifecycle assumptions remain unchanged.

## Output

Default CSV files:

```text
data/monitoring/features/
├── current_features.csv
└── current_feature_report.json
```

The report records version 1.0, cutoff, computation timestamp, row count, stale count, per-feature missing counts and CSV checksum. Check command success before consuming output; if a filesystem export fails after the database commit, SQLite may be newer than an older CSV/report until refresh succeeds.

```powershell
explorer .\data\monitoring\features
```

SQLite: the `current_features` table in `data/tyremind.db` holds the latest snapshot (normally 120 rows). It includes foreign keys and computation/cutoff metadata. Snapshot replacement is transactional: if a refresh fails, the earlier snapshot remains. Repeated refreshes do not accumulate duplicate current rows. This derived table is a current view of the fleet, not a permanent feature-history archive.

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/tyremind.db'); print(c.execute('SELECT COUNT(*) FROM current_features').fetchone()[0]); c.close()"
```

The default `data/features/tyre_features.csv` used for training remains unchanged, as do saved models and raw source histories. Hidden simulation truth and labels are not queried. Raw operational tables are read without alteration by the feature refresh. Source ingestion happens earlier in the worker cycle.

Worker custom paths include `--source-database` and `--feature-output`; the standalone feature command supports `--database` and `--output`.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The suite now has 85 tests with prior ML/RAG dependencies installed; Windows skips the POSIX shutdown test. New tests independently verify pressure/distance formulas, idempotent snapshots, preservation of training/source data, exclusion of future evidence, stale handling after rejected readings, NULL inspection history, snapshot rollback and full cycle integration.

The actual development run generated/ingested a further 120 readings and refreshed 120 feature rows with zero stale records. **Saved-model inference, risk levels and agent triggers are not connected yet.**
