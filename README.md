# TyreMind AI

Automated tyre monitoring proof of concept. **Implemented: Phase 1, Steps 1–5 — historical generation, preprocessing, SQLite storage, features and training/evaluation splits.** Wear-model comparison is implemented below; agents and the dashboard are future steps.

## Run

Python 3.10 or newer:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1
python -m pip install -e .
tyremind-generate
```

Without installing the package, if NumPy and Pandas are already available:

```bash
PYTHONPATH=src python -m tyremind.data.generate
```

Defaults: 20 trucks, six positions each, 180 days starting 2026-01-01 UTC, seed 42. This produces 21,600 daily telemetry records. The date is fixed for reproducibility, not taken from the current day.

```bash
tyremind-generate --trucks 4 --days 60 --seed 7 --start-date 2026-01-01 --output data/custom
```

The same arguments reproduce the same data within the same dependency environment. Rerunning replaces files in the selected output directory. Generated datasets are excluded from Git by default at `data/generated/`; exclude any custom output directory before committing.

## Generated files

| File | Contents |
|---|---|
| `trucks.csv` | Vehicle identities |
| `tyres.csv` | Tyre identities, installation, position, reference pressure and initial tread |
| `telemetry.csv` | Daily pressure (kPa), temperature (°C), distance and cumulative mileage (km), load ratio |
| `inspections.csv` | Noisy measured tread depth (mm), every 14 days and at the final observation; also at repair |
| `maintenance_events.csv` | Simulated leak repairs and reinflation |
| `evaluation_labels.csv` | Assigned scenario, active flags and simulator true tread; evaluation only, never ML inputs |
| `manifest.json` | Configuration, generator version and row counts |

See [simulation assumptions](docs/simulation.md). No real fleet data, credentials or LLM API access are required.

## Checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests check reproducibility, identities, mileage, shared truck loads, declining tread, leaks, elevated temperatures and repair recovery.

## GitHub

Repository: https://github.com/Yash-15-02/TyreMind

Download with **Code → Download ZIP**, extract it, then follow the Run instructions above. Source and documentation are versioned; regenerate the full dataset locally with `tyremind-generate`. A small preview is available in `data/sample/`.

## Step 2: validate and preprocess (Windows, macOS or Linux)

From the folder containing `pyproject.toml`:

```bash
python -m pip install -e .
python -m tyremind.data.generate
python -m tyremind.data.preprocess
python -m unittest discover -s tests -v
```

If you already generated the historical data, skip the generator command. The preprocessing command reads `data/generated/` and writes `data/processed/clean/`, `data/processed/rejected/` and `data/processed/validation_report.json`. It never modifies the input CSVs. Rerunning replaces the processed outputs.

Custom directories:

```bash
python -m tyremind.data.preprocess --input data/generated --output data/processed
```

No missing readings are imputed. Missing or invalid required values are quarantined with their original source values and reasons. Exact normalized duplicates retain one row; conflicting identities/timestamps quarantine the entire conflicting group. Known fault readings are preserved: low pressure or high temperature alone is not a data-quality rejection. No certified safety limits are applied.

Timestamps become UTC and numeric values are explicitly converted. Checks include finite numbers, nonnegative distance/load/tread, positive reference pressure, known truck/tyre relationships, installation dates and nondecreasing mileage. This version assumes one installation per tyre and one tyre per vehicle position, matching Step 1. Tyre replacement lifecycle support is future work.

A missing required column or missing input file stops processing before any new output is written; existing output from an earlier successful run can remain. Rejected rows have `source_row` (CSV header is row 1) and `rejection_reasons`. A row may have several reasons. Additional columns are reported and excluded from clean output. Simulation evaluation labels are intentionally excluded from operational preprocessing and remain in the original generated directory.

Expected default result: all 21,600 telemetry rows accepted and zero rejected across the five operational tables. Rejected CSVs still contain headers when empty. The full test suite has 10 tests, including deliberately corrupted temporary inputs; these tests do not alter your dataset.

## Step 3: store clean records in SQLite

Update your cloned repository and install the new command:

```bash
git pull
python -m pip install -e .
python -m tyremind.database.load
```

This reads the five CSVs in `data/processed/clean/`. If they do not exist yet, run the generation and preprocessing commands from Steps 1–2 first.

The loader creates `data/tyremind.db` using Python's built-in `sqlite3` (no extra dependency or database server). It creates trucks, tyres, telemetry, inspections and maintenance_events tables, primary and foreign keys, nonnegative-value constraints and query indexes. All writes are inside one transaction. It revalidates input before importing; foreign keys are explicitly enabled on its connection. Every future database connection must also enable foreign keys.

```bash
python -m tyremind.database.load --input data/processed/clean --database data/tyremind.db
python -m unittest discover -s tests -v
```

Expected first import: 20 trucks, 120 tyres, 21,600 telemetry records, 1,699 inspections and 20 maintenance events. Importing identical records again inserts zero rows. Existing keys with changed values raise an error and roll back all writes instead of silently overwriting history. Existing records absent from the input are retained. This loader is for complete cleaned historical exports; incremental live ingestion and replacement lifecycles will be implemented separately.

The printed report includes input, inserted, unchanged and total counts for each table. Database row counts and foreign keys are checked before commit. The complete suite now has 14 tests, including persistence, repeat imports, constraint enforcement and rollback. Generated database files are excluded from Git.

To check the stored telemetry count without installing any SQLite extension:

```bash
python -c "import sqlite3; c=sqlite3.connect('data/tyremind.db'); print(c.execute('SELECT COUNT(*) FROM telemetry').fetchone()[0]); c.close()"
```

Expected: `21600`. Database schema changes will need migrations in future steps; this version only initializes new tables and does not migrate existing schemas.

## Step 4: point-in-time feature engineering

From the cloned project folder, run each command separately:

```bash
git pull
python -m pip install -e .
python -m tyremind.features.build
python -m unittest discover -s tests -v
```

Requires the Step 3 database. If starting fresh, run generation, preprocessing and database loading first. Step 4 opens the database read-only and exports `data/features/tyre_features.csv` plus `data/features/feature_report.json`. Default output: 21,600 rows, 120 tyres and 19 numeric features, plus identity/timestamp columns. It does not modify the database or train any models. The full suite now contains 20 tests.

See [feature definitions](docs/features.md) for formulas, availability assumptions and missing-history handling. No PCA, correlation filtering or imputation is performed. Model-specific feature selection and train-only preprocessing belong to later training steps.

## Step 5: prepare targets and split by time and truck

From the cloned folder, run these commands one at a time:

```bash
git pull
python -m pip install -e .
python -m tyremind.ml.split
python -m unittest discover -s tests -v
```

Requires Step 4's `data/features/tyre_features.csv` and Step 1's matching `data/generated/evaluation_labels.csv`. Do not mix files generated with different settings/seeds. The command prepares synthetic tread-depth targets at exactly another 5,000 km and creates training, validation and final testing datasets. No model is trained, and no imputation/scaling is fitted. The suite now contains 26 tests.

Results are in `data/splits/`: `train.csv`, `validation_seen.csv`, `validation_unseen.csv`, `test_seen.csv`, `test_unseen.csv`, `excluded.csv`, and `split_report.json`. See [split design](docs/splits.md) for boundaries, counts, exclusions and target definitions. Output files are regenerated on rerun and excluded from Git.

## Collect technical documents for RAG

Three manufacturer reference PDFs are enabled and download-verified; FMCSA is disabled at the user's request. To collect them locally:

```bash
git pull
python -m pip install -e .
python -m tyremind.rag.collect
python -m unittest discover -s tests -v
```

Expected collection summary: `Downloaded 3; failed 0.` Output: `data/knowledge/raw/` with PDFs and a collection manifest. The source catalog and usage notes are in [knowledge/README.md](knowledge/README.md) and [knowledge/sources.json](knowledge/sources.json). Publisher originals are not committed to Git. No embeddings or ChromaDB index are built yet. The complete test suite now has 33 tests.

## Split technical text and preserve source references

```bash
git pull
python -m pip install -e .
python -m tyremind.rag.chunk
python -m unittest discover -s tests -v
```

Requires the three manufacturer PDFs downloaded earlier. Output: `data/knowledge/processed/pages.jsonl`, `chunks.jsonl` and `chunk_report.json`. The default run extracts 224 pages and produces 82 selected passages. Each passage preserves its title, publisher, edition, URL, PDF page, section, file checksum and text offsets. Full source text remains local, outside Git. FMCSA is skipped; unverified inflation tables stay outside retrieval chunks. Read [the chunking guide](docs/chunking.md) for page selection and limitations. No embeddings or vector index exist yet.

## Generate embeddings

```powershell
git pull
python -m pip install -e ".[rag]"
python -m tyremind.rag.embed
python -m unittest discover -s tests -v
```

This encodes the existing source-linked passages locally with sentence-transformers/all-MiniLM-L6-v2. The first run downloads model weights; no API key or GPU is needed. Default result: 82 vectors with 384 values each, saved alongside source metadata under `data/knowledge/embeddings/`. Token-limit checks avoid silent passage truncation. The full suite has 38 tests. See [embedding instructions](docs/embeddings.md). ChromaDB indexing is the next step.

## Store embeddings in local ChromaDB

```powershell
git pull
python -m pip install -e ".[rag]"
python -m tyremind.rag.store
python -m unittest discover -s tests -v
```

No ChromaDB account/API key is needed. The importer reuses your existing embeddings and saves 82 passage/vector/source records in `data/knowledge/chroma/`. Rerunning the same import adds zero duplicates. `active_index.json` identifies the current verified collection. All 42 tests pass with the optional dependencies installed. See [ChromaDB instructions](docs/chromadb.md). General retrieval and agent tools remain later work.

## Train and compare Random Forest and XGBoost

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.ml.train
python -m unittest discover -s tests -v
```

This compares three configurations per model with three purged time/group folds, then saves both fitted wear-prediction pipelines and metrics in `models/wear/`. No final-test data is used for selection. Real measured results: XGBoost CV MAE 0.1999 mm; Random Forest 0.2056 mm; the simpler wear-rate baseline performs better at 0.1826 mm. These are synthetic-data results. Read [training details](docs/wear_training.md) and [measured comparison](docs/wear_model_results.md). The full suite has 48 tests with ML and RAG dependencies installed. Isolation Forest is implemented in the anomaly-detection section below.

## Train Isolation Forest anomaly detection

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.ml.anomaly
python -m unittest discover -s tests -v
```

The model fits seven operating/wear features on normal-reference history, calibrates its threshold using validation data only, and reports precision/recall/F1/F2, false-alert rates and held-out-truck performance. Saved pipeline and reports are in `models/anomaly/`. All 55 tests pass with optional dependencies installed. **Measured held-out recall is only 7.72% at the selected threshold; this experimental model must not be the sole alert gate.** See [design, commands and measured results](docs/isolation_forest.md). This does not mark worn-out tyres or implement the scheduled risk engine.

## Final check: reload, evaluate and inventory saved models

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.ml.finalize
python -m unittest discover -s tests -v
```

Requires both training commands to have completed locally. This command reloads the three saved models, verifies unchanged input exports and reproduces their recorded held-out metrics. It does not retrain or retune. Expected: 3 verified models; outputs in `models/evaluation/`. The complete test suite now has 57 tests with optional dependencies installed. See [final model inventory and commands](docs/model_finalization.md).
