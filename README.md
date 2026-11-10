# TyreMind AI

Automated tyre monitoring proof of concept. **Implemented: Phase 1, Steps 1–2 — historical generation and Pandas preprocessing.** Database creation, ML, agents and the dashboard are future steps.

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
