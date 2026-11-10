# TyreMind AI

Automated tyre monitoring proof of concept. **Implemented: Phase 1, Step 1 — historical synthetic data generation only.** Preprocessing, database creation, ML, agents and the dashboard are future steps.

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
