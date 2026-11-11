# Phase 2, Step 2: further simulated tyre readings

## What happens

Step 3 now connects validation/operational ingestion to the default worker. This guide describes standalone simulation/raw output; see [live ingestion](live_ingestion.md) for the current worker behavior.

The worker now continues the existing simulated fleet, advancing one simulated day per cycle. It keeps truck/tyre identities, reference pressures, accumulated mileage, remaining hidden tread, wear coefficients and fault scenario timelines. It adds a new day's distance and operating readings with small random variation, then calculates further tread loss using the Phase 1 demonstration equation. This is simulated evidence, not readings predicted by the ML models.

For the default history ending June 29, 2026, batch 1 is June 30 and batch 2 July 1. Each contains 120 readings. Every 60 wall-clock seconds advances one simulated day. Short test intervals advance simulated days faster; there is no claim these represent minute-frequency real sensors.

## Run on Windows

From the cloned project root, one command at a time:

```powershell
git pull
python -m pip install -e .
python -m tyremind.monitoring.worker --run-once
```

Expected first run: simulated date June 30, batch ID 1, 120 generated readings. If already run, the batch/date continue from saved state rather than restarting. Inspect the folder:

```powershell
explorer .\data\simulator\batches
```

Check automatic scheduling with three further batches, five seconds apart:

```powershell
python -m tyremind.monitoring.worker --interval-seconds 5 --max-cycles 3
```

Then normal scheduling:

```powershell
python -m tyremind.monitoring.worker
```

Press Ctrl+C to stop. The same saved simulation resumes next time. Run only one worker for this demo. The old scheduler-only check remains available with `--heartbeat-only`.

## Initialization and reproducibility

The first run reads the historical operational SQLite database, generator manifest and simulation labels. The Phase 1 generator can optionally expose its hidden final state without changing its default output. Initialization replays the historical generator to recover the exact wear parameters and scenarios, verifies matching truck/tyre/telemetry records and labels, then checkpoints final state. This replay is internal bootstrap work; it does not overwrite historical CSVs or insert duplicate history. It requires matching Phase 1 files and an operational database containing that original historical snapshot.

The live simulator starts a separate seeded random stream (default seed 43). It is not a claim that new readings equal an earlier 181-day regeneration; historical random draws were organized differently. The new RNG state is checkpointed after every batch, so restarting produces the same continuation as uninterrupted execution. Initialization is only needed once. Source paths and seed are pinned to a checkpoint; changing them requires a separate simulator output folder. Do not delete state casually or regenerate a different history over the existing experiment.

## Physical assumptions and boundaries

- Each truck gets a daily distance and base load; all six tyre positions share these values.
- Pressure remains near its reference for normal tyres; existing leak timelines continue and pressure loss remains capped at 210 kPa, as in Phase 1.
- Operating temperature varies with simulated ambient conditions, load, pressure loss and the elevated-temperature scenario.
- Wear increases with distance and the configured pressure/load/temperature/fault multipliers. True remaining tread cannot increase or fall below zero.
- Existing repaired leak scenarios remain repaired. No new maintenance decisions, tyre replacements or arbitrary new faults are introduced by this step.
- Tread measurements are emitted every 14 simulated days, with inspection noise. New inspection IDs are prefixed `LIVE-INSP` and do not collide with historical IDs.
- When true tread reaches zero, it stays zero. The simulator does not authorize continued operation or auto-replace a tyre. Later risk rules/human approval must interpret inspection evidence. These are demo coefficients, not validated wear/safety physics.

## Saved state and raw outputs

```text
data/simulator/
├── simulator.db
└── batches/
    ├── batch-000001/
    │   ├── telemetry.csv
    │   ├── inspections.csv
    │   └── simulation_truth.csv
    └── batch-000002/ ...
```

`simulator.db` is a private checkpoint/raw-batch queue, separate from the application's operational database. State and each raw batch commit together in one SQLite transaction. CSV export happens afterward; if export fails, the next call retries that same committed batch before advancing another day. This prevents lost batches or skipped simulated days from a failed export. Export markers mean CSV output is complete, not that operational ingestion happened. A future consumer must still deduplicate ingestion by tyre/timestamp or batch identity.

`simulation_truth.csv` contains hidden tread/scenario values for evaluation only. It must never become an ML input. `telemetry.csv` has only operational measurements; measured tread lives in `inspections.csv`. Empty inspection batches contain headers. Hidden truth and inspections are distinct.

All generated outputs are ignored by Git. The original historical CSVs, operational `data/tyremind.db`, ML artifacts and RAG data remain unchanged. **This step does not validate/ingest new batches into the operational database, engineer live features, run ML or trigger risk events.** Those are the next Phase 2 steps. Worker logs/status now report generated reading counts rather than falsely claiming they were analyzed.

## Standalone command and checks

```powershell
python -m tyremind.monitoring.simulator
python -m unittest discover -s tests -v
```

The standalone command also creates exactly one further batch using the same state; it advances the simulation and is not a read-only preview.

Custom source/output settings are supported on the worker: `--source-database`, `--history`, `--simulator-output`, `--simulation-seed`. The standalone simulator uses `--source-database`, `--history`, `--output`, `--seed`.

The full suite has 70 tests with ML/RAG dependencies installed (one POSIX-only test is skipped on Windows). New checks cover identity/time/mileage continuity, reproducibility across restarts, shared truck values, declining true tread, separation of truth, retrying a failed export, incompatible seeds/histories, and actual worker-to-simulator integration.
