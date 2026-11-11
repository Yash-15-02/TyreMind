# Phase 2, Step 1: standalone scheduled worker

## What is implemented

A separate Python program starts APScheduler 3.x and triggers `run_monitoring_cycle()` every 60 seconds by default. This first step runs a scheduler heartbeat only: it records a successful trigger and explicitly reports zero tyres processed. Telemetry generation, validation/storage, inference and risk events will be connected in later steps. Existing datasets, databases and saved models are unchanged.

The interval applies to one fleet cycle, not a 60-second inspection of one tyre. The planned cycle processes the fleet batch. Scheduler interval is wall-clock time, independent of simulated fleet time, which has not yet been implemented.

## Run on Windows

From the cloned repository root, run one command at a time:

```powershell
git pull
python -m pip install -e .
python -m tyremind.monitoring.worker
```

Keep the terminal open. The first cycle runs after 60 seconds, then on the interval schedule. Press Ctrl+C to request shutdown; an in-progress cycle is allowed to finish. No browser, dashboard, API key, retraining or GPU is required. Closing the terminal forcibly or shutting down the computer stops the process and may prevent a final status write. This is not an OS service.

Quick single-cycle check:

```powershell
python -m tyremind.monitoring.worker --run-once
```

Quick scheduler check (three ticks, then automatic exit):

```powershell
python -m tyremind.monitoring.worker --interval-seconds 5 --max-cycles 3
```

These test options do not change the normal 60-second default. Short intervals are for checking the timer, not production settings.

## Files and status

Local output is ignored by Git:

```text
data/monitoring/
├── worker.log
└── worker_status.json
```

Logs go to both the terminal and a rotating file (5 MB, up to three backups). Status records UTC timestamps, run ID, process ID, interval, cycle attempts/successes/failures, last duration, result/error and running/stopped state. Counters reset each worker run; old log entries remain until rotation. The JSON status is replaced atomically so readers do not see partially written JSON. A future dashboard must inspect timestamps/process liveness; a leftover file alone does not prove the worker is still running.

```powershell
Get-Content .\data\monitoring\worker_status.json
Get-Content .\data\monitoring\worker.log -Tail 20
```

Run only one worker against the same output folder. This step prevents overlapping jobs within one worker, not duplicate programs on separate processes/machines. Cross-process ownership and durable telemetry idempotency will need to be handled as ingestion is connected.

## Scheduling and failure behavior

- `max_instances=1` and an in-process lock prevent overlapping cycles.
- `coalesce=True` collapses accumulated overdue triggers; `misfire_grace_time=30` limits how late a missed scheduled execution may start.
- If a cycle is still running at the next interval, the overlapping trigger is skipped; the worker does not build an unbounded backlog.
- Job exceptions are logged, failure counts are recorded and later scheduled cycles remain enabled. There is no immediate retry loop.
- SIGINT/Ctrl+C and SIGTERM request shutdown. A currently running Python function must complete; no forceful cancellation is attempted.
- A finite run that experienced any cycle failures exits nonzero. Immediate `--run-once` failures also exit nonzero.
- Closing the program stops scheduling. Restarts create a fresh interval schedule, without replaying downtime.

The cycle callback is a separate function so later work can connect the complete monitoring pipeline without moving the scheduler into the dashboard. This step does not claim to monitor tyres yet.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The suite contains 63 tests with optional ML/RAG dependencies installed. Windows skips the POSIX SIGTERM test; Ctrl+C behavior should be checked in the terminal. New tests cover persisted heartbeat results, invalid intervals, concurrent-call rejection, recovery after a failed cycle, actual timed scheduler ticks/automatic stopping and POSIX signal shutdown. The injected-failure test deliberately logs an error; the test passes when failure accounting and the following cycle work correctly.
