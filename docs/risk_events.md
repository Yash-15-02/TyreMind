# Phase 2, Step 7: risk cases and deduplicated investigation queue

The default worker now also publishes the Phase 3 inbox after queueing. Use `--events-only` to stop at this guide's endpoint; the standalone event handler remains unchanged. See [Phase 3 handoff](phase3_handoff.md).

## Run on Windows

Stop the worker with Ctrl+C before pulling. From the cloned repository root, run one command at a time:

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.monitoring.events
python -m unittest discover -s tests -v
```

The standalone event handler consumes previously unhandled risk assessments already in SQLite. It does not generate readings, retrain, rerun ML, call an LLM or execute agents. Earlier risk history is drained in simulated-time order, so the first run may create many cases. A repeat without new assessments reports zero processed assessments/events/jobs.

The default worker now runs the full Phase 2 path through event routing:

```powershell
python -m tyremind.monitoring.worker --run-once
python -m tyremind.monitoring.worker
```

Run the second command only when continuous operation is desired; Ctrl+C stops it. `--risk-only` preserves the prior behavior through risk assessment without consuming the event queue. Earlier `--prediction-only`, `--features-only` and `--heartbeat-only` modes remain.

## Case behavior

- High/Critical creates one OPEN event if this tyre has no unresolved event. Event ID is distinct from truck/tyre IDs.
- A SQLite partial unique index enforces one OPEN event per tyre. Different rule concerns for the same tyre are grouped into that case in V1.
- Repeated High/Critical updates last seen, latest assessment and history; it does not open another case or job every minute.
- Event history references immutable risk assessments (measurements, model evidence and policy are retained there).
- Each assessment is routed once, with primary-key protection against repeated attachment. Normal/Medium without a case is recorded as continuation; unknown/stale evidence is data review, not a new case.
- Upward severity or a newly observed rule increments the event revision. Highest severity is retained even if current conditions improve.
- A complete Normal/Medium assessment can record recovery, but does not resolve the case. Unknown/stale data never counts as recovery.
- Late evidence is attached without rewinding current/last-seen state. Older actionable evidence can move first-seen earlier; a previously unseen historical concern/peak can still request investigation.

The original `monitoring_decisions` row records the pre-handler disposition, so its AWAIT_EVENT_HANDLER value remains historical. Actual dispatch progress is recorded in `event_routing`, `risk_events` and `investigations`.

## Investigation queue and lifecycle

```text
PENDING → RUNNING → COMPLETED
                  ↘ FAILED
```

A second partial unique index permits at most one PENDING/RUNNING investigation per event.

- A PENDING job is updated with the latest actionable assessment, priority and revision instead of duplicated.
- A RUNNING job's claimed snapshot is not changed. New material evidence increments the event revision. When the running job finishes, at most one follow-up job is queued for that newer revision.
- COMPLETED reports are not requeued for unchanged conditions.
- A FAILED job does not retry every monitoring cycle. A future consumer can explicitly retry it; repeated retry calls return the same retry job. New material evidence can independently justify a new job.
- Pending Critical cases are claimed before High cases. Claiming is atomic and returns an ownership token; a different/stale token cannot complete a job.

`claim_investigation`, `finish_investigation` and `retry_failed_investigation` are Python hooks for the future Phase 3 consumer. They are not automatically called by monitoring. Jobs therefore stay PENDING until a consumer is connected. There is no fake agent execution or fake completed report.

This local queue does not yet implement claim leases/timeouts or recovery of a consumer that crashes while RUNNING. Those belong to Phase 3 execution management. Monitoring continues and newer concerns are retained while a job runs.

## Human resolution

The Python hook `resolve_event(database, event_id, reviewed_by, reason)` requires a named reviewer and reason, an OPEN event and no active investigation. No automatic worker calls it. A later fresh High/Critical assessment can create a new case after explicit resolution; old evidence is not replayed into a new case. This is an audit requirement in a local demo, not production authentication or an implemented approval UI.

## Storage and files

SQLite `data/tyremind.db` gains:

| Table | Contents |
|---|---|
| risk_events | Current/peak severity, timestamps, revision, concerns and resolution fields |
| event_routing | One disposition for each consumed assessment, including continuation/data review |
| event_assessments | Attached evidence history, escalation/new-rule/recovery indicators |
| investigations | Durable pending/running/completed/failed jobs, snapshot reference and result/error |

New tables/indexes are additive. Event creation/update, assessment routing and queue changes commit atomically. A failed queue write rolls back the whole routing attempt. Each run serializes writes using SQLite transactions. This is a local MVP, not a distributed broker.

Exports in `data/monitoring/events/`:

- `open_events.csv`: unresolved cases.
- `investigation_queue.csv`: job history/status, excluding claim tokens.
- `event_report.json`: new/updated/processed counts, open/pending/running totals, `agents_executed: false`.

Exports happen after database commit. If export fails, retrying the command reconstructs files without duplicating committed cases/jobs. Closing the worker does not erase queue state. Raw histories, trained artifacts and RAG data are unchanged.

```powershell
explorer .\data\monitoring\events
```

The worker reports mode `risk_events_ready`, open event count and pending investigation count. A backlog check created 61 cases and 61 jobs; rerunning created none. Actual counts vary with history/policy/simulation progression.

## Verification and next step

The full suite has 124 tests with ML/RAG extras installed; Windows skips the POSIX signal test. Thirteen new tests cover continuation/no-event cases, replay, persistent High, escalation, running-job follow-up, unchanged completed work, recovery, human resolution/new episode, explicit retry, claim ownership, concurrent claims, late evidence, rollback and Critical queue priority.

**This completes event creation/update and queueing, not investigations themselves.** Phase 3 will connect LangGraph tools/agents to claim jobs, retrieve RAG evidence and persist structured findings. No alerts to external services or work orders are sent/created by this step.
