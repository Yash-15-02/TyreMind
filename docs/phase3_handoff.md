# Phase 2 final step: publish actionable events to the Phase 3 inbox

This is a local durable handoff, not a call to a running LangGraph service. Phase 3 agents remain unstarted. No API key, external account, broker or LLM call is needed for delivery preparation.

## Run on Windows

Stop the worker before updating. Run one command at a time from the cloned project root:

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.monitoring.handoff
python -m unittest discover -s tests -v
```

The standalone handoff command publishes existing PENDING investigation jobs. It does not generate new readings, classify risk, create new events or claim jobs. If jobs have not been queued yet, first run `python -m tyremind.monitoring.events` after risk assessment.

Open the inbox:

```powershell
explorer .\data\phase3
```

Files:

```text
data/phase3/
├── inbox.json
└── handoff_report.json
```

The default worker now performs generation → ingestion → features → inference → risk → event queue → handoff preparation. `--events-only` stops after queueing, preserving Step 7 behavior. Existing earlier stage flags remain. Full once-only cycle:

```powershell
python -m tyremind.monitoring.worker --run-once
```

For continuous monitoring, omit `--run-once`; Ctrl+C stops it. The inbox/state remain saved for tomorrow's Phase 3 work.

## Payload contract

`inbox.json` has schema_version 1.0, a ready_count and a messages array. Each message contains:

| Field | Contents |
|---|---|
| handoff_id | Deterministic identity for this payload snapshot |
| message_type | INVESTIGATE_TYRE |
| investigation | Job ID, event ID, target revision, requested assessment ID and High/Critical priority |
| event | Truck/tyre IDs, current/peak severity, first/last seen, open status and revision |
| assessment | Source timestamp/cutoff, risk/data status, policy version/hash and prediction reference |
| triggered_rules | Rule IDs, priorities, measurements and thresholds from the actionable assessment |
| data_issues | Missing/partial evidence to preserve uncertainty |
| evidence | Operational readings, trends and saved-model results captured by the assessment |

Only unresolved events with PENDING jobs are included. A queued case may have since shown recovery; the payload retains both its current event state and the earlier actionable snapshot because human review is still required. It does not falsely label an older High reading as a new current measurement. No hidden simulator truth or claim token is exported.

## Durability and duplicates

SQLite table `phase3_handoffs` stores the structured payload, checksum, revision and transport state. There is one current handoff per investigation ID. An unchanged rerun creates no duplicate. Coalesced evidence/revisions update the unclaimed payload in place; its handoff ID changes, so a consumer can reject an outdated copy.

Transport states are READY, CLAIMED, COMPLETED, FAILED or CANCELLED. Actual job status in `investigations` remains authoritative. Publishing does not turn PENDING into RUNNING. Existing result/claim hooks reconcile handoff status. Claimed/completed/failed jobs are not exported as READY again; explicit retry or a follow-up is a different job.

Database preparation is transactional. The inbox JSON is replaced atomically afterward. If a file export fails, rerunning regenerates it from the committed database without creating another job/handoff. The report includes the inbox checksum. An exported file is a snapshot and may become old; the consumer must verify/claim against SQLite before execution, not blindly process every file message.

## Phase 3 consumer hook

`claim_phase3_handoff(database, investigation_id, handoff_id)` verifies READY/PENDING state, payload checksum, unchanged event/job snapshot and revision. It atomically claims the investigation and returns an ownership token with its payload. Concurrent consumers cannot claim the same job. Outdated envelope IDs/snapshots are rejected and require refresh.

Phase 3 Step 1 now provides a [receiver](receive_event.md) using the same atomic claim checks. It saves an initial investigation state and resumes it after restart. LangGraph, LLM calls, agent tools and investigation reports remain subsequent steps. There is no automatic claim lease; keep the consumer ID stable to resume its saved receipt.

## Verification

The full suite has 130 tests with ML/RAG extras installed; Windows skips the POSIX signal test. Six new tests verify source/evidence payloads, no automatic claims, repeat/updated delivery, atomic claims, completion handling, stale-snapshot rejection, export-failure recovery and empty inboxes for non-actionable data. Claims/completions in tests use temporary fixtures, not the actual pending jobs.

A real preparation run published 62 READY payloads while leaving all jobs PENDING and agents_executed false. A repeat created zero duplicates. Your count depends on your fleet's queued cases; zero READY messages is valid when there is no pending work.

**Phase 2 ends here. Phase 3 will execute the queued investigations.**
