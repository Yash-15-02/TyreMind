# Phase 3 Step 1: receive an actionable event

Run from the cloned project root. Stop the monitoring worker with Ctrl+C before updating code.

```powershell
git pull
python -m pip install -e ".[agents]"
python -m tyremind.agents.receive
```

The receiver uses the existing local `data/tyremind.db`. Nothing needs to be retrained. Phase 2 must have queued an actionable investigation; otherwise `NO_WORK` is expected. The Phase 2 worker still publishes events only; reception is a separate command at this step.

```text
Phase 2 pending High/Critical investigation
                 ↓
Refresh the prepared handoff from SQLite
                 ↓
Validate schema, source IDs, revision and checksum
                 ↓
Atomically claim one job, prioritizing Critical
                 ↓
Save initial investigation state in SQLite
                 ↓
Export readable JSON for the next Phase 3 step
```

Pydantic validates this message contract. Pandas preprocessing remains unchanged. No LLM, LangGraph execution, diagnosis or maintenance recommendation runs yet; no API key is needed.

SQLite `phase3_receipts` stores the initial state, ownership token and checksum. The token stays in the database. Readable exports are `data/phase3/received/<investigation_id>.json` and `receive_report.json`. The source handoff preserves readings, feature evidence, saved-model outputs, triggered rules and document-ready references where present. Analysis/result fields start empty.

The first call returns `RECEIVED`. Repeating with the default consumer ID returns `RESUMED` for the same job. The investigation's queue status is `RUNNING` to indicate ownership, while its actual analysis stage is `RECEIVED`. This does not claim that agents have investigated it. Keep the consumer ID unchanged; a different ID represents another consumer and may claim another job. There is no timed lease or automatic reassignment in this step.

Claim and checkpoint commit in one transaction. If exporting JSON fails, the next call resumes the saved receipt and rewrites the file. Invalid input rolls back the claim. Once later steps finish the investigation through the existing completion hook, the consumer can receive its next job. A receipt records the original evidence snapshot even if monitoring subsequently updates the case.

To inspect the output:

```powershell
Get-Content .\data\phase3\received\receive_report.json
explorer .\data\phase3\received
python -m unittest discover -s tests -p test_receive.py -v
```

Next: [start the LangGraph supervisor](supervisor.md) using the same consumer and received event. The receipt remains the immutable starting snapshot; the supervisor has a separate saved state.
