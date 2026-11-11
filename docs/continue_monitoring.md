# Normal/Medium branch: save results and continue monitoring

The condition after risk assessment is now explicitly recorded:

```text
Risk is Normal or Medium?
    Yes → Save the risk/evidence and CONTINUE_MONITORING decision
           → Return from the cycle normally
           → APScheduler triggers the next cycle at its regular interval
    No  → High/Critical: AWAIT_EVENT_HANDLER (not implemented yet)
          Unclassified/stale: DATA_REVIEW_REQUIRED, continue collecting readings
```

A result other than High/Critical is not automatically Normal: missing/stale evidence remains unclassified and retains its review requirement. Medium with partial evidence still continues monitoring but also records `data_review_required=1`. All levels keep the monitoring loop active; this step never shuts the worker down because of a risk level.

## Run on Windows

Stop an old worker before updating, then run commands individually from the repository root:

```powershell
git pull
python -m tyremind.monitoring.risk
python -m unittest discover -s tests -v
```

No dependency change or model retraining is needed. The standalone risk command records decisions for existing data and exits; it does not start a scheduler. For automatic ongoing monitoring:

```powershell
python -m tyremind.monitoring.worker
```

Press Ctrl+C to stop. Explicit `--max-cycles` limits still stop a test worker normally.

## What gets saved

Existing source readings, feature and prediction results, and risk assessments are retained. New SQLite table `monitoring_decisions` links each decision to its risk assessment and stores:

- `next_action`: CONTINUE_MONITORING, AWAIT_EVENT_HANDLER or DATA_REVIEW_REQUIRED.
- `monitoring_continues`: 1, meaning the policy permits continued scheduled collection, not proof that the process is running.
- `data_review_required`: whether the assessment has incomplete/stale evidence.
- Human-readable reason and recorded-at timestamp.

A decision and its assessment commit together. Identical reruns create no duplicate decision. Existing assessments receive their missing decision when reevaluated after this update. Failed transactions roll back both; prior history remains intact.

`data/monitoring/risk/latest_risk.csv` now includes the decision fields. `risk_report.json` and worker status include continue/pending/review counts. These are dispositions and saved records; no email, notification, work order, investigation or LLM action is performed here.

```powershell
explorer .\data\monitoring\risk
```

## Verification

The full suite has 111 tests with ML/RAG dependencies installed (one POSIX signal test is skipped on Windows). Four new tests verify the Normal/Medium continuation branch, separation of High/Critical, preservation of review flags, persisted/replay-safe decisions and absence of investigation tables.

The development fleet produced 59 continuation and 61 pending-handler decisions, with monitoring still permitted. Counts vary as simulation advances. High/Critical event creation and repeated-event deduplication remain the next step.
