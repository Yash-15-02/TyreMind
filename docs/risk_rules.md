# Phase 2, Step 6: deterministic risk rules

Normal/Medium continuation decisions are now explicitly persisted in `monitoring_decisions`; High/Critical remain pending the later event handler. See [continuation branch](continue_monitoring.md).

## Run on Windows

Stop a running worker with Ctrl+C, then run commands individually from the repository root:

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.monitoring.risk
python -m unittest discover -s tests -v
```

The standalone risk command evaluates existing current features/readings and matching predictions. It does not generate a day, fit models or call an LLM. If features are outdated, refresh them; run `python -m tyremind.monitoring.predict` if current prediction evidence is missing. Direct pressure/temperature rules still work without ML, but incomplete evidence cannot produce a confident Normal result.

Full automatic cycle, once:

```powershell
python -m tyremind.monitoring.worker --run-once
```

Then `python -m tyremind.monitoring.worker` runs the complete pipeline every 60 seconds. Press Ctrl+C to stop. RF remains the default wear model for this requested workflow; `--wear-model xgboost` is optional. For inference without risk classification use `--prediction-only`; earlier `--features-only` and `--heartbeat-only` modes remain.

## Configuration and exact rules

Rules are Python conditions configured in `config/risk_rules.json` (demo-1.0). None are manufacturer-approved operating/safety/service thresholds. Pressure readings are not distinguished as hot/cold, so these demonstration deviations must not be represented as validated inflation assessments.

| Evidence | Medium | High | Critical |
|---|---|---|---|
| Pressure below its reference | >=10%, <20% | >=20%, <30% | >=30% |
| Current temperature | >=55°C, <65°C | >=65°C, <80°C | >=80°C |
| Fresh measured tread | — | <=3 mm, >1.6 mm | <=1.6 mm |
| Predicted tread after 5,000 km | — | <=3 mm (finite, nonnegative prediction) | Never from prediction alone |
| Sustained pressure decline | — | Three consecutive decreases across four readings, >=15 kPa total drop, within seven elapsed days | — |
| Seven-day average load ratio | >1.10 | — | — |
| IF anomaly flag alone | Medium | — | — |
| IF flag plus pressure/temperature/load concern | — | High | Existing Critical still wins |

A measured tread inspection is considered fresh only at age 0–14 simulated days. Older/missing measurements are not treated as current low tread. A negative future tread prediction is flagged as invalid evidence, not silently clipped. Rule boundaries are inclusive except the load condition (strictly greater than 1.10). The highest triggered level always wins; an IF result of zero never cancels independent rules.

The pressure decline test is a separate rule on ordered historical readings, not a learned model. Numerical settings can be edited in the config file; invalid ordering/types fail before assessment. The policy content hash and full JSON are persisted alongside each assessment, so changes are traceable even if the version string was not bumped. Use a new version label when changing a policy intentionally.

## Data sufficiency and meanings

- Fresh evidence with no triggered rule and all required evidence present: Normal / complete. This means no configured rule triggered, not a declaration of safety.
- Stale latest reading: no assigned risk level, data_status stale. Old Critical evidence is not relabeled as a fresh reading. Historical assessments remain stored.
- Missing/old evidence with no alerting condition: no assigned level, data_status insufficient_data.
- An independent rule triggers despite other missing evidence: preserve that level and data_status partial. Missing ML never suppresses a direct Critical pressure/temperature rule.
- Full evidence with a trigger: its highest level / complete.

Prediction evidence must match the tyre, reading time, feature cutoff/version and exact feature values. In the worker it must also match the two loaded artifact hashes. Standalone mode selects the latest matching stored prediction for its requested wear family. NULL skipped predictions remain NULL, not a normal anomaly result. Numerical anomaly scores are not probabilities or certified diagnosis; the detector's held-out recall remains poor.

## Storage and evidence

Latest export:

```text
data/monitoring/risk/
├── latest_risk.csv
└── risk_report.json
```

```powershell
explorer .\data\monitoring\risk
```

SQLite table `risk_assessments` retains assessment history, level/status, rule IDs and values/thresholds, data issues, complete evidence snapshot, policy version/hash/JSON and matching prediction ID. Raw input measurements, measured tread and predicted tread are distinguished in evidence. No hidden simulation labels/truth are queried.

Assessment ID hashes evidence and policy. Identical reruns create no duplicates; changed evidence or policy creates another assessment. The complete fleet write is transactional, so failed writes roll back the attempt. Exports happen after commit; a failed export can leave prior CSV files until rerun. There is no event/work-order deduplication in this step, because no investigation events or work orders are created yet.

The worker status reports `risk_counts`, `unclassified_risk_rows` and mode `risk_levels_ready`. Actual counts vary with simulated history; a verified development cycle produced 56 Normal, 3 Medium, 60 High and 1 Critical.

## Verification and next step

The suite has 107 tests with ML/RAG extras installed (Windows skips the POSIX signal test). Fifteen risk-specific tests cover normal completeness, pressure boundaries, priority, IF independence, combined evidence, sustained decline, stale/old/missing data, observed versus predicted tread, invalid policy, persistence/replay, SQL rollback and NULL skipped predictions.

Risk tests check programmed behavior; they do not validate the suitability of thresholds for real tyres. **High/Critical event creation, repeated-event deduplication and agent invocation are the next steps.**
