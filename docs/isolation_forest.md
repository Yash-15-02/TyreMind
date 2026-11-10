# Isolation Forest: operating-pattern anomaly detection

## Run on Windows

From the cloned project folder, run commands separately:

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.ml.anomaly
python -m unittest discover -s tests -v
```

Requires `data/features/tyre_features.csv`, matching `data/generated/evaluation_labels.csv`, and Step 5's `data/splits/split_report.json`. The input exports must belong to the same generated dataset. No RAG API, LLM or GPU is needed. The full suite has 55 tests with the ML and RAG dependency groups installed; RAG integration tests are skipped if its optional dependencies are absent.

## What it detects

This model flags unusual operating/wear patterns. It does not declare a tyre worn out, safe, unsafe or diagnose a physical failure. Predictors are pressure deviation, pressure change per day, rolling temperature mean/variability, rolling load mean, overload fraction and historical observed wear rate. Tyre age, raw mileage and absolute tread are deliberately excluded to avoid treating ordinary aging as an anomaly. Inputs come from the existing time-aware feature export.

The inference helper `predict_anomalies(artifact, features)` needs only those seven operational columns. Monitoring must also enforce the recorded warmup requirement (three available observations in the seven-day window). Early warmup rows are excluded from training/evaluation, not assigned a normal result.

## Normal reference and labels

The fitting algorithm is unsupervised: Isolation Forest fits only feature values. Synthetic labels are used externally to curate a normal reference and calibrate/evaluate its threshold, making the overall workflow normal-reference/label-assisted rather than fully unlabeled learning.

A fault label is `scenario_active OR overload_active`. A tyre assigned the normal scenario can experience truck overloading and is then abnormal. A worn-but-stable tread is not automatically a fault. Normal training rows must have no active injected fault or overloading in the trailing 28 days. This past-only washout reduces contamination from rolling histories after maintenance; it is a demo heuristic, not proof every wear interval is unaffected. Rows require at least three readings in the seven-day window. Simulator true tread and scenario labels never enter predictors.

## Time and truck separation

The same Step 5 truck groups and cutoffs are used. Normal-reference rows from the 14 training trucks before 2026-04-19 fit the model and median imputer. Validation covers Apr 19–May 24 on training trucks and the three reserved validation trucks. Final tests cover May 25 onward on familiar trucks and three held-out test trucks.

Anomaly detection uses current operating conditions, so no future 5,000-km target or target-horizon purging is needed. It therefore evaluates all eligible current readings, including the late-history rows that wear prediction cannot label. Default evaluation counts are 3,024 seen / 648 unseen rows in each validation and final test period.

## Calibration policy

Compare three 200-tree configurations (max_samples 256 or 512, max_features 1.0 or 0.8). This step uses a fixed validation period, not wear-model cross-validation. Each candidate fits on exactly the same normal training reference. The median imputer is fitted only on that reference. `contamination='auto'` is used for fitting, but its default predict threshold is not used.

Anomaly score = negative `score_samples`, so larger scores mean more unusual. It is not a probability. Validation-score quantiles provide a bounded threshold grid. Select the candidate/threshold with highest mean F2 across seen/unseen validation groups, subject to false-positive rate <=5% in each group. F2 gives recall more weight than precision; the false-alert constraint can still severely limit recall. These are configurable demonstration choices, not certified safety thresholds.

The selected fitted pipeline and threshold are retained together; they are not refitted before final testing because refitting could shift the score distribution and invalidate calibration. Final test results do not alter selection. Changing the design after inspecting these results requires fresh holdouts for a credible new final evaluation.

## Outputs and measured limits

`models/anomaly/` contains `isolation_forest.joblib`, `anomaly_report.json`, `metrics_summary.csv` and `anomaly_predictions.csv`. The artifact stores the imputer, forest, feature order, threshold and score direction. Reloaded inference was checked against the original. Outputs replace prior files and are ignored by Git. Only load trusted joblib files.

Metrics include precision, recall, F1/F2, accuracy, false-positive rate, specificity, average precision (PR summary), ROC-AUC and confusion counts. Assigned-scenario groups include inactive periods and may contain independent truck overloading; active-fault recall is also reported separately. Metrics are row-level, not distinct incidents or detection delay. Precision depends on prevalence, and the injected fault prevalence is not representative of a real fleet.

Default fit uses 6,211 normal-reference rows and excludes 240 warmup rows. Selected threshold is approximately 0.57931. Actual [test results](results/anomaly_metrics.csv):

| Final evaluation | Precision | Recall | F1 | False-positive rate | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Familiar trucks | 98.82% | 57.21% | 72.47% | 1.44% | 0.8927 |
| Held-out trucks | 80.65% | **7.72%** | 14.08% | 1.85% | 0.7507 |

This detector misses many injected faults, especially on held-out trucks. It is not suitable as the sole alert trigger. Its output is an experimental signal for the planned risk engine alongside independent deterministic checks; that combined engine is not implemented by this step. Low false-alert rate and passing software tests do not establish adequate detection performance. Full aggregate results, validation selections and active-fault recalls are in [anomaly_results.json](results/anomaly_results.json).
