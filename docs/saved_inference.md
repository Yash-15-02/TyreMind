# Phase 2, Step 5: run saved RF and IF models

The default worker now also evaluates risk rules after inference (Step 6). Use `--prediction-only` for inference without risk classification; the standalone prediction command is unchanged. See [risk rules](risk_rules.md).

## Run on Windows

Stop the worker with Ctrl+C before pulling. From the cloned project root:

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.monitoring.predict
python -m unittest discover -s tests -v
```

This standalone command reads `current_features` and runs the saved models without generating another day, retraining, or fitting preprocessing. If telemetry is newer than the snapshot, first run `python -m tyremind.monitoring.features`. Models must already exist locally from the earlier training commands. Missing artifacts fail before simulation advances in the worker. Load only trusted joblib artifacts.

To run one complete cycle:

```powershell
python -m tyremind.monitoring.worker --run-once
```

Normal automatic operation:

```powershell
python -m tyremind.monitoring.worker
```

The worker now generates → validates/stores → calculates features → runs saved models. It loads model files once before starting and reuses the fitted pipelines across cycles. Changing artifact files while the worker is running does not hot-reload them; restart to load another version. Press Ctrl+C to stop.

## Which models and preprocessing?

**Random Forest is the default wear predictor for this requested step**, together with Isolation Forest for anomalies. This runtime choice does not change the earlier comparison: XGBoost narrowly won the ML CV comparison while the simple wear-rate baseline performed better overall. No model selection/tuning occurs during inference.

Artifacts:

- `models/wear/random_forest.joblib`: fitted imputer and Random Forest, explicit 19-feature order and 5,000-km horizon.
- `models/anomaly/isolation_forest.joblib`: fitted imputer, seven-feature Isolation Forest, score direction and validation-selected threshold.

The saved imputers transform missing values with their training-time statistics. They are not fitted again. Feature schema/version, artifact family/horizon and finite inputs/outputs are checked. No labels, simulator truth, IDs or timestamps enter the numeric predictors. Missing inspection values can be imputed by the saved wear pipeline; that remains an estimate, not a new physical measurement.

XGBoost is an optional alternative without running a new comparison:

```powershell
python -m tyremind.monitoring.predict --wear-model xgboost
python -m tyremind.monitoring.worker --wear-model xgboost --run-once
```

Only the chosen wear predictor and IF run, not RF and XGBoost together. The existing wear-rate baseline is not connected by this command.

## Outputs and meaning

`data/monitoring/predictions/` contains:

- `latest_predictions.csv`: one result/status row per current tyre, source timestamp/cutoff, predicted tread depth after another 5,000 km, anomaly score/flag, threshold, prediction ID and artifact hashes.
- `prediction_report.json`: counts, selected model names, checksums and `retrained: false`.

The `predictions` table in operational `data/tyremind.db` preserves inference history and the exact numeric input feature snapshot as JSON, plus feature version, model hashes and inference timestamp. Reading `timestamp` is the source-feature time; `predicted_at` is the wall-clock inference time. Tread prediction is at a mileage horizon, not a specific date.

Anomaly score = negative `score_samples`; larger means more unusual. The saved validation threshold determines `anomaly_flag`. This is not a probability, diagnosis, risk category or maintenance authorization. Recorded IF held-out recall is poor, so absence of a flag cannot establish a normal/safe tyre.

Stale feature rows or those with fewer than three recent observations are explicitly skipped. Their predictions/scores/flags are NULL, not zero/normal. Status is `stale_features` or `insufficient_history`; successful rows have `scored`. A stale snapshot older than newly stored telemetry fails and requests refresh rather than scoring misleadingly old fleet data.

## Persistence and failure handling

Each prediction ID hashes tyre/time, feature version/cutoff/values/status and both model-file hashes. Identical reruns do not append duplicates. New observations, changed features or a different model version produce a separate record. All rows commit together in one transaction; failed predictions/writes roll back that inference attempt. Source data, training exports and saved model files are unchanged.

CSV/report exports happen after database commit. A disk export failure can leave prior files until the standalone command is successfully rerun. The report records a CSV checksum. This is current-snapshot inference, not a durable job queue guaranteeing a prediction for every historical batch; failed cycles should be inspected/retried before proceeding when complete per-cycle coverage is needed.

The original Step 4 mode is available as `python -m tyremind.monitoring.worker --features-only`; it does not require saved models. `--heartbeat-only` remains available for timer checks. Worker arguments also support `--wear-artifact`, `--anomaly-artifact` and `--prediction-output` for explicit local paths.

## Verification

The actual saved models scored all 120 current tyres and also completed a full worker cycle, without training calls. The optional XGBoost path was checked separately. The suite has 92 tests with ML/RAG dependencies installed (one POSIX signal test is skipped on Windows). New tests compare stored predictions with loaded pipelines, prohibit `.fit()` during inference, check repeat/rollback behavior, skip stale/warmup records, validate artifacts and run the real worker with test artifacts.

```powershell
explorer .\data\monitoring\predictions
```

**Risk levels, deduplicated investigation events and AI-agent execution are not implemented in this step.**
