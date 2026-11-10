# Random Forest versus XGBoost: training and evaluation

## Run on Windows

From the cloned project folder, execute these commands individually:

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.ml.train
python -m unittest discover -s tests -v
```

The previous feature export and its matching simulation-label file are required. The command does not regenerate data. RAG dependencies are separate: `python -m pip install -e ".[ml,rag]"` installs both groups if needed. The run uses two CPU threads by default; no GPU is required. Windows installation can download a large XGBoost wheel. Training time depends on the machine.

## Validation design

Both supervised regressors predict simulator true tread after another 5,000 km. Only the explicit 19-feature allowlist enters the models. IDs, timestamps, future target values and scenario labels are excluded. The existing deterministic 14/3/3 truck grouping and final test period are reconstructed with seed 42, matching Step 5. Changing the seed changes the experiment.

Three expanding training folds have nonoverlapping validation windows:

| Fold | Train with targets available before | Validate readings in | Validation targets available before |
|---|---|---|---|
| 1 | Feb 6, 2026 UTC | Feb 6–Mar 13 | Mar 14 |
| 2 | Mar 14, 2026 UTC | Mar 14–Apr 18 | Apr 19 |
| 3 | Apr 19, 2026 UTC | Apr 19–May 24 | May 25 |

Every fold evaluates later observations of training trucks (`seen`) and reserved validation trucks (`unseen`). Reserved test trucks never enter folds. Earlier validation-truck history, formerly excluded from fixed splits, is used only for validation in the additional folds, never fitting. Cross-validation windows are before final testing. Target-horizon purging removes rows whose future outcome crosses a boundary, leaving fewer usable rows than the calendar windows suggest.

A median imputer with missingness indicators is fitted separately on each fold's training rows, with all-empty training features retained as zero placeholders. No validation/test medians are used. This is model-specific preprocessing; source/feature data is unchanged and still has no imputation. Later final fitting uses only training-group trucks and labels available before May 25. All relevant training rows before that boundary are used, including earlier purged gaps that now have known targets. Final fitting contains 10,422 rows; Step 5's original early training partition contains 7,428.

Three explicit configurations per model × three folds = 18 CV fits. Forest configurations use 200 bootstrapped regression trees, bounded depth/leaf sizes and either 70% or all post-imputation features per split. XGBoost configurations vary depth, learning rate, boosting rounds, row/column subsampling and regularization. Fixed boosting rounds are compared; no early stopping is used. Final pipelines include the fitted imputer so inference repeats training preprocessing.

## Selection and test evaluation

Each fold scores mean absolute error separately on seen/unseen trucks. Their average gets 50/50 weight so the larger familiar-truck group cannot dominate selection. Average that balanced MAE across the three folds to select each family's parameters and the ML candidate. Two baselines are also compared: latest observed tread unchanged and extrapolation from historical observed wear rate. The wear baseline accounts for distance since the latest inspection plus the additional 5,000 km. Fallback wear/tread values come only from training.

Model/configuration choices are frozen before final tests. Both model families are refitted on pre-test training-truck records, then evaluated on the same final test_seen/test_unseen rows. Test metrics are reported for comparison, not used to select parameters or change the CV winner. Do not repeatedly tune against these published test results; a future redesign requires a new untouched evaluation.

Metrics: MAE, RMSE, R², median and 90th-percentile absolute error, signed bias, and fractions within ±0.25 mm / ±0.5 mm. The fractions are tolerance-based regression measures, not classification accuracy. Precision, recall and F1 are not applicable to continuous tread prediction. They will apply when evaluating anomaly detection against defined labels. For maintenance alerts, recall measures catching faults and precision measures how many alerts are genuine; the preferred tradeoff depends on missed-fault and false-alert costs.

Scenario breakdowns group assigned simulator scenario labels for evaluation only, including inactive periods. They are not physical diagnoses, and independent truck overloading can coexist with any assigned tyre scenario. Fold standard deviations describe variation across just three dependent expanding-window fits; they are not statistical confidence intervals.

## Artifacts

`models/wear/` contains:

- `random_forest.joblib` and `xgboost.joblib`: fitted pipeline, predictor allowlist, model settings and horizon metadata.
- `comparison_report.json`: candidate/fold metrics, selection, final test and scenario results, baseline settings, dependency versions and input checksums.
- `metrics_summary.csv`: final aggregate metric table.
- `test_predictions.csv`: row-level final targets and predictions for both models/baselines.

Artifacts stay local and are ignored by Git. Rerunning replaces outputs. Never load an untrusted joblib file; its format can execute Python code. Reproduce with the same input data, seed and dependency versions when comparing runs.

See [measured results](wear_model_results.md), [aggregate CSV](results/wear_metrics.csv) and [CV summary](results/wear_cv_summary.json). Models were trained in the development environment; run the command to create your own local artifacts. The full suite has 48 tests when both `ml` and `rag` extras are installed, including an end-to-end small training run. Isolation Forest is not trained in this step.
