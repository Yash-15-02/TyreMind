# Final model evaluation and saved-artifact inventory

Training already saved the fitted pipelines. This step checks the files again without retraining or tuning. It does not deploy a model.

## Run

From the cloned repository root:

```powershell
git pull
python -m pip install -e ".[ml]"
python -m tyremind.ml.finalize
python -m unittest discover -s tests -v
```

If the Isolation Forest artifact does not exist yet, run `python -m tyremind.ml.anomaly` first. If wear artifacts are absent, run `python -m tyremind.ml.train` first. Existing successful training runs need not be repeated.

The finalization command supports the standard default project paths. Keep original features, synthetic labels and split report available; the command checks their hashes against training reports before re-evaluating. Editing/replacing input exports after training causes an error rather than silently comparing incompatible experiments.

## What is verified

1. Input checksums still match both training reports.
2. Saved Random Forest and XGBoost joblib bundles reload and retain the predictor schema.
3. Their held-out seen/unseen predictions reproduce recorded row counts, MAE, RMSE and R².
4. Saved Isolation Forest reloads with its fitted preprocessing, predictor schema and threshold.
5. Its held-out scores reproduce recorded precision, recall, F1, false-positive rates and confusion counts.
6. All three file SHA-256 hashes and sizes are recorded, alongside reloaded test metrics.

No new final-test-driven parameter/model selection is performed. This is a reproducibility check, not independent additional evidence of generalization. Only load locally trusted joblib artifacts. The actual default run verified all three persisted models. The full suite has 57 tests with ML/RAG extras installed; two finalization-specific checks reject input and metric mismatches.

## Saved files

```text
models/
├── wear/
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── comparison_report.json
│   ├── metrics_summary.csv
│   └── test_predictions.csv
├── anomaly/
│   ├── isolation_forest.joblib
│   ├── anomaly_report.json
│   ├── metrics_summary.csv
│   └── anomaly_predictions.csv
└── evaluation/
    ├── model_manifest.json
    └── saved_model_metrics.csv
```

For regression rows in the combined CSV, classification columns are blank; for anomaly rows, regression columns are blank. Different tasks need different metrics. Model artifacts are created on the machine running training, are ignored by Git and do not appear merely by cloning/pulling.

```powershell
explorer .\models
Get-Content .\models\evaluation\model_manifest.json
```

The manifest retains the CV-selected ML candidate and the baseline recommendation separately. In the recorded experiment, XGBoost narrowly wins among the wear ML models while the historical wear-rate baseline performs better. Isolation Forest has poor held-out recall. Saved/tested does not mean production-ready; these are synthetic-data experimental artifacts awaiting monitoring/risk-engine integration and further evaluation.
