# Measured wear-model comparison

Default synthetic dataset, seed 42. Both real model families were trained and evaluated. These results describe the simulator, not real-world tyre performance. Aggregate machine-readable files are in `docs/results/`.

## Cross-validation selection

| Candidate | Balanced CV MAE (mm) | Fold MAE standard deviation (mm) |
|---|---:|---:|
| Random Forest | 0.2056 | 0.0543 |
| XGBoost | 0.1999 | 0.0528 |
| Historical wear-rate baseline | **0.1826** | See detailed local report |
| Latest measured tread unchanged | 0.5729 | See detailed local report |

XGBoost has the lower MAE among the ML candidates by about 0.0057 mm, a small difference that does not establish a universally superior model. The historical wear-rate baseline is better on the preselected CV criterion. It should remain the reference recommendation; there is no evidence here to prefer added model complexity solely on predictive performance.

## Final test results after refitting

| Model | Familiar-truck MAE | Familiar RMSE | Familiar R² | Held-out-truck MAE | Held-out RMSE | Held-out R² |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.2629 | 0.4526 | 0.8752 | 0.1525 | 0.1810 | 0.9431 |
| XGBoost | 0.3006 | 0.5252 | 0.8319 | 0.1438 | 0.1810 | 0.9431 |
| Wear-rate baseline | **0.1523** | **0.1898** | **0.9780** | **0.1429** | **0.1786** | **0.9446** |
| Latest measured tread | 0.6699 | 0.7268 | 0.6780 | 0.5775 | 0.6235 | 0.3250 |

Errors are in mm. There are 1,344 familiar-truck and 276 held-out-truck test rows. Held-out trucks happen to be easier in this simulation; lower unseen error does not imply new trucks are generally easier. The test comparison does not change the CV selection.

The wear-rate baseline's strength is consistent with the simulator's explicitly mileage-driven wear equation. That is an interpretation, not proof that ML is unnecessary on real fleets. Model scores can differ across dependency versions/hardware; recorded versions and selected parameters are in the CV summary.
