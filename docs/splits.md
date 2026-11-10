# Step 5: target preparation and evaluation splits

## Target definition

Random Forest will predict tread depth after exactly another 5,000 km. Daily records rarely land at that exact mileage, so this step linearly interpolates the simulator's true tread between the two bracketing mileage observations. It is a synthetic training target, not a future measured inspection or a model feature. Rows without enough future distance have no target and are excluded.

`target_available_at` is the timestamp of the later interpolation bracket. The outcome is not available until that record exists. At an exact mileage match, it is that matching record's timestamp. Both truth and target availability are excluded from `feature_columns`; IDs and timestamps are metadata. Scenario labels are not read.

This consumes the historical feature export and its matching evaluation-label file. Their keys must match, and mileage must be strictly increasing within each tyre. Do not mix exports from different simulation runs. The fixed horizon is 5,000 km in the CLI. The Python function exposes a horizon argument for testing.

## Two independent dimensions

- Time: earliest 60% of distinct timestamps is training; next 20% validation; final 20% testing. These are time-period proportions, not final row proportions.
- Truck groups: deterministic seeded shuffle, with roughly 70% training trucks, 15% reserved validation trucks and 15% reserved test trucks. Integer rounding applies; at least three trucks are required. All tyre positions of a truck stay in the same group.
- Seen evaluation uses later observations of training-group trucks. Unseen validation uses validation-group trucks; unseen final testing uses test-group trucks. The reserved test trucks never enter training or validation.
- Past telemetry/inspection history of a held-out truck can be used to construct its features at prediction time, as live inference would. Those earlier rows are not training examples for that truck.

## Boundary protection

Training rows require `target_available_at < validation_start`. Validation rows require `target_available_at < test_start`. Consequently, a gap of approximately the travel time for 5,000 km appears before each boundary. No fixed number of days is assumed: each tyre's mileage determines target availability. Final test rows also need sufficient future history inside the dataset.

Training and test features retain expected missing-history values. Later training must fit any imputation, feature selection or other learned preprocessing on training data only. Validation guides tuning; final test sets should be evaluated after tuning is finished. No global normalization or PCA is fitted here.

## Default results: 20 trucks × 180 days

| Dataset | Rows | Truck group and period |
|---|---:|---|
| train.csv | 7,428 | 14 training trucks, early period, target before validation |
| validation_seen.csv | 1,350 | Training trucks, validation period |
| validation_unseen.csv | 288 | 3 reserved validation trucks, validation period |
| test_seen.csv | 1,344 | Training trucks, final period |
| test_unseen.csv | 276 | 3 reserved test trucks, final period |
| excluded.csv | 10,914 | Reasons recorded per row |

Validation starts at 2026-04-19 UTC; final testing starts at 2026-05-25 UTC. Seed 42 determines truck membership. Rows are nonoverlapping and account for all 21,600 feature rows.

Exclusions: 4,830 reserved-truck rows outside their evaluation period, 3,678 rows whose target crosses the next cutoff, and 2,406 without enough future distance. Missing-target reasons take precedence when multiple exclusions could apply. These counts are for the default data; different dependency versions or simulation configurations can change them.

`split_report.json` records cutoffs, groups, counts, predictor allowlist and metadata. Only use its `feature_columns` for model inputs; never select every numeric CSV column because that would include the target. These files prepare supervised wear prediction; anomaly-model training and its scenario-based evaluation will be handled separately in the next model steps.

Tests cover exact/interpolated targets, cutoff purging, truck separation, deterministic exports, row accounting, invalid inputs and independence of training targets from post-cutoff truth changes. Results on this synthetic data do not establish real-world prediction accuracy.
