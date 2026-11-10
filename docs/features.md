# Step 4: feature definitions

The source is the Step 3 SQLite database. All features describe information available at the row timestamp. IDs and timestamps are context columns, not numeric model inputs. Current cumulative mileage is retained alongside derived features.

| Feature | Definition |
|---|---|
| pressure_deviation_ratio | Actual pressure / reference pressure minus 1 |
| pressure_change_kpa_per_day | Change from the previous available reading divided by elapsed days |
| pressure_mean_7d_kpa | Mean pressure within the trailing seven-day window |
| temperature_mean_7d_c | Mean temperature within that window |
| temperature_std_7d_c | Sample temperature standard deviation within that window |
| distance_sum_7d_km | Sum of reported daily distance within that window |
| load_mean_7d | Mean load ratio within that window |
| overload_fraction_7d | Fraction of available readings with load ratio greater than 1 |
| observations_7d | Count of available readings in the window |
| tyre_age_days | Elapsed time since installation, in days |
| cumulative_mileage_km | Current accumulated tyre mileage |
| latest_measured_tread_mm | Latest inspection measurement at or before this timestamp |
| days_since_inspection | Elapsed days since that inspection |
| distance_since_inspection_km | Current mileage minus mileage associated with that inspection |
| historical_wear_mm_per_1000km | Tread decline between the latest two available inspections / distance between them × 1,000 |
| days_since_maintenance | Elapsed days since the latest available maintenance event |
| has_inspection | 1 if a prior/current measured inspection exists, otherwise 0 |
| has_wear_history | 1 if two inspections with increasing associated mileage support a wear calculation |
| has_maintenance | 1 if a prior/current maintenance event exists, otherwise 0 |

## Time and missing history

- Seven-day windows are `(timestamp - 7 days, timestamp]`, with at least three readings for statistics. Counts are available immediately. These are actual elapsed-time windows, not seven-row windows.
- Daily distance sums assume Step 1's one daily observation per tyre; irregular/multiple readings per day would require distance interval semantics before live use. Overload fraction is observation-weighted, not duration-weighted.
- First-row pressure change is blank. Wear rate is blank until two inspections with positive distance between them exist. Inspection and maintenance fields remain blank until the corresponding records exist. No imputation, backfill or extrapolated current tread is used.
- An old measured tread remains explicitly identified as measured, with age and distance since inspection. It is not declared to be the current true tread.
- Negative observed wear rates can occur from inspection noise; they are retained, not silently clipped.
- Inspection mileage is the most recent telemetry mileage at or before the inspection. In sparse telemetry this is approximate. Never interpolate using a later reading.
- Inspections and maintenance are assumed available at their event timestamp, including exact matches. Real delayed ingestion will require availability timestamps. Same-time duplicate inspections use the lexicographically last inspection ID as a deterministic tie-break.
- Each tyre's history is processed separately. No hidden simulation labels, future tread targets or future observations are read. The simulator has one installation per tyre; replacement lifecycle support is future work.

## Output and later training

`tyre_features.csv` is a feature snapshot, not a complete training dataset. `feature_report.json` records missing counts, feature version and availability rules. Output files are replaced on rerun and excluded from Git. The input database is unchanged.

No PCA or automated correlation-based deletion is applied. Later steps will create the 5,000-km target, split by time/truck with horizon-aware boundaries, select model-specific inputs and fit any missing-value handling on training data only. Historical windows may use earlier observations at evaluation time, just as live inference does; future outcomes may not leak backward.
