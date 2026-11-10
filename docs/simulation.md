# Synthetic-data assumptions

All numbers are demonstration assumptions, not Michelin specifications, service limits or validated tyre physics. The vehicle is an abstract six-position truck. Each tyre remains installed for this dataset; tyre replacements will need explicit lifecycle handling in later steps.

- Daily truck distance: normal distribution with mean 260 km and standard deviation 45 km, clipped to 80–450 km. Six tyres share the same distance and load.
- Base load ratio: mean 0.85, standard deviation 0.07, clipped to 0.5–1.05. Every fourth truck receives an additional 0.3 after one third of the history, affecting all six tyres. This can coexist with a tyre-specific fault.
- Reference pressure: 800 kPa. Normal pressure noise has standard deviation 5 kPa. This simplified model does not perform hot/cold pressure compensation.
- Initial tread: uniformly 14–17 mm. Baseline wear: uniformly 0.055–0.085 mm per 1,000 km.
- Daily wear equals baseline wear × distance/1,000 × pressure factor × load factor × temperature factor × fault factor × small multiplicative noise. Tread cannot increase or fall below zero.
- Pressure factor: `1 + 2 × max(0, (reference − measured pressure)/reference)`.
- Load factor: `1 + 1.5 × max(0, load ratio − 1)`.
- Temperature factor: `1 + max(0, temperature − 60)/80`.
- Ambient temperature cycles around 23°C. Operating temperature adds 22°C plus load, pressure-loss effects and noise. No safety cutoff is implied.
- Tyres are assigned normal, slow-leak, elevated-temperature, overloading, accelerated-wear or maintenance-intervention scenarios in a balanced repeating sequence. Scenario onset varies between one third and one half of the history. Truck overloading follows the shared truck schedule instead.
- Leaks lose 3 kPa per day after onset, capped at 210 kPa. Elevated-temperature faults add 22°C. Accelerated-wear faults multiply wear by 2.3.
- Maintenance-intervention tyres develop a leak and recover pressure on the recorded repair day; lost tread is not restored.
- Inspection measurement noise has standard deviation 0.06 mm. Occasional apparent increases between measured inspections can therefore occur even though true tread declines.
- Telemetry contains no tread sensor. Inspection measurements are separate from simulator truth. Future preprocessing must use only inspections available at the prediction timestamp.
- `scenario_label` identifies the assigned scenario; `scenario_active` records whether it is active. `overload_active` independently records truck overloading. A normal-assigned tyre can still experience truck overloading.

This step creates clean source histories. Missing readings, duplicates and out-of-order fault injection are deferred to preprocessing work. No future 5,000-km labels or ML features are generated yet. Use simulation truth only to construct/evaluate future targets with time-aware boundaries in later steps.
