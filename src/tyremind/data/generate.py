"""Generate reproducible daily histories; all coefficients are demo assumptions."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


POSITIONS = ("FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT_OUTER",
             "REAR_LEFT_INNER", "REAR_RIGHT_INNER", "REAR_RIGHT_OUTER")
SCENARIOS = ("normal", "slow_leak", "elevated_temperature", "overloading",
             "accelerated_wear", "maintenance_intervention")


@dataclass(frozen=True)
class Config:
    trucks: int = 20
    days: int = 180
    seed: int = 42
    start_date: str = "2026-01-01"


def generate(config: Config, *, include_state: bool = False) -> dict[str, pd.DataFrame]:
    """Return relational CSV-ready tables without writing to disk."""
    if config.trucks < 1 or config.days < 30:
        raise ValueError("Use at least one truck and 30 days.")
    rng = np.random.default_rng(config.seed)
    dates = pd.date_range(config.start_date, periods=config.days, freq="D", tz="UTC")
    trucks, tyres, telemetry, inspections, maintenance, labels = [], [], [], [], [], []
    continuation = []
    for truck_index in range(config.trucks):
        truck_id = f"TRK-{truck_index + 1:03d}"
        trucks.append(dict(truck_id=truck_id, vehicle_type="simulated_6_position_truck"))
        # Distance and load belong to the truck: all six positions share them.
        distance = np.clip(rng.normal(260, 45, config.days), 80, 450)
        load = np.clip(rng.normal(0.85, 0.07, config.days), 0.5, 1.05)
        ambient = 23 + 6 * np.sin(np.arange(config.days) * 2 * np.pi / 90)
        for position_index, position in enumerate(POSITIONS):
            tyre_id = f"TYR-{truck_index * 6 + position_index + 1:04d}"
            scenario = SCENARIOS[(truck_index * 6 + position_index) % len(SCENARIOS)]
            onset = int(rng.integers(max(1, config.days // 3), max(2, config.days // 2)))
            service_day = min(config.days - 1, onset + max(10, config.days // 5))
            reference = 800.0
            initial_tread = float(rng.uniform(14, 17))
            base_wear = float(rng.uniform(0.055, 0.085))
            tread, mileage = initial_tread, 0.0
            tyres.append(dict(tyre_id=tyre_id, truck_id=truck_id, position=position,
                              installation_timestamp=dates[0].isoformat(),
                              reference_pressure_kpa=reference,
                              initial_tread_depth_mm=initial_tread))
            for day, timestamp in enumerate(dates):
                active = day >= onset
                repaired = scenario == "maintenance_intervention" and day >= service_day
                leaking = active and (scenario == "slow_leak" or
                                      (scenario == "maintenance_intervention" and not repaired))
                loss = min(210, (day - onset + 1) * 3) if leaking else 0
                pressure = reference - loss + rng.normal(0, 5)
                # Truck-level overloading affects every tyre on that vehicle.
                overloaded = truck_index % 4 == 3 and day >= config.days // 3
                actual_load = float(load[day] + (0.3 if overloaded else 0))
                hot = active and scenario == "elevated_temperature"
                temperature = (ambient[day] + 22 + 18 * max(0, actual_load - 0.85)
                               + loss * 0.035 + (22 if hot else 0) + rng.normal(0, 2))
                pressure_factor = 1 + 2 * max(0, (reference - pressure) / reference)
                load_factor = 1 + 1.5 * max(0, actual_load - 1)
                temperature_factor = 1 + max(0, temperature - 60) / 80
                extra_wear = 2.3 if active and scenario == "accelerated_wear" else 1
                wear = (base_wear * distance[day] / 1000 * pressure_factor * load_factor
                        * temperature_factor * extra_wear * rng.uniform(0.95, 1.05))
                tread = max(0, tread - wear)
                mileage += distance[day]
                ts = timestamp.isoformat()
                telemetry.append(dict(truck_id=truck_id, tyre_id=tyre_id, timestamp=ts,
                                      pressure_kpa=round(pressure, 3),
                                      temperature_c=round(temperature, 3),
                                      daily_distance_km=round(float(distance[day]), 3),
                                      cumulative_mileage_km=round(float(mileage), 3),
                                      load_ratio=round(actual_load, 4)))
                inspection_day = day % 14 == 0 or day == config.days - 1 or (
                    scenario == "maintenance_intervention" and day == service_day)
                if inspection_day:
                    inspections.append(dict(inspection_id=f"INSP-{len(inspections)+1:06d}",
                                            tyre_id=tyre_id, timestamp=ts,
                                            tread_depth_mm=round(float(np.clip(
                                                tread + rng.normal(0, 0.06), 0, initial_tread)), 3)))
                if scenario == "maintenance_intervention" and day == service_day:
                    maintenance.append(dict(event_id=f"MAINT-{len(maintenance)+1:05d}",
                                            tyre_id=tyre_id, timestamp=ts,
                                            event_type="simulated_leak_repair_and_reinflation"))
                labels.append(dict(tyre_id=tyre_id, timestamp=ts,
                                   scenario_label=scenario,
                                   scenario_active=bool(overloaded if scenario == "overloading" else
                                                        active and scenario != "normal" and not repaired),
                                   overload_active=bool(overloaded),
                                   simulated_true_tread_depth_mm=round(tread, 4)))
            if include_state:
                continuation.append(dict(truck_id=truck_id, tyre_id=tyre_id, truck_index=truck_index,
                                         reference_pressure_kpa=reference, initial_tread_depth_mm=initial_tread,
                                         base_wear=base_wear, tread_depth_mm=tread, mileage_km=float(mileage),
                                         scenario=scenario, onset=onset, service_day=service_day))
    tables = {
        "trucks": pd.DataFrame(trucks), "tyres": pd.DataFrame(tyres),
        "telemetry": pd.DataFrame(telemetry), "inspections": pd.DataFrame(inspections),
        "maintenance_events": pd.DataFrame(maintenance, columns=[
            "event_id", "tyre_id", "timestamp", "event_type"]),
        "evaluation_labels": pd.DataFrame(labels),
    }
    if include_state:
        tables["simulator_state"] = pd.DataFrame(continuation)
    return tables


def write_dataset(config: Config, output: Path) -> dict:
    tables = generate(config)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    manifest = {"generator_version": "0.1.0", "config": asdict(config),
                "row_counts": {name: len(frame) for name, frame in tables.items()},
                "notice": "Synthetic demo assumptions; not Michelin specifications or validated safety data."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trucks", type=int, default=20)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--output", type=Path, default=Path("data/generated"))
    args = parser.parse_args()
    config = Config(args.trucks, args.days, args.seed, args.start_date)
    print(json.dumps(write_dataset(config, args.output), indent=2))


if __name__ == "__main__":
    main()
