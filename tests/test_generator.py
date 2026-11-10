"""Checks for reproducibility, relational integrity and scenario behavior."""
import unittest

import pandas as pd

from tyremind.data.generate import Config, generate


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = generate(Config(trucks=4, days=60))

    def test_reproducible(self):
        repeated = generate(Config(trucks=4, days=60))
        for name, frame in self.tables.items():
            pd.testing.assert_frame_equal(frame, repeated[name])

    def test_integrity(self):
        telemetry = self.tables["telemetry"]
        self.assertEqual(len(telemetry), 4 * 6 * 60)
        self.assertFalse(telemetry.duplicated(["tyre_id", "timestamp"]).any())
        self.assertFalse(telemetry.isna().any().any())
        ids = set(self.tables["tyres"].tyre_id)
        for name in ("telemetry", "inspections", "maintenance_events", "evaluation_labels"):
            self.assertTrue(set(self.tables[name].tyre_id) <= ids)
        self.assertTrue((telemetry.groupby("tyre_id").cumulative_mileage_km.diff().dropna() > 0).all())
        self.assertTrue((telemetry.groupby(["truck_id", "timestamp"]).load_ratio.nunique() == 1).all())
        self.assertNotIn("simulated_true_tread_depth_mm", telemetry.columns)

    def test_scenarios(self):
        joined = self.tables["telemetry"].merge(self.tables["evaluation_labels"], on=["tyre_id", "timestamp"])
        leak = joined[joined.scenario_label == "slow_leak"]
        self.assertLess(leak[leak.scenario_active].pressure_kpa.mean(),
                        leak[~leak.scenario_active].pressure_kpa.mean() - 20)
        hot = joined[joined.scenario_label == "elevated_temperature"]
        self.assertGreater(hot[hot.scenario_active].temperature_c.mean(),
                           hot[~hot.scenario_active].temperature_c.mean() + 10)
        for event in self.tables["maintenance_events"].itertuples():
            history = joined[joined.tyre_id == event.tyre_id].sort_values("timestamp")
            before = history[history.timestamp < event.timestamp].iloc[-1]
            after = history[history.timestamp == event.timestamp].iloc[0]
            self.assertGreater(after.pressure_kpa, before.pressure_kpa + 20)
        truth = self.tables["evaluation_labels"]
        self.assertTrue((truth.groupby("tyre_id").simulated_true_tread_depth_mm.diff().dropna() <= 0).all())

    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            generate(Config(trucks=0))


if __name__ == "__main__":
    unittest.main()
