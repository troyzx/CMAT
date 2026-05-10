import json
import unittest

import numpy as np

from cmat.config import (
    FitControls,
    OutputConfig,
    RunConfig,
    SimulationGrid,
    TargetConfig,
)


class ConfigTests(unittest.TestCase):
    def test_target_config_normalizes_storage_paths(self):
        target = TargetConfig("WASP-44 b", data_dir="data", product_subgroups="LC")

        self.assertEqual(target.target_data_dir.as_posix(), "data/WASP-44 b")
        self.assertEqual(target.product_subgroups, ("LC",))
        self.assertEqual(
            target.to_dict(),
            {
                "planet_name": "WASP-44 b",
                "data_dir": "data",
                "target_data_dir": "data/WASP-44 b",
                "product_subgroups": ["LC"],
            },
        )

    def test_target_config_rejects_blank_target_name(self):
        with self.assertRaises(ValueError):
            TargetConfig(" ")

    def test_fit_controls_validate_positive_counts(self):
        controls = FitControls(global_niter=10, global_npop=5)

        self.assertEqual(controls.to_dict()["global_niter"], 10)
        with self.assertRaises(ValueError):
            FitControls(mcmc_repeats=0)

    def test_simulation_grid_preserves_existing_parameter_order(self):
        grid = SimulationGrid(
            period_ratios=np.array([1.5, 2.0]),
            companion_masses=[10.0, 20.0],
            n_transit_simulations=4,
        )

        self.assertEqual(grid.parameter_count, 4)
        self.assertEqual(
            grid.parameter_pairs(),
            [(1.5, 10.0), (2.0, 10.0), (1.5, 20.0), (2.0, 20.0)],
        )
        self.assertEqual(grid.to_dict()["companion_masses"], [10.0, 20.0])

    def test_simulation_grid_requires_sorted_positive_masses(self):
        with self.assertRaises(ValueError):
            SimulationGrid(period_ratios=[1.5], companion_masses=[20.0, 10.0])
        with self.assertRaises(ValueError):
            SimulationGrid(period_ratios=[0.0], companion_masses=[10.0])

    def test_output_config_builds_run_artifact_paths(self):
        output = OutputConfig(root_dir="artifacts", run_name="wasp44-smoke")

        self.assertEqual(output.tables_dir.as_posix(), "artifacts/wasp44-smoke/tables")
        self.assertEqual(
            output.metadata_path.as_posix(),
            "artifacts/wasp44-smoke/run_metadata.json",
        )

    def test_run_config_is_json_serializable(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
            output=OutputConfig(run_name="wasp44-smoke"),
            random_seed=42,
        )

        serialized = json.dumps(config.to_dict(), sort_keys=True)

        self.assertIn('"random_seed": 42', serialized)
        self.assertIn('"parameter_count": 1', serialized)

    def test_run_config_validates_seed(self):
        with self.assertRaises(ValueError):
            RunConfig(target=TargetConfig("WASP-44 b"), random_seed=-1)


if __name__ == "__main__":
    unittest.main()
