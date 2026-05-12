import json
import unittest

import numpy as np

from cmat.config import (
    BayesianScoringConfig,
    ExecutionConfig,
    FitControls,
    OutputConfig,
    RunConfig,
    ScoringConfig,
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
        self.assertEqual(output.cache_dir.as_posix(), "artifacts/wasp44-smoke/cache")
        self.assertEqual(
            output.ttv_grid_cache_path.as_posix(),
            "artifacts/wasp44-smoke/cache/ttv_grid.npz",
        )
        self.assertEqual(
            output.megno_grid_cache_path.as_posix(),
            "artifacts/wasp44-smoke/cache/megno_grid.npz",
        )
        self.assertEqual(
            output.posterior_samples_cache_path.as_posix(),
            "artifacts/wasp44-smoke/cache/posterior_samples.json",
        )

    def test_scoring_config_is_json_serializable(self):
        scoring = ScoringConfig()

        self.assertEqual(scoring.to_dict(), {"backend": "chi2_rms"})

    def test_execution_config_serializes_runtime_controls(self):
        execution = ExecutionConfig(worker_count=4, start_method="spawn", show_progress=False)

        self.assertEqual(
            execution.to_dict(),
            {
                "worker_count": 4,
                "start_method": "spawn",
                "show_progress": False,
            },
        )

    def test_execution_config_rejects_invalid_runtime_controls(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(worker_count=0)
        with self.assertRaises(ValueError):
            ExecutionConfig(start_method="thread")
        with self.assertRaises(TypeError):
            ExecutionConfig(show_progress="yes")

    def test_scoring_config_supports_bayesian_contract(self):
        scoring = ScoringConfig(
            backend="bayesian",
            bayesian=BayesianScoringConfig(
                credible_interval=0.95,
                posterior_sample_count=256,
                warmup_draws=128,
                nuisance_parameters=("epoch_shift", "baseline_offset", "jitter"),
            ),
        )

        self.assertEqual(
            scoring.to_dict(),
            {
                "backend": "bayesian",
                "bayesian": {
                    "credible_interval": 0.95,
                    "rejection_log_bayes_factor_threshold": -5.0,
                    "posterior_sample_count": 256,
                    "warmup_draws": 128,
                    "nuisance_parameters": [
                        "epoch_shift",
                        "baseline_offset",
                        "jitter",
                    ],
                    "store_chains": False,
                },
            },
        )

    def test_scoring_config_rejects_bayesian_options_for_non_bayesian_backend(self):
        with self.assertRaises(ValueError):
            ScoringConfig(
                backend="chi2_rms",
                bayesian=BayesianScoringConfig(),
            )

    def test_scoring_config_rejects_unknown_backend(self):
        with self.assertRaises(ValueError):
            ScoringConfig(backend="bayes")

    def test_bayesian_scoring_config_rejects_invalid_contract_settings(self):
        with self.assertRaises(ValueError):
            BayesianScoringConfig(credible_interval=1.0)
        with self.assertRaises(ValueError):
            BayesianScoringConfig(nuisance_parameters=())
        with self.assertRaises(ValueError):
            BayesianScoringConfig(posterior_sample_count=0)
        with self.assertRaises(ValueError):
            BayesianScoringConfig(rejection_log_bayes_factor_threshold=np.nan)

    def test_bayesian_scoring_config_rejects_unsupported_nuisance_parameters(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported Bayesian nuisance parameters: stale_parameter",
        ):
            BayesianScoringConfig(nuisance_parameters=("epoch_shift", "stale_parameter"))

    def test_bayesian_scoring_config_accepts_supported_nuisance_parameter_subset(self):
        config = BayesianScoringConfig(nuisance_parameters=("epoch_shift",))

        self.assertEqual(config.nuisance_parameters, ("epoch_shift",))
        self.assertEqual(config.to_dict()["nuisance_parameters"], ["epoch_shift"])

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
        self.assertIn('"backend": "chi2_rms"', serialized)
        self.assertIn('"worker_count": 1', serialized)

    def test_run_config_validates_seed(self):
        with self.assertRaises(ValueError):
            RunConfig(target=TargetConfig("WASP-44 b"), random_seed=-1)

    def test_run_config_validates_scoring_type(self):
        with self.assertRaises(TypeError):
            RunConfig(target=TargetConfig("WASP-44 b"), scoring="chi2_rms")

    def test_run_config_validates_execution_type(self):
        with self.assertRaises(TypeError):
            RunConfig(target=TargetConfig("WASP-44 b"), execution="spawn")


if __name__ == "__main__":
    unittest.main()
