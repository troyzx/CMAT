import json
import unittest
from dataclasses import asdict

import numpy as np

from cmat.config import BayesianScoringConfig, RunConfig, ScoringConfig, SimulationGrid, TargetConfig
from cmat.scoring import BayesianMassThresholdScorer, Chi2AndRmsMassThresholdScorer, MassThresholds
from cmat.workflow import legacy_data_dir, make_ttv_simulation, workflow_manifest


class WorkflowTests(unittest.TestCase):
    def test_legacy_data_dir_matches_existing_fitlpf_expectation(self):
        target = TargetConfig("WASP-44 b", data_dir="data")

        self.assertEqual(legacy_data_dir(target), "data/")

    def test_make_ttv_simulation_uses_configured_grid(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(
                period_ratios=[1.5, 2.0],
                companion_masses=[10.0, 20.0],
                n_transit_simulations=6,
                megno_dt=0.1,
                megno_runtime=100.0,
            ),
        )
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]

        simulation = make_ttv_simulation(
            config,
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([0.0, 1.0, 0.0]),
            ttv_err=np.ones(3),
            prop=prop,
        )

        np.testing.assert_array_equal(simulation.rs, np.array([1.5, 2.0]))
        np.testing.assert_array_equal(simulation.mp2s, np.array([10.0, 20.0]))
        self.assertEqual(simulation.N, 6)
        self.assertEqual(simulation.megno_dt, 0.1)
        self.assertEqual(simulation.megno_runtime, 100.0)
        self.assertIsInstance(simulation.scoring_backend, Chi2AndRmsMassThresholdScorer)

    def test_make_ttv_simulation_builds_bayesian_contract_backend_from_config(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
            scoring=ScoringConfig(
                backend="bayesian",
                bayesian=BayesianScoringConfig(
                    credible_interval=0.95,
                    posterior_sample_count=128,
                    warmup_draws=64,
                ),
            ),
        )

        simulation = make_ttv_simulation(
            config,
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([0.0, 1.0, 0.0]),
            ttv_err=np.ones(3),
            prop=[
                {
                    "orbital_distance": 1.0,
                    "orbital_period": 1.0,
                    "Mp": 1.0,
                    "Ms": 1.0,
                    "Rs": 1.0,
                    "Rp": 1.0,
                }
            ],
        )

        self.assertIsInstance(simulation.scoring_backend, BayesianMassThresholdScorer)
        self.assertEqual(
            asdict(simulation.scoring_backend.config),
            asdict(config.scoring.bayesian),
        )

    def test_make_ttv_simulation_requires_simulation_config(self):
        config = RunConfig(target=TargetConfig("WASP-44 b"))

        with self.assertRaises(ValueError):
            make_ttv_simulation(
                config,
                epochs=np.array([0]),
                ttv_mcmc=np.array([0.0]),
                ttv_err=np.ones(1),
                prop=[],
            )

    def test_make_ttv_simulation_accepts_custom_scoring_backend(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
        )
        scoring_backend = object()
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]

        simulation = make_ttv_simulation(
            config,
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([0.0, 1.0, 0.0]),
            ttv_err=np.ones(3),
            prop=prop,
            scoring_backend=scoring_backend,
        )

        self.assertIs(simulation.scoring_backend, scoring_backend)

    def test_workflow_manifest_is_json_serializable(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
            random_seed=42,
        )

        manifest = workflow_manifest(
            config,
            dependency_versions={"numpy": np.__version__},
            notes={"stage": "rebuild"},
            scoring_summary=MassThresholds(
                chi2=np.array([10.0]),
                rms=np.array([20.0]),
                backend="chi2_rms",
                chi2_threshold=1.5,
                rms_threshold=2.5,
            ),
        )
        serialized = json.dumps(manifest, sort_keys=True)

        self.assertIn('"dependency_versions"', serialized)
        self.assertIn('"stage": "rebuild"', serialized)
        self.assertIn('"scoring_summary"', serialized)
        self.assertIn('"backend": "chi2_rms"', serialized)

    def test_workflow_manifest_serializes_bayesian_scoring_summary(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
            scoring=ScoringConfig(
                backend="bayesian",
                bayesian=BayesianScoringConfig(
                    credible_interval=0.8,
                    posterior_sample_count=64,
                    warmup_draws=24,
                ),
            ),
        )
        simulation = make_ttv_simulation(
            config,
            epochs=np.array([0, 1, 2, 3, 4]),
            ttv_mcmc=np.array([0.3, -0.2, 0.1, -0.25, 0.35]),
            ttv_err=np.full(5, 0.05),
            prop=[
                {
                    "orbital_distance": 1.0,
                    "orbital_period": 1.0,
                    "Mp": 1.0,
                    "Ms": 1.0,
                    "Rs": 1.0,
                    "Rp": 1.0,
                }
            ],
        )
        simulation.ttv_results = [np.array([0.3, -0.2, 0.1, -0.25, 0.35, 0.0])]
        simulation.get_m_crit()

        manifest = workflow_manifest(config, scoring_summary=simulation.mass_thresholds)
        serialized = json.dumps(manifest, sort_keys=True)

        self.assertIn('"backend": "bayesian"', serialized)
        self.assertIn('"posterior_sampled"', serialized)
        self.assertIn('"mass_limits"', serialized)
        self.assertIn('"log_evidence"', serialized)
        self.assertIn('"model_probabilities"', serialized)


if __name__ == "__main__":
    unittest.main()
