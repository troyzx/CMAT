import json
from pathlib import Path
import tempfile
import unittest
import warnings
from dataclasses import asdict
from unittest import mock

import numpy as np

from cmat.config import BayesianScoringConfig, ExecutionConfig, RunConfig, ScoringConfig, SimulationGrid, TargetConfig
from cmat.scoring import (
    BayesianMassLimitCurve,
    BayesianMassPosterior,
    BayesianMassThresholdScorer,
    BayesianPosteriorInterval,
    BayesianSamplerDiagnostics,
    BayesianScoringSummary,
    Chi2AndRmsMassThresholdScorer,
    MassThresholds,
)
from cmat.workflow import (
    legacy_data_dir,
    load_megno_grid_cache,
    load_posterior_samples_cache,
    load_ttv_grid_cache,
    make_ttv_simulation,
    provenance_code_version,
    provenance_dependency_versions,
    provenance_runtime,
    run_simulator_adapter,
    simulator_adapter_manifest,
    write_simulator_adapter_portfolio,
    write_simulator_adapter_manifest,
    write_megno_grid_cache,
    write_posterior_samples_cache,
    write_ttv_grid_cache,
    workflow_manifest,
    write_workflow_manifest,
    SimulatorAdapterRunResult,
)


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

    def test_make_ttv_simulation_applies_execution_controls(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
            execution=ExecutionConfig(worker_count=3, start_method="spawn", show_progress=False),
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
                }
            ],
        )

        self.assertEqual(simulation.worker_count, 3)
        self.assertEqual(simulation.start_method, "spawn")
        self.assertFalse(simulation.show_progress)

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

    def test_workflow_manifest_accepts_code_and_runtime_metadata(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
        )

        manifest = workflow_manifest(
            config,
            code_version={"git_commit": "abc123", "git_dirty": False},
            runtime={"python_version": "3.11.0"},
        )

        self.assertEqual(manifest["code_version"]["git_commit"], "abc123")
        self.assertEqual(manifest["runtime"]["python_version"], "3.11.0")

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulation.get_m_crit()

        manifest = workflow_manifest(config, scoring_summary=simulation.mass_thresholds)
        serialized = json.dumps(manifest, sort_keys=True)

        self.assertIn('"backend": "bayesian"', serialized)
        self.assertIn('"posterior_sampled"', serialized)
        self.assertIn('"mass_limits"', serialized)
        self.assertIn('"log_evidence"', serialized)
        self.assertIn('"model_probabilities"', serialized)

    def test_provenance_dependency_versions_collects_installed_packages(self):
        versions = provenance_dependency_versions(("CMAT-astro", "numpy"))

        self.assertIn("CMAT-astro", versions)
        self.assertIn("numpy", versions)

    def test_provenance_runtime_records_python_and_cwd(self):
        runtime = provenance_runtime()

        self.assertIn("created_at_utc", runtime)
        self.assertIn("python_version", runtime)
        self.assertIn("cwd", runtime)

    def test_provenance_code_version_returns_git_fields(self):
        code_version = provenance_code_version()

        self.assertIn("git_commit", code_version)
        self.assertIn("git_dirty", code_version)

    def test_write_workflow_manifest_persists_metadata_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                target=TargetConfig("WASP-44 b"),
                simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
            )
            output_path = Path(tmpdir) / "artifacts" / "manifest.json"

            with mock.patch(
                "cmat.workflow.provenance_dependency_versions",
                return_value={"numpy": "2.0.0"},
            ), mock.patch(
                "cmat.workflow.provenance_code_version",
                return_value={"git_commit": "abc123", "git_dirty": False},
            ), mock.patch(
                "cmat.workflow.provenance_runtime",
                return_value={"python_version": "3.11.0"},
            ):
                written_path = write_workflow_manifest(
                    config,
                    metadata_path=output_path,
                    notes={"stage": "phase6"},
                )

            self.assertEqual(written_path, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dependency_versions"], {"numpy": "2.0.0"})
            self.assertEqual(payload["code_version"]["git_commit"], "abc123")
            self.assertEqual(payload["runtime"]["python_version"], "3.11.0")
            self.assertEqual(payload["notes"]["stage"], "phase6")

    def test_write_ttv_grid_cache_round_trips_simulation_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                target=TargetConfig("WASP-44 b"),
                simulation=SimulationGrid(period_ratios=[1.5, 2.0], companion_masses=[10.0, 20.0]),
            )
            cache_path = Path(tmpdir) / "ttv_cache.npz"

            written_path = write_ttv_grid_cache(
                config,
                epochs=np.array([0, 1, 2]),
                ttv_mcmc=np.array([0.1, -0.1, 0.0]),
                ttv_err=np.array([0.05, 0.05, 0.05]),
                ttv_results=np.array(
                    [
                        [0.1, -0.1, 0.0, 0.0],
                        [0.2, -0.2, 0.1, 0.0],
                        [0.3, -0.3, 0.2, 0.0],
                        [0.4, -0.4, 0.3, 0.0],
                    ]
                ),
                cache_path=cache_path,
            )

            self.assertEqual(written_path, cache_path)
            payload = load_ttv_grid_cache(config, cache_path=cache_path)
            np.testing.assert_array_equal(payload["period_ratios"], np.array([1.5, 2.0]))
            np.testing.assert_array_equal(payload["companion_masses"], np.array([10.0, 20.0]))
            np.testing.assert_array_equal(payload["epochs"], np.array([0, 1, 2]))
            self.assertEqual(payload["ttv_results"].shape, (4, 4))

    def test_write_megno_grid_cache_round_trips_grid_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                target=TargetConfig("WASP-44 b"),
                simulation=SimulationGrid(period_ratios=[1.5, 2.0], companion_masses=[10.0, 20.0]),
            )
            cache_path = Path(tmpdir) / "megno_cache.npz"

            written_path = write_megno_grid_cache(
                config,
                megno_results=np.array([2.0, 2.1, 9.8, 10.0]),
                cache_path=cache_path,
            )

            self.assertEqual(written_path, cache_path)
            payload = load_megno_grid_cache(config, cache_path=cache_path)
            np.testing.assert_array_equal(payload["period_ratios"], np.array([1.5, 2.0]))
            np.testing.assert_array_equal(payload["companion_masses"], np.array([10.0, 20.0]))
            np.testing.assert_array_equal(payload["megno_results"], np.array([2.0, 2.1, 9.8, 10.0]))

    def test_write_posterior_samples_cache_extracts_bayesian_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                target=TargetConfig("WASP-44 b"),
                simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
            )
            scoring_summary = MassThresholds(
                chi2=np.array([10.0]),
                rms=np.array([10.0]),
                backend="bayesian",
                bayesian=BayesianScoringSummary(
                    status="posterior_sampled",
                    contract_version="stage4",
                    sampler="emcee",
                    credible_interval=0.95,
                    rejection_log_bayes_factor_threshold=-5.0,
                    observed_transit_count=5,
                    sample_count=2,
                    requested_sample_count=2,
                    warmup_draws=1,
                    nuisance_parameters={
                        "epoch_shift": BayesianPosteriorInterval(median=0.0, lower=0.0, upper=1.0)
                    },
                    mass_limits=BayesianMassLimitCurve(
                        period_ratios=np.array([1.5]),
                        evaluated_masses=np.array([0.0, 10.0]),
                        credible_upper_bound=(10.0,),
                        rejection_upper_bound=(10.0,),
                        upper_bound=(10.0,),
                        posterior_by_period_ratio=(
                            BayesianMassPosterior(
                                period_ratio=1.5,
                                masses=np.array([0.0, 10.0]),
                                log_evidence=np.array([0.0, -1.0]),
                                model_probabilities=np.array([0.75, 0.25]),
                                cumulative_probability=np.array([0.75, 1.0]),
                                posterior_predictive_score=np.array([0.0, -1.0]),
                                best_mass=None,
                                credible_upper_bound=10.0,
                                rejection_upper_bound=10.0,
                                upper_bound=10.0,
                            ),
                        ),
                    ),
                    reference_solution={"period_ratio": None, "companion_mass": None},
                    diagnostics=BayesianSamplerDiagnostics(
                        walker_count=12,
                        step_count=3,
                        mean_acceptance_fraction=0.4,
                        max_alignment_count=2,
                    ),
                    posterior_samples={"epoch_shift": [0.0, 1.0]},
                ),
            )
            cache_path = Path(tmpdir) / "posterior_samples.json"

            written_path = write_posterior_samples_cache(
                config,
                scoring_summary=scoring_summary,
                cache_path=cache_path,
            )

            self.assertEqual(written_path, cache_path)
            payload = load_posterior_samples_cache(config, cache_path=cache_path)
            self.assertEqual(payload["sampler"], "emcee")
            np.testing.assert_array_equal(payload["posterior_samples"]["epoch_shift"], np.array([0.0, 1.0]))
            self.assertNotIn("mass_limits", payload)

    def test_write_ttv_grid_cache_rejects_mismatched_epoch_shapes(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
        )

        with self.assertRaises(ValueError):
            write_ttv_grid_cache(
                config,
                epochs=np.array([0, 1]),
                ttv_mcmc=np.array([0.1, -0.1, 0.0]),
                ttv_err=np.array([0.05, 0.05, 0.05]),
                ttv_results=np.array([[0.1, -0.1, 0.0, 0.0]]),
            )

    def test_write_posterior_samples_cache_requires_posterior_samples(self):
        config = RunConfig(
            target=TargetConfig("WASP-44 b"),
            simulation=SimulationGrid(period_ratios=[1.5], companion_masses=[10.0]),
        )

        with self.assertRaises(ValueError):
            write_posterior_samples_cache(
                config,
                scoring_summary={"backend": "bayesian", "bayesian": {"sample_count": 4}},
            )

    def test_run_simulator_adapter_returns_generic_inverse_model_result(self):
        class DemoAdapter:
            def parameter_grid(self):
                return [{"stiffness": 1.0}, {"stiffness": 2.0}, {"stiffness": 3.0}]

            def simulate(self, parameters):
                return np.array([parameters["stiffness"], parameters["stiffness"] * 2.0])

            def score(self, simulated_observable):
                target = np.array([2.0, 4.0])
                return float(np.sqrt(np.mean((simulated_observable - target) ** 2)))

            def is_accepted(self, score):
                return score <= 0.75

            def summarize(self, *, parameter_grid, scores, accepted):
                best_index = int(np.argmin(scores))
                return {
                    "accepted_count": int(np.sum(accepted)),
                    "best_parameters": parameter_grid[best_index],
                }

        result = run_simulator_adapter(
            DemoAdapter(),
            execution=ExecutionConfig(worker_count=1, show_progress=False),
        )

        self.assertIsInstance(result, SimulatorAdapterRunResult)
        np.testing.assert_allclose(result.scores, np.array([1.58113883, 0.0, 1.58113883]))
        np.testing.assert_array_equal(result.accepted, np.array([False, True, False]))
        self.assertEqual(result.summary["accepted_count"], 1)
        self.assertEqual(result.summary["best_parameters"], {"stiffness": 2.0})
        serialized = json.dumps(result.to_dict(), sort_keys=True)
        self.assertIn('"accepted_count": 1', serialized)

    def test_run_simulator_adapter_uses_execution_controls(self):
        class DemoAdapter:
            def parameter_grid(self):
                return [{"x": 1.0}, {"x": 2.0}]

            def simulate(self, parameters):
                return parameters["x"]

            def score(self, simulated_observable):
                return float(simulated_observable)

            def is_accepted(self, score):
                return score < 2.0

            def summarize(self, *, parameter_grid, scores, accepted):
                return {"count": len(parameter_grid)}

        context_calls = []

        class FakePool:
            def __init__(self, process_count):
                self.process_count = process_count

            def imap(self, func, tasks):
                return map(func, tasks)

            def close(self):
                return None

            def join(self):
                return None

        class FakeContext:
            def Pool(self, process_count):
                context_calls.append(process_count)
                return FakePool(process_count)

        with mock.patch("cmat.workflow.get_context", return_value=FakeContext()) as get_context_mock, mock.patch(
            "cmat.workflow.tqdm"
        ) as tqdm_mock:
            result = run_simulator_adapter(
                DemoAdapter(),
                execution=ExecutionConfig(
                    worker_count=3,
                    start_method="spawn",
                    show_progress=False,
                ),
            )

        get_context_mock.assert_called_once_with("spawn")
        tqdm_mock.assert_not_called()
        self.assertEqual(context_calls, [3])
        np.testing.assert_array_equal(result.accepted, np.array([True, False]))

    def test_write_simulator_adapter_portfolio_writes_report_table_and_figure(self):
        result = SimulatorAdapterRunResult(
            parameter_grid=(
                {"damping": 0.3, "stiffness": 3.0},
                {"damping": 0.6, "stiffness": 4.0},
            ),
            scores=np.array([0.25, 0.0]),
            accepted=np.array([False, True]),
            summary={"accepted_count": 1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_simulator_adapter_portfolio(
                result,
                output_dir=Path(tmpdir) / "portfolio",
                title="Demo portfolio",
            )

            self.assertTrue(paths["report"].exists())
            self.assertTrue(paths["summary"].exists())
            self.assertTrue(paths["table"].exists())
            self.assertTrue(paths["figure"].exists())

            summary_payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual(summary_payload["accepted_count"], 1)
            self.assertEqual(summary_payload["best_parameters"], {"damping": 0.6, "stiffness": 4.0})
            table_payload = paths["table"].read_text(encoding="utf-8")
            self.assertIn("damping,stiffness,score,accepted", table_payload)
            self.assertIn("0.6,4.0,0.0,True", table_payload)

    def test_simulator_adapter_manifest_is_json_serializable(self):
        result = SimulatorAdapterRunResult(
            parameter_grid=(
                {"damping": 0.3, "stiffness": 3.0},
                {"damping": 0.6, "stiffness": 4.0},
            ),
            scores=np.array([0.25, 0.0]),
            accepted=np.array([False, True]),
            summary={"accepted_count": 1},
        )

        manifest = simulator_adapter_manifest(
            result,
            adapter_name="damped_oscillator",
            execution=ExecutionConfig(worker_count=2, start_method="spawn", show_progress=False),
            dependency_versions={"numpy": "2.0.0"},
            code_version={"git_commit": "abc123", "git_dirty": False},
            runtime={"python_version": "3.11.0"},
            notes={"mode": "batch"},
        )
        serialized = json.dumps(manifest, sort_keys=True)

        self.assertIn('"adapter_name": "damped_oscillator"', serialized)
        self.assertIn('"accepted_count": 1', serialized)
        self.assertIn('"worker_count": 2', serialized)
        self.assertIn('"best_score": 0.0', serialized)
        self.assertIn('"mode": "batch"', serialized)

    def test_write_simulator_adapter_manifest_persists_metadata_file(self):
        result = SimulatorAdapterRunResult(
            parameter_grid=(
                {"damping": 0.3, "stiffness": 3.0},
                {"damping": 0.6, "stiffness": 4.0},
            ),
            scores=np.array([0.25, 0.0]),
            accepted=np.array([False, True]),
            summary={"accepted_count": 1},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "deployment" / "run_metadata.json"

            with mock.patch(
                "cmat.workflow.provenance_dependency_versions",
                return_value={"numpy": "2.0.0"},
            ), mock.patch(
                "cmat.workflow.provenance_code_version",
                return_value={"git_commit": "abc123", "git_dirty": False},
            ), mock.patch(
                "cmat.workflow.provenance_runtime",
                return_value={"python_version": "3.11.0"},
            ):
                written_path = write_simulator_adapter_manifest(
                    result,
                    adapter_name="damped_oscillator",
                    metadata_path=output_path,
                    execution=ExecutionConfig(worker_count=1, show_progress=False),
                    notes={"mode": "batch"},
                )

            self.assertEqual(written_path, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["adapter_name"], "damped_oscillator")
            self.assertEqual(payload["dependency_versions"], {"numpy": "2.0.0"})
            self.assertEqual(payload["code_version"]["git_commit"], "abc123")
            self.assertEqual(payload["runtime"]["python_version"], "3.11.0")
            self.assertEqual(payload["notes"]["mode"], "batch")
            self.assertEqual(payload["best_parameters"], {"damping": 0.6, "stiffness": 4.0})


if __name__ == "__main__":
    unittest.main()
