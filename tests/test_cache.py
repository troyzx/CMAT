import os
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np

from cmat import TTVSimulation
from cmat.scoring import MassThresholds, BayesianScoringSummary, BayesianMassLimitCurve, BayesianPosteriorInterval, BayesianMassPosterior


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.prop = [{"orbital_distance": 1.0, "orbital_period": 1.0, "Mp": 1.0, "Ms": 1.0, "Rs": 1.0, "Rp": 1.0}]
        self.sim = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=self.prop,
        )

    def test_save_and_load_ttv_grid_restores_results(self):
        self.sim.ttv_results = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0, 2.0])
        ]
        self.sim.ttv_rebound = np.array(self.sim.ttv_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            self.sim.save_ttv_grid_cache(path)
            
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            sim2.load_ttv_grid_cache(path)
            
            np.testing.assert_allclose(sim2.ttv_rebound, self.sim.ttv_rebound)
            self.assertEqual(len(sim2.ttv_results), len(self.sim.ttv_results))

    def test_get_ttv_rebound_all_with_cache_loads_without_recomputing(self):
        self.sim.ttv_results = [np.ones(4) for _ in range(4)]
        self.sim.ttv_rebound = np.array(self.sim.ttv_results)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            self.sim.save_ttv_grid_cache(path)
            
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            with mock.patch("cmat.ttv_sim.get_context") as mock_context:
                sim2.get_ttv_rebound_all(use_cache=True, cache_path=path)
                mock_context.assert_not_called()
                
            np.testing.assert_allclose(sim2.ttv_rebound, self.sim.ttv_rebound)

    def test_get_ttv_rebound_all_requires_cache_path_when_cache_enabled(self):
        with self.assertRaisesRegex(ValueError, "cache_path is required"):
            self.sim.get_ttv_rebound_all(use_cache=True)

    def test_get_ttv_rebound_all_with_cache_saves_after_computing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            
            class FakePool:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def imap(self, func, params):
                    return iter([np.ones(4) for _ in params])

            with mock.patch("cmat.ttv_sim.get_context") as mock_context:
                mock_context.return_value.Pool.return_value = FakePool()
                self.sim.get_ttv_rebound_all(use_cache=True, cache_path=path)
                
            self.assertTrue(os.path.exists(path))
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            sim2.load_ttv_grid_cache(path)
            np.testing.assert_allclose(sim2.ttv_rebound, self.sim.ttv_rebound)

    def test_incompatible_cache_raises_value_error(self):
        self.sim.ttv_results = [np.ones(4) for _ in range(4)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            self.sim.save_ttv_grid_cache(path)
            
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=np.array([1.0, 3.0]), # Changed rs
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            with self.assertRaises(ValueError):
                sim2.load_ttv_grid_cache(path)

    def test_incompatible_epochs_raises_value_error(self):
        self.sim.ttv_results = [np.ones(4) for _ in range(4)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            self.sim.save_ttv_grid_cache(path)
            
            sim2 = TTVSimulation(
                epochs=np.array([0, 1, 3]), # Changed epochs
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            with self.assertRaises(ValueError):
                sim2.load_ttv_grid_cache(path)

    def test_save_and_load_megno_grid_restores_results(self):
        self.sim.megno_results = [1.0, 2.0, 3.0, 4.0]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "megno.npz")
            self.sim.save_megno_grid_cache(path)
            
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            sim2.load_megno_grid_cache(path)
            self.assertEqual(sim2.megno_results, self.sim.megno_results)

    def test_run_megno_requires_cache_path_when_cache_enabled(self):
        with self.assertRaisesRegex(ValueError, "cache_path is required"):
            self.sim.run_megno(use_cache=True)

    def test_run_megno_with_cache_loads_without_recomputing(self):
        self.sim.megno_results = [1.0, 2.0, 3.0, 4.0]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "megno.npz")
            self.sim.save_megno_grid_cache(path)
            
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            with mock.patch("cmat.ttv_sim.get_context") as mock_context:
                sim2.run_megno(use_cache=True, cache_path=path)
                mock_context.assert_not_called()
                
            self.assertEqual(sim2.megno_results, self.sim.megno_results)

    def test_save_and_load_scoring_summary_preserves_surfaces(self):
        self.sim.ttv_results = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]
        self.sim.get_m_crit()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scoring.npz")
            self.sim.save_scoring_summary(path)
            
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            sim2.load_scoring_summary(path)
            
            np.testing.assert_allclose(sim2.get_chi2_surface(), self.sim.get_chi2_surface())
            np.testing.assert_allclose(sim2.get_reduced_chi2_surface(), self.sim.get_reduced_chi2_surface())
            np.testing.assert_allclose(sim2.get_relative_log_likelihood_surface(), self.sim.get_relative_log_likelihood_surface())

    def test_scoring_summary_preserves_bayesian_summary(self):
        posteriors = [
            BayesianMassPosterior(
                period_ratio=1.5,
                masses=np.array([10.0, 20.0]),
                log_evidence=np.array([-1.0, -2.0]),
                model_probabilities=np.array([0.9, 0.1]),
                cumulative_probability=np.array([0.9, 1.0]),
                posterior_predictive_score=np.array([0.5, 0.5]),
                best_mass=10.0,
                credible_upper_bound=10.0,
                rejection_upper_bound=20.0,
                upper_bound=10.0,
            )
        ]
        limits = BayesianMassLimitCurve(
            period_ratios=np.array([1.5]),
            evaluated_masses=np.array([10.0, 20.0]),
            credible_upper_bound=(10.0,),
            rejection_upper_bound=(20.0,),
            upper_bound=(10.0,),
            units="earth_masses",
            posterior_by_period_ratio=tuple(posteriors)
        )
        bayesian = BayesianScoringSummary(
            status="sampled",
            contract_version="1",
            sampler="emcee",
            credible_interval=0.9,
            rejection_log_bayes_factor_threshold=-5.0,
            observed_transit_count=3,
            sample_count=100,
            requested_sample_count=100,
            warmup_draws=50,
            nuisance_parameters={"epoch": BayesianPosteriorInterval(0, 0, 0)},
            mass_limits=limits,
            reference_solution={"epoch": 0},
            diagnostics=None,
            posterior_samples={"epoch": [0.0]}
        )
        self.sim.mass_thresholds = MassThresholds(
            chi2=np.array([10.0]),
            rms=np.array([10.0]),
            bayesian=bayesian
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scoring.npz")
            self.sim.save_scoring_summary(path)
            
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            sim2.load_scoring_summary(path)
            
            loaded_bayesian = sim2.mass_thresholds.bayesian
            self.assertIsNotNone(loaded_bayesian)
            self.assertEqual(loaded_bayesian.status, "sampled")
            self.assertEqual(loaded_bayesian.mass_limits.credible_upper_bound, (10.0,))

    def test_cache_files_are_npz_format(self):
        self.sim.ttv_results = [np.ones(4) for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            self.sim.save_ttv_grid_cache(path)
            with open(path, "rb") as f:
                header = f.read(4)
                self.assertEqual(header, b"PK\x03\x04") # npz is a zip file

    def test_cache_schema_version_mismatch_raises(self):
        self.sim.ttv_results = [np.ones(4) for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            self.sim.save_ttv_grid_cache(path)
            
            # Tamper with the file
            data = dict(np.load(path))
            data["cache_schema_version"] = np.array(["999"])
            np.savez_compressed(path, **data)
            
            with self.assertRaisesRegex(ValueError, "schema version"):
                self.sim.load_ttv_grid_cache(path)

    def test_invalid_ttv_results_dimensionality_raises(self):
        self.sim.ttv_results = [np.ones(4) for _ in range(4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            self.sim.save_ttv_grid_cache(path)

            data = dict(np.load(path))
            data["ttv_results"] = np.array([0.0, 1.0, 2.0, 3.0])
            np.savez_compressed(path, **data)

            with self.assertRaisesRegex(ValueError, "ttv_results dimensionality mismatch"):
                self.sim.load_ttv_grid_cache(path)

    def test_save_and_load_checkpoint_round_trips_all_results(self):
        self.sim.ttv_results = [np.ones(4) for _ in range(4)]
        self.sim.megno_results = [1.0, 2.0, 3.0, 4.0]
        self.sim.mass_thresholds = MassThresholds(
            chi2=np.array([10.0]),
            rms=np.array([10.0]),
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            self.sim.save_checkpoint(tmpdir)
            
            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            sim2.load_checkpoint(tmpdir)
            
            self.assertIsNotNone(sim2.ttv_results)
            self.assertIsNotNone(sim2.megno_results)
            self.assertIsNotNone(sim2.mass_thresholds)

    def test_save_checkpoint_accepts_numpy_array_ttv_results(self):
        self.sim.ttv_results = np.ones((4, 4))
        self.sim.ttv_rebound = np.array(self.sim.ttv_results)

        with tempfile.TemporaryDirectory() as tmpdir:
            self.sim.save_checkpoint(tmpdir)

            sim2 = TTVSimulation(
                epochs=self.sim.epochs,
                ttv_mcmc=self.sim.ttv_mcmc,
                ttv_err=self.sim.ttv_err,
                rs=self.sim.rs,
                mp2s=self.sim.mp2s,
                prop=self.prop,
            )
            sim2.load_checkpoint(tmpdir)

            np.testing.assert_allclose(sim2.ttv_rebound, self.sim.ttv_rebound)

    def test_overwrite_cache_forces_recomputation(self):
        self.sim.ttv_results = [np.ones(4) for _ in range(4)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ttv.npz")
            self.sim.save_ttv_grid_cache(path)
            
            class FakePool:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def imap(self, func, params):
                    return iter([np.zeros(4) for _ in params])

            with mock.patch("cmat.ttv_sim.get_context") as mock_context:
                mock_context.return_value.Pool.return_value = FakePool()
                self.sim.get_ttv_rebound_all(use_cache=True, cache_path=path, overwrite_cache=True)
                
            np.testing.assert_allclose(self.sim.ttv_results[0], np.zeros(4))
