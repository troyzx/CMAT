import importlib
import os
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from matplotlib import pyplot as plt

from cmat import TTVSimulation, scoring
from cmat.config import BayesianScoringConfig
from cmat.scoring import BayesianMassThresholdScorer, MassThresholds, make_mass_threshold_scorer
from cmat.ttv_sim import ttv_sim


class TtvScoringTests(unittest.TestCase):
    def test_get_rms_returns_root_mean_square(self):
        residuals = np.array([3.0, 4.0])

        self.assertAlmostEqual(
            scoring.get_rms(residuals),
            np.sqrt((3.0**2 + 4.0**2) / 2.0),
        )

    def test_get_chi2_uses_best_epoch_alignment(self):
        epoch = np.array([10, 11, 12])
        ttv_mcmc = np.array([1.0, 2.0, 3.0])
        ttv_err = np.ones_like(ttv_mcmc)
        ttv_rebound = np.array([5.0, 1.0, 2.0, 3.0])

        self.assertAlmostEqual(
            scoring.get_chi2(ttv_rebound, epoch, ttv_mcmc, ttv_err),
            0.0,
        )

    def test_get_chi2_prefers_recovering_injected_signal(self):
        epoch = np.array([10, 11, 12, 13])
        ttv_err = np.full(4, 0.2)
        injected_signal = np.array([4.0, 1.5, -2.0, 3.0, -1.0, 0.5])
        recovered_ttv = injected_signal[1:5] + np.array([0.05, -0.05, 0.1, -0.1])
        mismatched_signal = np.array([4.0, -1.5, 2.0, -3.0, 1.0, -0.5])

        recovered_score = scoring.get_chi2(
            injected_signal, epoch, recovered_ttv, ttv_err
        )
        mismatched_score = scoring.get_chi2(
            mismatched_signal, epoch, recovered_ttv, ttv_err
        )

        self.assertLess(recovered_score, 1.0)
        self.assertGreater(mismatched_score, recovered_score)

    def test_legacy_scoring_functions_remain_on_ttv_module(self):
        ttv_module = importlib.import_module("cmat.ttv_sim")

        self.assertIs(ttv_module.get_rms, scoring.get_rms)
        self.assertIs(ttv_module.get_chi2, scoring.get_chi2)

    def test_get_m_crit_returns_first_rejected_mass_per_period_ratio(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        low_signal = np.array([0.5, 0.5, 0.5, 0.5])
        high_signal = np.array([10.0, 10.0, 10.0, 10.0])
        sim.ttv_results = [
            low_signal,
            high_signal,
            high_signal,
            high_signal,
        ]

        chi2_limit, rms_limit = sim.get_m_crit()

        np.testing.assert_array_equal(chi2_limit, np.array([20.0, 10.0]))
        np.testing.assert_array_equal(rms_limit, np.array([20.0, 10.0]))
        scoring_summary = sim.get_scoring_summary()
        self.assertEqual(scoring_summary["backend"], "chi2_rms")
        np.testing.assert_array_equal(scoring_summary["chi2"], [20.0, 10.0])
        np.testing.assert_array_equal(scoring_summary["rms"], [20.0, 10.0])

    def test_get_m_crit_retains_chi2_reduced_chi2_and_relative_log_likelihood_surfaces(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        low_signal = np.array([0.5, 0.5, 0.5, 0.5])
        high_signal = np.array([10.0, 10.0, 10.0, 10.0])
        sim.ttv_results = [
            low_signal,
            high_signal,
            high_signal,
            high_signal,
        ]

        sim.get_m_crit()

        expected_chi2 = np.array([[0.75, 243.0], [243.0, 243.0]])
        np.testing.assert_allclose(sim.get_chi2_surface(), expected_chi2)
        np.testing.assert_allclose(
            sim.get_reduced_chi2_surface(),
            expected_chi2 / 3.0,
        )
        np.testing.assert_allclose(
            sim.get_relative_log_likelihood_surface(),
            -0.5 * expected_chi2,
        )

        summary = sim.get_scoring_summary()
        self.assertEqual(summary["chi2_degrees_of_freedom"], 3)
        self.assertEqual(summary["period_ratios"], [1.0, 2.0])
        self.assertEqual(summary["companion_masses"], [10.0, 20.0])
        np.testing.assert_allclose(summary["chi2_surface"], expected_chi2)
        np.testing.assert_allclose(summary["reduced_chi2_surface"], expected_chi2 / 3.0)
        np.testing.assert_allclose(
            summary["relative_log_likelihood_surface"],
            -0.5 * expected_chi2,
        )

    def test_plot_chi2_contour_draws_score_surface(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.ttv_results = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]
        sim.get_m_crit()

        fig, ax = sim.plot_chi2_contour(vmin=0.0, vmax=sim.mass_thresholds.chi2_threshold)

        self.assertEqual(ax.get_xlabel(), r"$P_2/P_1$")
        self.assertEqual(ax.get_ylabel(), r"Mass [$M_\oplus$]")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertGreaterEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_plot_chi2_contour_supports_reduced_chi2_surface(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.ttv_results = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]
        sim.get_m_crit()

        fig, _ = sim.plot_chi2_contour(
            statistic="reduced_chi2",
            vmin=0.0,
            vmax=sim.mass_thresholds.chi2_threshold / 3.0,
        )

        self.assertIn("reduced", fig.axes[1].get_ylabel())
        plt.close(fig)

    def test_plot_chi2_contour_supports_relative_log_likelihood_surface(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.ttv_results = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]
        sim.get_m_crit()

        fig, _ = sim.plot_chi2_contour(
            statistic="relative_log_likelihood",
            levels=4,
        )

        self.assertIn("log likelihood", fig.axes[1].get_ylabel())
        plt.close(fig)

    def test_plot_chi2_contour_draws_constant_finite_surface_with_pcolor(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        identical_signal = np.array([1.0, 1.0, 1.0, 1.0])
        sim.ttv_results = [
            identical_signal,
            identical_signal,
            identical_signal,
            identical_signal,
        ]
        sim.get_m_crit()

        fig, ax = sim.plot_chi2_contour()

        self.assertEqual(ax.get_yscale(), "log")
        self.assertGreaterEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_get_m_crit_ignores_zero_signal_rows(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.ttv_results = [
            np.zeros(4),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]

        chi2_limit, rms_limit = sim.get_m_crit()

        np.testing.assert_array_equal(chi2_limit, np.array([20.0]))
        np.testing.assert_array_equal(rms_limit, np.array([20.0]))

    def test_get_m_crit_excludes_nonfinite_rows(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.ttv_results = [
            np.full(4, np.nan),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]

        chi2_limit, rms_limit = sim.get_m_crit()

        np.testing.assert_array_equal(chi2_limit, np.array([20.0]))
        np.testing.assert_array_equal(rms_limit, np.array([20.0]))

    def test_get_m_crit_returns_empty_limits_when_no_mass_is_rejected(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([5.0, 5.0, 5.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.ttv_results = [
            np.array([0.0, 5.0, 5.0, 5.0]),
            np.array([0.0, 5.0, 5.0, 5.0]),
            np.array([0.0, 5.0, 5.0, 5.0]),
            np.array([0.0, 5.0, 5.0, 5.0]),
        ]

        chi2_limit, rms_limit = sim.get_m_crit()

        np.testing.assert_array_equal(chi2_limit, np.array([]))
        np.testing.assert_array_equal(rms_limit, np.array([]))

    def test_get_m_crit_recovers_injected_signal_until_larger_mass_is_rejected(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        injected_signal = np.array([1.0, -1.0, 2.0])
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=injected_signal,
            ttv_err=np.full(3, 0.2),
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.ttv_results = [
            np.array([1.0, -1.0, 2.0, 0.5]),
            np.array([0.9, -0.9, 1.8, 0.4]),
            np.array([4.0, -4.0, 8.0, 2.0]),
            np.array([3.6, -3.6, 7.2, 1.6]),
        ]

        chi2_limit, rms_limit = sim.get_m_crit()

        np.testing.assert_array_equal(chi2_limit, np.array([20.0, 20.0]))
        np.testing.assert_array_equal(rms_limit, np.array([20.0, 20.0]))

    def test_preferred_ttvsimulation_alias_exposes_mass_threshold_method(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        simulation.ttv_results = [
            np.array([0.0, 5.0, 5.0, 5.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]

        expected_chi2_limit, expected_rms_limit = simulation.get_m_crit()
        chi2_limit, rms_limit = simulation.get_critical_masses()

        np.testing.assert_array_equal(chi2_limit, expected_chi2_limit)
        np.testing.assert_array_equal(rms_limit, expected_rms_limit)

    def test_get_ttv_rebound_all_uses_configured_execution_controls(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=prop,
        )
        simulation.worker_count = 2
        simulation.start_method = "spawn"
        simulation.show_progress = False

        context_calls = []

        class FakePool:
            def __init__(self, process_count):
                self.process_count = process_count

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, parameters):
                return iter([np.array([0.1, 0.2, 0.3, 0.4]) for _ in parameters])

        class FakeContext:
            def Pool(self, process_count):
                context_calls.append(process_count)
                return FakePool(process_count)

        with mock.patch("cmat.ttv_sim.get_context", return_value=FakeContext()) as get_context_mock, mock.patch(
            "cmat.ttv_sim.tqdm"
        ) as tqdm_mock:
            result = simulation.get_ttv_rebound_all()

        get_context_mock.assert_called_once_with("spawn")
        tqdm_mock.assert_not_called()
        self.assertEqual(context_calls, [2])
        self.assertEqual(result.shape, (1, 4))

    def test_run_megno_uses_configured_execution_controls(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=prop,
        )
        simulation.worker_count = 3
        simulation.start_method = "forkserver"
        simulation.show_progress = False

        context_calls = []

        class FakePool:
            def __init__(self, process_count):
                self.process_count = process_count

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, parameters):
                return iter([2.1 for _ in parameters])

        class FakeContext:
            def Pool(self, process_count):
                context_calls.append(process_count)
                return FakePool(process_count)

        with mock.patch("cmat.ttv_sim.get_context", return_value=FakeContext()) as get_context_mock, mock.patch(
            "cmat.ttv_sim.tqdm"
        ) as tqdm_mock:
            result = simulation.run_megno()

        get_context_mock.assert_called_once_with("forkserver")
        tqdm_mock.assert_not_called()
        self.assertEqual(context_calls, [3])
        self.assertEqual(result, [2.1])

    def test_get_ttv_rebound_all_allows_legacy_thread_override(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=prop,
        )
        simulation.worker_count = 8
        simulation.start_method = "spawn"
        simulation.show_progress = False

        context_calls = []

        class FakePool:
            def __init__(self, process_count):
                self.process_count = process_count

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, parameters):
                return iter([np.array([0.1, 0.2, 0.3, 0.4]) for _ in parameters])

        class FakeContext:
            def Pool(self, process_count):
                context_calls.append(process_count)
                return FakePool(process_count)

        with mock.patch("cmat.ttv_sim.get_context", return_value=FakeContext()):
            simulation.get_ttv_rebound_all(number_of_thread=4)

        self.assertEqual(context_calls, [4])

    def test_get_ttv_rebound_all_rejects_non_integral_thread_count(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=prop,
        )

        with self.assertRaises(TypeError):
            simulation.get_ttv_rebound_all(number_of_thread=3.5)

    def test_run_megno_allows_legacy_thread_override(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=prop,
        )
        simulation.worker_count = 8
        simulation.start_method = "spawn"
        simulation.show_progress = False

        context_calls = []

        class FakePool:
            def __init__(self, process_count):
                self.process_count = process_count

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, parameters):
                return iter([2.1 for _ in parameters])

        class FakeContext:
            def Pool(self, process_count):
                context_calls.append(process_count)
                return FakePool(process_count)

        with mock.patch("cmat.ttv_sim.get_context", return_value=FakeContext()):
            simulation.run_megno(number_of_threads=4)

        self.assertEqual(context_calls, [4])

    def test_get_m_crit_can_delegate_to_custom_scoring_backend(self):
        class StubScoringBackend:
            def __init__(self):
                self.calls = []

            def critical_masses(
                self,
                *,
                ttv_results,
                epoch,
                ttv_mcmc,
                ttv_err,
                period_ratios,
                companion_masses,
            ):
                self.calls.append(
                    {
                        "ttv_results": list(ttv_results),
                        "epoch": epoch.copy(),
                        "ttv_mcmc": ttv_mcmc.copy(),
                        "ttv_err": ttv_err.copy(),
                        "period_ratios": period_ratios.copy(),
                        "companion_masses": companion_masses.copy(),
                    }
                )
                return MassThresholds(
                    chi2=np.array([42.0]),
                    rms=np.array([24.0]),
                )

        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        scoring_backend = StubScoringBackend()
        simulation = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
            scoring_backend=scoring_backend,
        )
        simulation.ttv_results = [
            np.array([0.0, 5.0, 5.0, 5.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]

        chi2_limit, rms_limit = simulation.get_m_crit()

        np.testing.assert_array_equal(chi2_limit, np.array([42.0]))
        np.testing.assert_array_equal(rms_limit, np.array([24.0]))
        self.assertEqual(len(scoring_backend.calls), 1)
        self.assertEqual(simulation.get_scoring_summary()["backend"], "chi2_rms")
        np.testing.assert_array_equal(
            scoring_backend.calls[0]["period_ratios"],
            np.array([1.0]),
        )
        np.testing.assert_array_equal(
            scoring_backend.calls[0]["companion_masses"],
            np.array([10.0, 20.0]),
        )

        with self.assertRaisesRegex(ValueError, "chi2_surface"):
            simulation.get_chi2_surface()

    def test_scorer_factory_supports_bayesian_contract_backend(self):
        scorer = make_mass_threshold_scorer("bayesian")

        self.assertIsInstance(scorer, BayesianMassThresholdScorer)

    def test_bayesian_backend_returns_posterior_mass_summary(self):
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2, 3, 4]),
            ttv_mcmc=np.array([0.3, -0.2, 0.1, -0.25, 0.35]),
            ttv_err=np.full(5, 0.05),
            rs=np.array([1.5]),
            mp2s=np.array([10.0, 20.0]),
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
            scoring_backend=BayesianMassThresholdScorer(
                config=BayesianScoringConfig(
                    credible_interval=0.8,
                    posterior_sample_count=64,
                    warmup_draws=24,
                )
            ),
        )
        simulation.ttv_results = [
            np.array([0.3, -0.2, 0.1, -0.25, 0.35, 0.0]),
            np.array([3.0, -2.0, 1.0, -2.5, 3.5, 0.0]),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2_limit, rms_limit = simulation.get_m_crit()
        summary = simulation.get_scoring_summary()

        np.testing.assert_array_equal(chi2_limit, np.array([]))
        np.testing.assert_array_equal(rms_limit, np.array([]))
        self.assertEqual(summary["backend"], "bayesian")
        self.assertEqual(summary["bayesian"]["status"], "posterior_sampled")
        self.assertEqual(summary["bayesian"]["contract_version"], "stage4_phase2")
        self.assertEqual(summary["bayesian"]["sampler"], "emcee")
        self.assertEqual(summary["bayesian"]["credible_interval"], 0.8)
        self.assertEqual(summary["bayesian"]["rejection_log_bayes_factor_threshold"], -5.0)
        self.assertEqual(summary["bayesian"]["sample_count"], 64)
        self.assertEqual(summary["bayesian"]["requested_sample_count"], 64)
        self.assertEqual(summary["bayesian"]["warmup_draws"], 24)
        self.assertEqual(summary["bayesian"]["nuisance_parameters"].keys(), {"epoch_shift", "baseline_offset", "jitter"})
        self.assertEqual(summary["bayesian"]["mass_limits"]["units"], "earth_masses")
        self.assertEqual(summary["bayesian"]["mass_limits"]["period_ratios"], [1.5])
        self.assertEqual(summary["bayesian"]["mass_limits"]["evaluated_masses"], [10.0, 20.0])
        self.assertEqual(summary["bayesian"]["mass_limits"]["credible_upper_bound"], [10.0])
        self.assertEqual(summary["bayesian"]["mass_limits"]["rejection_upper_bound"], [20.0])
        self.assertEqual(summary["bayesian"]["mass_limits"]["upper_bound"], [10.0])
        self.assertEqual(
            summary["bayesian"]["mass_limits"]["upper_bound"],
            summary["bayesian"]["mass_limits"]["credible_upper_bound"],
        )
        self.assertGreater(
            summary["bayesian"]["mass_limits"]["posterior_by_period_ratio"][0]["model_probabilities"][1],
            0.9,
        )

    def test_bayesian_backend_supports_nuisance_parameter_subsets(self):
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2, 3, 4]),
            ttv_mcmc=np.array([0.3, -0.2, 0.1, -0.25, 0.35]),
            ttv_err=np.full(5, 0.05),
            rs=np.array([1.5]),
            mp2s=np.array([10.0, 20.0]),
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
            scoring_backend=BayesianMassThresholdScorer(
                config=BayesianScoringConfig(
                    credible_interval=0.8,
                    posterior_sample_count=64,
                    warmup_draws=24,
                    nuisance_parameters=("epoch_shift",),
                )
            ),
        )
        simulation.ttv_results = [
            np.array([0.3, -0.2, 0.1, -0.25, 0.35, 0.0]),
            np.array([3.0, -2.0, 1.0, -2.5, 3.5, 0.0]),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2_limit, rms_limit = simulation.get_m_crit()
        summary = simulation.get_scoring_summary()

        np.testing.assert_array_equal(chi2_limit, np.array([]))
        np.testing.assert_array_equal(rms_limit, np.array([]))
        self.assertEqual(summary["backend"], "bayesian")
        self.assertEqual(summary["bayesian"]["nuisance_parameters"].keys(), {"epoch_shift"})
        self.assertEqual(summary["bayesian"]["sample_count"], 64)
        self.assertEqual(summary["bayesian"]["requested_sample_count"], 64)
        self.assertEqual(summary["bayesian"]["posterior_samples"], None)
        self.assertEqual(summary["bayesian"]["mass_limits"]["period_ratios"], [1.5])

    def test_bayesian_backend_excludes_invalid_models_from_evidence(self):
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2, 3, 4]),
            ttv_mcmc=np.array([0.3, -0.2, 0.1, -0.25, 0.35]),
            ttv_err=np.full(5, 0.05),
            rs=np.array([1.5]),
            mp2s=np.array([10.0, 20.0]),
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
            scoring_backend=BayesianMassThresholdScorer(
                config=BayesianScoringConfig(
                    credible_interval=0.8,
                    posterior_sample_count=64,
                    warmup_draws=24,
                )
            ),
        )
        simulation.ttv_results = [
            np.full(6, np.nan),
            np.array([0.3, -0.2, 0.1, -0.25, 0.35, 0.0]),
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulation.get_m_crit()
        posterior = simulation.get_scoring_summary()["bayesian"]["mass_limits"][
            "posterior_by_period_ratio"
        ][0]

        self.assertEqual(posterior["model_probabilities"][1], 0.0)
        self.assertGreater(posterior["model_probabilities"][2], 0.0)

    def test_bayesian_backend_uses_log_evidence_for_model_probabilities(self):
        scorer = BayesianMassThresholdScorer(
            config=BayesianScoringConfig(
                posterior_sample_count=8,
                warmup_draws=4,
                nuisance_parameters=("epoch_shift",),
            )
        )
        fake_result = scoring._BayesianModelResult
        neutral_interval = {"epoch_shift": scoring.BayesianPosteriorInterval(0.0, 0.0, 0.0)}

        with mock.patch.object(
            BayesianMassThresholdScorer,
            "_sample_model",
            side_effect=[
                fake_result(
                    support_score=0.0,
                    log_evidence=0.0,
                    sample_count=8,
                    intervals=neutral_interval,
                    mean_acceptance_fraction=0.5,
                    alignment_count=1,
                ),
                fake_result(
                    support_score=8.0,
                    log_evidence=-8.0,
                    sample_count=8,
                    intervals=neutral_interval,
                    mean_acceptance_fraction=0.5,
                    alignment_count=1,
                ),
                fake_result(
                    support_score=7.0,
                    log_evidence=-9.0,
                    sample_count=8,
                    intervals=neutral_interval,
                    mean_acceptance_fraction=0.5,
                    alignment_count=1,
                ),
            ],
        ):
            result = scorer.critical_masses(
                ttv_results=[
                    np.zeros(3, dtype=float),
                    np.ones(3, dtype=float),
                ],
                epoch=np.array([0, 1, 2], dtype=int),
                ttv_mcmc=np.array([0.0, 0.0, 0.0], dtype=float),
                ttv_err=np.ones(3, dtype=float),
                period_ratios=np.array([1.2], dtype=float),
                companion_masses=np.array([10.0, 20.0], dtype=float),
            )

        posterior = result.bayesian.mass_limits.posterior_by_period_ratio[0]
        self.assertIsNone(posterior.best_mass)
        self.assertIsNone(posterior.credible_upper_bound)
        self.assertEqual(posterior.rejection_upper_bound, 10.0)
        self.assertIsNone(posterior.upper_bound)
        self.assertEqual(posterior.upper_bound, posterior.credible_upper_bound)
        self.assertGreater(posterior.model_probabilities[0], posterior.model_probabilities[1])
        self.assertGreater(posterior.posterior_predictive_score[1], posterior.posterior_predictive_score[0])
        np.testing.assert_allclose(posterior.log_evidence, np.array([0.0, -8.0, -9.0]))

    def test_bayesian_rejection_upper_bound_scans_only_high_mass_side_of_best_model(self):
        scorer = BayesianMassThresholdScorer(
            config=BayesianScoringConfig(
                posterior_sample_count=8,
                warmup_draws=4,
                nuisance_parameters=("epoch_shift",),
            )
        )
        fake_result = scoring._BayesianModelResult
        neutral_interval = {"epoch_shift": scoring.BayesianPosteriorInterval(0.0, 0.0, 0.0)}

        with mock.patch.object(
            BayesianMassThresholdScorer,
            "_sample_model",
            side_effect=[
                fake_result(
                    support_score=0.0,
                    log_evidence=-12.0,
                    sample_count=8,
                    intervals=neutral_interval,
                    mean_acceptance_fraction=0.5,
                    alignment_count=1,
                ),
                fake_result(
                    support_score=-8.0,
                    log_evidence=-8.0,
                    sample_count=8,
                    intervals=neutral_interval,
                    mean_acceptance_fraction=0.5,
                    alignment_count=1,
                ),
                fake_result(
                    support_score=0.0,
                    log_evidence=0.0,
                    sample_count=8,
                    intervals=neutral_interval,
                    mean_acceptance_fraction=0.5,
                    alignment_count=1,
                ),
                fake_result(
                    support_score=-7.0,
                    log_evidence=-7.0,
                    sample_count=8,
                    intervals=neutral_interval,
                    mean_acceptance_fraction=0.5,
                    alignment_count=1,
                ),
            ],
        ):
            result = scorer.critical_masses(
                ttv_results=[
                    np.zeros(3, dtype=float),
                    np.ones(3, dtype=float),
                    np.full(3, 2.0, dtype=float),
                ],
                epoch=np.array([0, 1, 2], dtype=int),
                ttv_mcmc=np.array([0.0, 0.0, 0.0], dtype=float),
                ttv_err=np.ones(3, dtype=float),
                period_ratios=np.array([1.2], dtype=float),
                companion_masses=np.array([10.0, 20.0, 30.0], dtype=float),
            )

        posterior = result.bayesian.mass_limits.posterior_by_period_ratio[0]
        self.assertEqual(posterior.best_mass, 20.0)
        self.assertEqual(posterior.rejection_upper_bound, 30.0)

    def test_bayesian_backend_retains_representative_posterior_draws(self):
        class FakeSampler:
            def __init__(self, flat_chain):
                self._flat_chain = flat_chain
                self.acceptance_fraction = np.array([0.5] * 12)
                self.random_state = None

            def run_mcmc(self, initial_position, nsteps, progress=False):
                return None

            def get_chain(self, discard=0, flat=False):
                return self._flat_chain

        scorer = BayesianMassThresholdScorer(
            config=BayesianScoringConfig(
                posterior_sample_count=4,
                warmup_draws=2,
                nuisance_parameters=("baseline_offset",),
                store_chains=True,
            )
        )
        flat_chain = np.arange(8, dtype=float).reshape(-1, 1)

        with mock.patch(
            "cmat.scoring.emcee.EnsembleSampler",
            return_value=FakeSampler(flat_chain),
        ):
            result = scorer._sample_model(
                ttv_rebound=np.zeros(5, dtype=float),
                epoch=np.array([0, 1, 2], dtype=int),
                observed_ttv=np.array([0.1, 0.2, 0.3], dtype=float),
                observed_err=np.full(3, 0.05, dtype=float),
                nuisance_parameters=("baseline_offset",),
                seed_hint=("representative-retention",),
            )

        self.assertEqual(result.sample_count, 4)
        self.assertEqual(result.posterior_samples["baseline_offset"], [1.0, 3.0, 5.0, 7.0])

    def test_bayesian_backend_seeds_emcee_random_state_from_stable_seed(self):
        class FakeSampler:
            def __init__(self):
                self.acceptance_fraction = np.array([0.5] * 12)
                self.random_state = None

            def run_mcmc(self, initial_position, nsteps, progress=False):
                return None

            def get_chain(self, discard=0, flat=False):
                return np.zeros((8, 1), dtype=float)

        fake_sampler = FakeSampler()
        scorer = BayesianMassThresholdScorer(
            config=BayesianScoringConfig(
                posterior_sample_count=4,
                warmup_draws=2,
                nuisance_parameters=("baseline_offset",),
            )
        )

        with mock.patch(
            "cmat.scoring.emcee.EnsembleSampler",
            return_value=fake_sampler,
        ):
            scorer._sample_model(
                ttv_rebound=np.zeros(5, dtype=float),
                epoch=np.array([0, 1, 2], dtype=int),
                observed_ttv=np.array([0.1, 0.2, 0.3], dtype=float),
                observed_err=np.full(3, 0.05, dtype=float),
                nuisance_parameters=("baseline_offset",),
                seed_hint=("sampler-seed",),
            )

        self.assertIsNotNone(fake_sampler.random_state)
        self.assertEqual(fake_sampler.random_state[0], "MT19937")

    def test_get_mass_thresholds_returns_full_bayesian_result_object(self):
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2, 3, 4]),
            ttv_mcmc=np.array([0.3, -0.2, 0.1, -0.25, 0.35]),
            ttv_err=np.full(5, 0.05),
            rs=np.array([1.5]),
            mp2s=np.array([10.0, 20.0]),
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
            scoring_backend=BayesianMassThresholdScorer(
                config=BayesianScoringConfig(
                    credible_interval=0.8,
                    posterior_sample_count=64,
                    warmup_draws=24,
                )
            ),
        )
        simulation.ttv_results = [
            np.array([0.3, -0.2, 0.1, -0.25, 0.35, 0.0]),
            np.array([3.0, -2.0, 1.0, -2.5, 3.5, 0.0]),
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulation.get_m_crit()

        thresholds = simulation.get_mass_thresholds()
        self.assertEqual(thresholds.backend, "bayesian")
        self.assertIsNotNone(thresholds.bayesian)
        self.assertTrue(any("experimental Stage 4 backend" in str(w.message) for w in caught))

    def test_get_scoring_summary_requires_prior_scoring_run(self):
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=[
                {
                    "orbital_distance": 1.0,
                    "orbital_period": 1.0,
                    "Mp": 1.0,
                    "Ms": 1.0,
                }
            ],
        )

        with self.assertRaises(ValueError):
            simulation.get_scoring_summary()

    def test_get_mass_thresholds_requires_prior_scoring_run(self):
        simulation = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=[
                {
                    "orbital_distance": 1.0,
                    "orbital_period": 1.0,
                    "Mp": 1.0,
                    "Ms": 1.0,
                }
            ],
        )

        with self.assertRaises(ValueError):
            simulation.get_mass_thresholds()


if __name__ == "__main__":
    unittest.main()
