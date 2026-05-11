import importlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from cmat import TTVSimulation, scoring
from cmat.scoring import MassThresholds
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


if __name__ == "__main__":
    unittest.main()
