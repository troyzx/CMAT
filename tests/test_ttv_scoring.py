import importlib
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from cmat import scoring
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


if __name__ == "__main__":
    unittest.main()
