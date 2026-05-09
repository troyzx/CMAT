import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def load_ttv_sim_module():
    module_path = Path(__file__).resolve().parents[1] / "cmat" / "ttv_sim.py"
    spec = importlib.util.spec_from_file_location("ttv_sim_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TtvScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ttv_sim = load_ttv_sim_module()

    def test_get_rms_returns_root_mean_square(self):
        residuals = np.array([3.0, 4.0])

        self.assertAlmostEqual(
            self.ttv_sim.get_rms(residuals),
            np.sqrt((3.0**2 + 4.0**2) / 2.0),
        )

    def test_get_chi2_uses_best_epoch_alignment(self):
        epoch = np.array([10, 11, 12])
        ttv_mcmc = np.array([1.0, 2.0, 3.0])
        ttv_err = np.ones_like(ttv_mcmc)
        ttv_rebound = np.array([5.0, 1.0, 2.0, 3.0])

        self.assertAlmostEqual(
            self.ttv_sim.get_chi2(ttv_rebound, epoch, ttv_mcmc, ttv_err),
            0.0,
        )

    def test_get_m_crit_returns_first_rejected_mass_per_period_ratio(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = self.ttv_sim.ttv_sim(
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


if __name__ == "__main__":
    unittest.main()
