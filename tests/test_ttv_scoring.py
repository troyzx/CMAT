import importlib.util
from pathlib import Path
import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
