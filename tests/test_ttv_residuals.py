import importlib
import unittest

import numpy as np


class TtvResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.base = importlib.import_module("cmat.base")
        except Exception as exc:
            raise unittest.SkipTest(f"cmat.base import is unavailable: {exc}")

    def test_calculate_ttv_removes_linear_ephemeris_and_centers_residuals(self):
        fit = self.base.Fitlpf.__new__(self.base.Fitlpf)
        epochs = np.array([5, 6, 7, 8])
        epoch_offsets = epochs - epochs[0]
        residual_days = np.array([0.0, 1.0, -2.0, 1.0]) * 1e-4
        transit_centers = 100.0 + 2.0 * epoch_offsets + residual_days
        transit_center_errors = np.array([1.0, 2.0, 3.0, 4.0]) * 1e-5

        fit.epochs = epochs
        fit.period = self.base.ufloat(2.0, 1e-3)
        fit.tcs = [
            self.base.ufloat(center, error)
            for center, error in zip(transit_centers, transit_center_errors)
        ]

        fit.calculate_ttv()

        expected_ttv = residual_days * self.base.DAY_TO_SEC
        expected_ttv_err = (
            transit_center_errors + fit.period.s * epoch_offsets
        ) * self.base.DAY_TO_SEC
        np.testing.assert_allclose(fit.ttv_mcmc_raw, expected_ttv, atol=1e-8)
        np.testing.assert_allclose(fit.ttv_mcmc, expected_ttv, atol=1e-8)
        np.testing.assert_allclose(fit.ttv_err, expected_ttv_err, atol=1e-8)
