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

    class _FakeSeries:
        def __init__(self, mean_value, std_value):
            self._mean_value = mean_value
            self._std_value = std_value

        def mean(self):
            return self._mean_value

        def std(self):
            return self._std_value

    class _FakeSingle:
        def __init__(self, tc_mean, tc_std):
            self._posterior = {
                "tc": TtvResidualTests._FakeSeries(tc_mean, tc_std),
            }

        def posterior_samples(self):
            return self._posterior

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

    def test_get_posterior_samples_derives_epochs_from_zero_epoch_and_period(self):
        fit = self.base.Fitlpf.__new__(self.base.Fitlpf)
        fit.zero_epoch = self.base.ufloat(100.0, 1e-3)
        fit.period = self.base.ufloat(2.0, 1e-4)
        fit.singles = [
            self._FakeSingle(110.0, 0.01),
            self._FakeSingle(112.0, 0.02),
            self._FakeSingle(114.0, 0.03),
        ]

        fit.get_posterior_samples()

        np.testing.assert_array_equal(fit.epochs, np.array([5, 6, 7]))
        np.testing.assert_allclose(
            [tc.n for tc in fit.tcs],
            np.array([110.0, 112.0, 114.0]),
        )
        np.testing.assert_allclose(
            [tc.s for tc in fit.tcs],
            np.array([0.01, 0.02, 0.03]),
        )
