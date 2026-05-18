import unittest

import numpy as np
import scipy.stats

from cmat.domain.mass_limits import first_rejected_masses
from cmat.scoring import Chi2AndRmsMassThresholdScorer


class DomainMassLimitsTests(unittest.TestCase):
    def test_first_rejected_masses_returns_first_threshold_crossing_per_column(self):
        score_surface = np.array(
            [
                [0.75, 243.0],
                [243.0, 243.0],
            ]
        )
        valid_mask = np.array(
            [
                [True, True],
                [True, True],
            ]
        )

        rejected = first_rejected_masses(
            score_surface,
            masses=np.array([10.0, 20.0]),
            threshold=scipy.stats.chi2.ppf(0.997, 3),
            valid_mask=valid_mask,
        )

        np.testing.assert_array_equal(rejected, np.array([20.0, 10.0]))

    def test_first_rejected_masses_skips_invalid_and_nonfinite_rows(self):
        score_surface = np.array(
            [
                [np.nan, 100.0],
                [50.0, 200.0],
                [150.0, 300.0],
            ]
        )
        valid_mask = np.array(
            [
                [False, False],
                [True, True],
                [True, True],
            ]
        )

        rejected = first_rejected_masses(
            score_surface,
            masses=np.array([10.0, 20.0, 30.0]),
            threshold=120.0,
            valid_mask=valid_mask,
        )

        np.testing.assert_array_equal(rejected, np.array([30.0, 20.0]))

    def test_first_rejected_masses_omits_columns_without_rejections(self):
        rejected = first_rejected_masses(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            masses=np.array([10.0, 20.0]),
            threshold=10.0,
        )

        np.testing.assert_array_equal(rejected, np.array([]))

    def test_scoring_backend_preserves_legacy_wasp44_mass_limit_fixture(self):
        scorer = Chi2AndRmsMassThresholdScorer()
        low_signal = np.array([0.5, 0.5, 0.5, 0.5])
        high_signal = np.array([10.0, 10.0, 10.0, 10.0])

        result = scorer.critical_masses(
            ttv_results=[low_signal, high_signal, high_signal, high_signal],
            epoch=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            period_ratios=np.array([1.0, 2.0]),
            companion_masses=np.array([10.0, 20.0]),
        )

        np.testing.assert_array_equal(result.chi2, np.array([20.0, 10.0]))
        np.testing.assert_array_equal(result.rms, np.array([20.0, 10.0]))
        np.testing.assert_allclose(
            result.chi2,
            first_rejected_masses(
                result.chi2_surface,
                masses=np.array([10.0, 20.0]),
                threshold=result.chi2_threshold,
                valid_mask=np.isfinite(result.chi2_surface)
                & np.isfinite(result.relative_log_likelihood_surface),
            ),
        )


if __name__ == "__main__":
    unittest.main()
