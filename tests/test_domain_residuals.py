import unittest

import numpy as np

from cmat import scoring
from cmat.domain.residuals import chi2_with_epoch_shift, rms


class DomainResidualsTests(unittest.TestCase):
    def test_rms_matches_legacy_scoring_helper(self):
        residuals = np.array([3.0, 4.0])

        self.assertAlmostEqual(rms(residuals), 5.0 / np.sqrt(2.0))
        self.assertAlmostEqual(scoring.get_rms(residuals), rms(residuals))

    def test_chi2_with_epoch_shift_matches_best_alignment(self):
        epochs = np.array([10, 11, 12])
        observed = np.array([1.0, 2.0, 3.0])
        errors = np.ones_like(observed)
        simulated = np.array([5.0, 1.0, 2.0, 3.0])

        expected = chi2_with_epoch_shift(simulated, observed, errors, epochs)

        self.assertAlmostEqual(expected, 0.0)
        self.assertAlmostEqual(
            scoring.get_chi2(simulated, epochs, observed, errors), expected
        )

    def test_chi2_with_epoch_shift_prefers_recovered_signal(self):
        epochs = np.array([10, 11, 12, 13])
        errors = np.full(4, 0.2)
        injected_signal = np.array([4.0, 1.5, -2.0, 3.0, -1.0, 0.5])
        recovered = injected_signal[1:5] + np.array([0.05, -0.05, 0.1, -0.1])
        mismatched_signal = np.array([4.0, -1.5, 2.0, -3.0, 1.0, -0.5])

        recovered_score = chi2_with_epoch_shift(
            injected_signal, recovered, errors, epochs
        )
        mismatched_score = chi2_with_epoch_shift(
            mismatched_signal, recovered, errors, epochs
        )

        self.assertLess(recovered_score, 1.0)
        self.assertGreater(mismatched_score, recovered_score)

    def test_chi2_with_epoch_shift_handles_nontrivial_epoch_offset(self):
        epochs = np.array([5, 6, 7, 8])
        observed = np.array([2.0, -1.0, 0.5, 1.0])
        errors = np.full(4, 0.5)
        simulated = np.array([9.0, 2.0, -1.0, 0.5, 1.0, 4.0])

        score = chi2_with_epoch_shift(simulated, observed, errors, epochs)

        self.assertAlmostEqual(score, 0.0)
        self.assertAlmostEqual(
            scoring.get_chi2(simulated, epochs, observed, errors), score
        )


if __name__ == "__main__":
    unittest.main()
