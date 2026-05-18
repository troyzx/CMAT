import unittest

import numpy as np

from cmat.domain.timing import calculate_epochs, linear_ephemeris, timing_residuals


class DomainTimingTests(unittest.TestCase):
    def test_calculate_epochs_rounds_to_nearest_linear_ephemeris_epoch(self):
        transit_times = np.array([100.02, 102.01, 104.03, 106.0])

        epochs = calculate_epochs(transit_times, 100.0, 2.0)

        np.testing.assert_array_equal(epochs, np.array([0, 1, 2, 3]))

    def test_linear_ephemeris_returns_expected_transit_centers(self):
        np.testing.assert_allclose(
            linear_ephemeris(np.array([5, 6, 7]), 100.0, 2.0),
            np.array([110.0, 112.0, 114.0]),
        )

    def test_timing_residuals_respect_nonzero_epoch_offset(self):
        epochs = np.array([5, 6, 7, 8])
        transit_times = np.array([110.0, 112.0001, 113.9998, 116.0001])

        residuals = timing_residuals(transit_times, epochs, 100.0, 2.0)

        np.testing.assert_allclose(
            residuals,
            np.array([0.0, 0.0001, -0.0002, 0.0001]),
        )


if __name__ == "__main__":
    unittest.main()
