import unittest

import numpy as np

from cmat.ccs.ltte import circular_ltte_seconds


class LtteTests(unittest.TestCase):
    def test_massless_zero_and_half_orbit_sign(self):
        t = np.array([0., 25., 50., 75.])
        params = dict(star_mass_msun=1., outer_period_day=100.)
        np.testing.assert_array_equal(circular_ltte_seconds(t,
                                      outer_mass_mjup=0., **params), 0.)
        y = circular_ltte_seconds(t, outer_mass_mjup=1., **params)
        self.assertGreater(y[1], 0)
        self.assertAlmostEqual(y[1], -y[3], places=9)

    def test_mass_scaling_and_inclination(self):
        t = np.array([25.])
        p = dict(star_mass_msun=1., outer_period_day=100., time_day=t)
        a = circular_ltte_seconds(outer_mass_mjup=1., **p)[0]
        b = circular_ltte_seconds(outer_mass_mjup=2., **p)[0]
        self.assertAlmostEqual(b/a, 2., delta=0.01)
        z = circular_ltte_seconds(outer_mass_mjup=1., inclination_deg=0., **p)[0]
        self.assertAlmostEqual(z, 0., places=12)
