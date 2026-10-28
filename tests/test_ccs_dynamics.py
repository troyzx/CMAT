import importlib.util
import unittest

import numpy as np

from cmat.ccs.dynamics import companion_signals, simulate_transits


@unittest.skipUnless(importlib.util.find_spec("rebound"), "requires REBOUND")
class DynamicsPilotTests(unittest.TestCase):
    def test_two_body_events_follow_reference_period(self):
        epochs = np.array([0, 1, 4, 8])
        series = simulate_transits(epochs, star_mass_msun=1.,
                                   inner_mass_mjup=1., inner_period_day=2.)
        np.testing.assert_array_equal(series.epoch, epochs)
        np.testing.assert_allclose(series.transit_day, 2.*epochs, atol=1e-6)

    def test_outer_zero_signals_are_zero(self):
        models = companion_signals(
            np.array([0, 1, 3]), star_mass_msun=1., inner_mass_mjup=1.,
            inner_period_day=2., outer_mass_mjup=0.,
            outer_period_day=80., outer_phase_deg=60.,
        )
        for key in ("M0", "M1", "M2", "M3"):
            np.testing.assert_allclose(models[key], 0, atol=1e-8)

    def test_safety_cap(self):
        with self.assertRaisesRegex(ValueError, "safety cap"):
            simulate_transits(np.array([0, 5000]), star_mass_msun=1.,
                              inner_mass_mjup=1., inner_period_day=1.)

    def test_ltte_matches_analytic_outer_reflex_at_pilot_precision(self):
        from cmat.ccs.ltte import circular_ltte_seconds
        epochs = np.array([0, 1, 2, 4, 7, 11, 15, 20])
        signals = companion_signals(
            epochs, star_mass_msun=1., inner_mass_mjup=1.,
            inner_period_day=2., outer_mass_mjup=1.,
            outer_period_day=100., outer_phase_deg=30.,
        )
        analytic = circular_ltte_seconds(
            epochs*2., star_mass_msun=1., inner_mass_mjup=1.,
            outer_mass_mjup=1., outer_period_day=100., phase_deg=30.,
        )
        # Difference can have an arbitrary constant origin; compare variations.
        diff = (signals["M2"]-signals["M2"][0])-(analytic-analytic[0])
        self.assertLess(np.max(np.abs(diff)), 0.03)
