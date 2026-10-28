import unittest
from unittest.mock import patch

import numpy as np

from cmat.ccs.data import TimingData
from cmat.ccs.search import compare_candidate


class SearchInterfaceTests(unittest.TestCase):
    def test_same_observations_and_ephemeris_across_four_models(self):
        e = np.array([0, 2, 5, 12, 19, 28, 39])
        s = np.sin(e/5)
        data = TimingData(e, 2459000+e*2+(s+0.3)/86400,
                          np.full(len(e), .2))
        def fake_integrator(epoch, **kwargs):
            self.assertEqual(list(epoch), list(e))
            return dict(M0=np.zeros(len(e)), M1=s,
                        M2=np.zeros(len(e)), M3=s)
        with patch("cmat.ccs.search.companion_signals", fake_integrator):
            result = compare_candidate(data, star_mass_msun=1.,
                                       inner_mass_mjup=1., outer_mass_mjup=1.,
                                       outer_period_day=100., outer_phase_deg=0.)
        self.assertAlmostEqual(result["scores"]["M1"]["chi2"],
                               result["scores"]["M3"]["chi2"])
        self.assertLess(result["scores"]["M3"]["chi2"],
                        result["scores"]["M0"]["chi2"])
        self.assertIn("NOT_DETECTION", result["status"])
