import unittest

import numpy as np

from cmat.ccs.data import TimingData
from cmat.ccs.likelihood import fit_reference_ephemeris, score_model
from cmat.ccs.validation import synthetic_profile_check


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.e = np.array([0, 1, 3, 4, 9, 25, 40, 51])
        self.signal = 1.1 * np.sin(self.e/5)
        self.data = TimingData(
            self.e,
            2459000 + self.e*1.5 + (self.signal+0.3+0.005*self.e)/86400,
            np.array([.2, .4, 1., .3, .7, .2, .5, .2]),
        )
        self.kwargs = dict(reference_t0_bjd=2459000.0,
                           reference_period_day=1.5, reference_epoch=0)

    def test_linear_profile_removes_offset_and_linear_trend(self):
        x = score_model(self.data, self.signal, **self.kwargs, mode="linear")
        self.assertLess(x.chi2, 1e-4)
        self.assertAlmostEqual(x.offset_sec, 0.3, delta=0.0001)
        self.assertAlmostEqual(x.delta_period_sec, 0.005, delta=0.00001)

    def test_profile_nonincreasing_nested_scores(self):
        a = [score_model(self.data, self.signal, **self.kwargs, mode=kind).chi2
             for kind in ("fixed", "offset", "linear")]
        self.assertGreaterEqual(a[0]+1e-10, a[1])
        self.assertGreaterEqual(a[1]+1e-10, a[2])

    def test_epoch_shift_invariance_and_duplicates(self):
        e = np.r_[self.e, self.e[3]]
        t = np.r_[self.data.time_bjd_tdb, self.data.time_bjd_tdb[3]]
        s = np.r_[self.data.sigma_sec, self.data.sigma_sec[3]]
        d = TimingData(e, t, s)
        a = score_model(d, np.r_[self.signal, self.signal[3]],
                        **self.kwargs, mode="linear")
        kw = dict(self.kwargs)
        kw["reference_epoch"] = -100
        kw["reference_t0_bjd"] -= 150.0
        b = score_model(d, np.r_[self.signal, self.signal[3]],
                        **kw, mode="linear")
        self.assertAlmostEqual(a.chi2, b.chi2, places=6)

    def test_null_ephemeris_baseline(self):
        t, p, e = fit_reference_ephemeris(self.data)
        fit = score_model(self.data, np.zeros(self.data.n),
                          reference_t0_bjd=t, reference_period_day=p,
                          reference_epoch=e, mode="linear")
        self.assertAlmostEqual(fit.offset_sec, 0., delta=1e-3)
        self.assertAlmostEqual(fit.delta_period_sec, 0., delta=1e-6)

    def test_synthetic_demo(self):
        self.assertEqual(synthetic_profile_check()["status"],
                         "SYNTHETIC_PROFILE_PASS")
