import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cmat.ccs.data import TimingData, convert_wd_style_csv, load_timings_csv


class DataContractTests(unittest.TestCase):
    def test_reject_invalid_units_and_scales(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.csv"
            path.write_text("epoch,tmid,terr\n0,2459000,0.0001\n")
            with self.assertRaisesRegex(ValueError, "Canonical"):
                load_timings_csv(path)
            with self.assertRaisesRegex(ValueError, "assume_bjd_tdb"):
                convert_wd_style_csv(path, Path(temp)/"converted.csv")

    def test_duplicate_epochs_retained_and_legacy_converts_seconds(self):
        with tempfile.TemporaryDirectory() as temp:
            source, dest = Path(temp)/"old.csv", Path(temp)/"new.csv"
            with source.open("w", newline="") as handle:
                w = csv.writer(handle)
                w.writerow(["epoch", "tmid", "terr", "source", "telescope"])
                for ep in (0, 1, 1, 2):
                    w.writerow([ep, 2459000+ep*1.5, 1/86400, "survey", "instrument"])
            convert_wd_style_csv(source, dest, assume_bjd_tdb=True)
            d = load_timings_csv(dest)
            self.assertEqual(d.n, 4)
            np.testing.assert_allclose(d.sigma_sec, 1.0)
            self.assertEqual(d.instrument[0], "instrument")
            with self.assertRaises(FileExistsError):
                convert_wd_style_csv(source, dest, assume_bjd_tdb=True)

    def test_reject_nonpositive_uncertainties(self):
        with self.assertRaises(ValueError):
            TimingData(np.array([0, 1, 2]),
                       np.array([2459000., 2459001., 2459002.]),
                       np.array([1., 0., 1.]))
