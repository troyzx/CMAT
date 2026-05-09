import csv
from pathlib import Path
import statistics
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "WASP-44 b"


class Wasp44BaselineDataTests(unittest.TestCase):
    def test_timing_residual_csv_matches_baseline_summary(self):
        with (DATA_DIR / "tc_data.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        epochs = [int(row["epochs"]) for row in rows]
        residuals = [float(row["ttv_mcmc"]) for row in rows]
        errors = [float(row["ttv_err"]) for row in rows]

        self.assertEqual(len(rows), 8)
        self.assertEqual(min(epochs), 1218)
        self.assertEqual(max(epochs), 1226)
        self.assertAlmostEqual(statistics.mean(residuals), 0.0, places=10)
        self.assertAlmostEqual(statistics.stdev(residuals), 95.5548923824639)
        self.assertAlmostEqual(min(errors), 85.44855678354905)
        self.assertAlmostEqual(max(errors), 154.19079319858412)

    def test_tracked_tess_fits_files_match_baseline_sizes(self):
        tess_dir = (
            DATA_DIR
            / "mastDownload"
            / "TESS"
            / "tess2018263035959-s0003-0000000012862099-0123-s"
        )
        expected_sizes = {
            "tess2018263035959-s0003-0000000012862099-0123-s_lc.fits": 1998720,
            "tess2018263124740-s0003-s0003-0000000012862099-00405_dvt.fits": 3968640,
            "tess2018267104341-s0003-s0003-0000000012862099-00126_dvt.fits": 2949120,
        }

        actual_sizes = {
            path.name: path.stat().st_size
            for path in tess_dir.iterdir()
            if path.suffix == ".fits"
        }

        self.assertEqual(actual_sizes, expected_sizes)
