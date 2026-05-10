import csv
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from cmat.ttv_sim import ttv_sim


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "WASP-44 b"


class NotebookSmokeTests(unittest.TestCase):
    def test_reduced_cached_wasp44_path_runs_notebook_forward_slice(self):
        with (DATA_DIR / "tc_data.csv").open(newline="", encoding="utf-8") as handle:
            ttv_rows = list(csv.DictReader(handle))
        with (DATA_DIR / "prop_data.csv").open(newline="", encoding="utf-8") as handle:
            prop_row = next(csv.DictReader(handle))

        epochs = np.array([int(row["epochs"]) for row in ttv_rows])
        ttv_mcmc = np.array([float(row["ttv_mcmc"]) for row in ttv_rows])
        ttv_err = np.array([float(row["ttv_err"]) for row in ttv_rows])
        prop = [
            {
                key: float(prop_row[key])
                for key in ("orbital_distance", "orbital_period", "Mp", "Ms", "Rs", "Rp")
            }
        ]

        simulation = ttv_sim(
            epochs=epochs,
            ttv_mcmc=ttv_mcmc,
            ttv_err=ttv_err,
            rs=np.array([1.5]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )

        simulation.ttv_results = [
            simulation.calculate_rebound((1.5, 10.0)),
            simulation.calculate_rebound((1.5, 20.0)),
        ]
        chi2_limit, rms_limit = simulation.get_m_crit()

        simulation.megno_dt = 0.02
        simulation.megno_runtime = 50.0
        megno = simulation.simulation_m((1.5, 10.0))

        self.assertEqual(len(simulation.ttv_results), 2)
        self.assertTrue(
            all(result.ndim == 1 and result.size >= len(epochs) for result in simulation.ttv_results)
        )
        np.testing.assert_array_equal(chi2_limit, np.array([]))
        np.testing.assert_array_equal(rms_limit, np.array([]))
        self.assertTrue(np.isfinite(megno))
        self.assertGreater(megno, 0.0)


if __name__ == "__main__":
    unittest.main()
