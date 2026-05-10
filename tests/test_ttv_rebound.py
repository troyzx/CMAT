import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from cmat.ttv_sim import ttv_sim


class TtvReboundTests(unittest.TestCase):
    def test_calculate_rebound_matches_reduced_deterministic_fixture(self):
        prop = [
            {
                "orbital_distance": 0.055,
                "orbital_period": 4.0,
                "Mp": 1.0,
                "Ms": 1.0,
                "Rs": 1.0,
                "Rp": 1.0,
            }
        ]
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([0.0, 0.0, 0.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.5]),
            mp2s=np.array([10.0]),
            prop=prop,
        )

        ttv_rebound = sim.calculate_rebound((1.5, 10.0))

        np.testing.assert_allclose(
            ttv_rebound,
            np.array(
                [
                    -28.325572085764,
                    23.6329668558,
                    37.710782544297,
                    -33.018177315309,
                ]
            ),
            atol=1e-10,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
