import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from cmat.ttv_sim import ttv_sim


class TtvReboundTests(unittest.TestCase):
    def test_simulation_m_uses_orbital_distance_for_megno_setup(self):
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
        sim.megno_dt = 0.1
        sim.megno_runtime = 10.0

        created = []

        class FakeSimulation:
            def __init__(self):
                self.ri_whfast = type("WHFast", (), {"safe_mode": None})()
                self.particles = [type("Particle", (), {"P": 1.0})()]
                self.add_calls = []
                self.dt = None
                self.exit_max_distance = None
                created.append(self)

            def add(self, **kwargs):
                self.add_calls.append(kwargs)
                self.particles.append(
                    type("Particle", (), {"P": kwargs.get("a", 1.0)})()
                )

            def move_to_com(self):
                return None

            def init_megno(self):
                return None

            def integrate(self, runtime, exact_finish_time=0):
                self.integrated_runtime = runtime
                self.exact_finish_time = exact_finish_time

            def calculate_megno(self):
                return 2.0

        with patch("cmat.ttv_sim.rebound.Simulation", side_effect=FakeSimulation):
            megno = sim.simulation_m((1.5, 10.0))

        fake = created[0]

        self.assertEqual(megno, 2.0)
        self.assertAlmostEqual(fake.add_calls[1]["a"], prop[0]["orbital_distance"])
        self.assertAlmostEqual(
            fake.add_calls[2]["a"],
            prop[0]["orbital_distance"] * 1.5 ** (2 / 3),
        )
        self.assertAlmostEqual(
            fake.dt,
            sim.megno_dt * min(fake.particles[1].P, fake.particles[2].P),
        )

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
            atol=1e-9,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
