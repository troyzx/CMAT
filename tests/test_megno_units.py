"""Regressions for companion-mass units and the MEGNO orbit setup.

The N-body integration is mocked so these tests check the actual arguments
passed to REBOUND without relying on a lengthy stability integration.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class FakeSimulation:
    def __init__(self):
        self.ri_whfast = SimpleNamespace(safe_mode=1)
        self.particles = []
        self.integration_time = None

    def add(self, m, a=None, e=None):
        self.particles.append(SimpleNamespace(m=m, a=a, e=e, P=1.0))

    def move_to_com(self):
        pass

    def init_megno(self):
        pass

    def integrate(self, t, exact_finish_time=0):
        self.integration_time = t

    def calculate_megno(self):
        return 2.0


def load_ttv_module_with_fake_rebound():
    rebound = ModuleType("rebound")
    instances = []

    def make_simulation():
        simulation = FakeSimulation()
        instances.append(simulation)
        return simulation

    rebound.Simulation = make_simulation
    rebound.Escape = type("Escape", (Exception,), {})

    path = ROOT / "cmat" / "ttv_sim.py"
    spec = importlib.util.spec_from_file_location("ttv_megno_regression", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"rebound": rebound}):
        spec.loader.exec_module(module)
    return module, instances


class MegnoAndMassUnitTests(unittest.TestCase):
    def setUp(self):
        self.module, self.instances = load_ttv_module_with_fake_rebound()
        self.grid = self.module.ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.zeros(3),
            ttv_err=np.ones(3),
            rs=np.array([1.5, 2.0, 2.5]),
            mp2s=np.array([1.0, 10.0]),
            prop=[{
                "Ms": 0.893,
                "Mp": 1.7781,
                "orbital_distance": 0.02452,
                "orbital_period": 1.482246626,
            }],
        )
        self.addCleanup(plt.close, "all")

    def test_megno_uses_semimajor_axis_and_earth_mass_companion(self):
        self.assertEqual(self.grid.simulation_m((2.0, 10.0)), 2.0)
        self.assertEqual(len(self.instances), 1)
        particles = self.instances[0].particles
        self.assertAlmostEqual(particles[1].a, 0.02452)
        self.assertAlmostEqual(particles[2].a, 0.02452 * 2.0 ** (2 / 3))
        self.assertAlmostEqual(particles[1].m, 1.7781 * self.module.mj_to_ms)
        self.assertAlmostEqual(particles[2].m, 10.0 * self.module.me_to_ms)

    def test_megno_plot_uses_mass_rows_and_earth_mass_label(self):
        self.grid.megno_results = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        self.grid.plot_megno()
        ax = plt.gcf().axes[0]
        np.testing.assert_array_equal(
            ax.images[0].get_array(),
            np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]),
        )
        self.assertIn(r"\oplus", ax.get_ylabel())

    def test_example_notebook_uses_earth_mass_labels(self):
        notebook = json.loads((ROOT / "example.ipynb").read_text(encoding="utf-8"))
        plot_cells = [cell for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and
                      "[$M_{\\oplus}$]" in "".join(cell["source"])]
        self.assertEqual(len(plot_cells), 3)
        for cell in plot_cells:
            self.assertNotIn("[$M_j$]", "".join(cell["source"]))
            self.assertFalse(cell["outputs"])  # Old mislabeled images cleared.


if __name__ == "__main__":
    unittest.main()
