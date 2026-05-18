# ruff: noqa: E402

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "cmat-test-mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from matplotlib import pyplot as plt

from cmat.plotting.megno import plot_megno_surface
from cmat.ttv_sim import TTVSimulation


class PlottingMegnoTests(unittest.TestCase):
    def test_helper_plots_megno_surface_without_simulation_instance(self):
        fig, ax = plot_megno_surface(
            period_ratios=np.array([1.5, 2.0]),
            companion_masses=np.array([10.0, 20.0]),
            megno_results=np.array([2.0, 2.1, 9.8, 10.0]),
        )

        self.assertEqual(ax.get_xlabel(), r"$P_2/P_1$")
        self.assertEqual(ax.get_ylabel(), r"$M_2$ [$M_\oplus$]")
        self.assertGreaterEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_legacy_plot_megno_smoke(self):
        prop = [
            {
                "orbital_distance": 1.0,
                "orbital_period": 1.0,
                "Mp": 1.0,
                "Ms": 1.0,
            }
        ]
        sim = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.5, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.megno_results = [2.0, 2.1, 9.8, 10.0]

        fig, ax = sim.plot_megno()

        self.assertEqual(ax.get_xlabel(), r"$P_2/P_1$")
        self.assertEqual(ax.get_ylabel(), r"$M_2$ [$M_\oplus$]")
        plt.close(fig)
