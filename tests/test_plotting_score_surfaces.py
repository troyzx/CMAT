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

from cmat.plotting.score_surfaces import plot_score_surface
from cmat.ttv_sim import TTVSimulation


class PlottingScoreSurfaceTests(unittest.TestCase):
    def test_helper_plots_surface_without_simulation_instance(self):
        fig, ax = plot_score_surface(
            period_ratios=np.array([1.0, 2.0]),
            companion_masses=np.array([10.0, 20.0]),
            surface=np.array([[1.0, 2.0], [3.0, 4.0]]),
            statistic_label=r"$\chi^2$",
            threshold=2.5,
            levels=4,
        )

        self.assertEqual(ax.get_xlabel(), r"$P_2/P_1$")
        self.assertEqual(ax.get_ylabel(), r"Mass [$M_\oplus$]")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertGreaterEqual(len(fig.axes), 2)
        plt.close(fig)

    def test_legacy_plot_chi2_contour_still_delegates(self):
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
            rs=np.array([1.0, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=prop,
        )
        sim.ttv_results = [
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
            np.array([10.0, 10.0, 10.0, 10.0]),
        ]
        sim.get_m_crit()

        fig, ax = sim.plot_chi2_contour(statistic="reduced_chi2", levels=4)

        self.assertEqual(ax.get_xlabel(), r"$P_2/P_1$")
        self.assertIn("reduced", fig.axes[1].get_ylabel())
        plt.close(fig)
