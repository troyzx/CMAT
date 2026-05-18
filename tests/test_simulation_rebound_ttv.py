import importlib
import unittest
from unittest.mock import patch

import numpy as np

from cmat.simulation.rebound_ttv import calculate_rebound_ttv
from cmat.ttv_sim import ttv_sim


class SimulationReboundTtvTests(unittest.TestCase):
    def setUp(self):
        self.prop = [
            {
                "orbital_distance": 0.055,
                "orbital_period": 4.0,
                "Mp": 1.0,
                "Ms": 1.0,
                "Rs": 1.0,
                "Rp": 1.0,
            }
        ]

    def test_helper_matches_legacy_facade_for_reduced_fixture(self):
        sim = ttv_sim(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([0.0, 0.0, 0.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.5]),
            mp2s=np.array([10.0]),
            prop=self.prop,
        )

        legacy = sim.calculate_rebound((1.5, 10.0))
        helper = calculate_rebound_ttv(
            parameters=(1.5, 10.0),
            prop=self.prop,
            n_transit_simulations=(sim.epochs[-1] - sim.epochs[0]) * 2,
        )

        np.testing.assert_allclose(legacy, helper, atol=1e-12, rtol=0.0)

    def test_helper_preserves_mass_and_radius_unit_conversions(self):
        prop = [
            {
                "orbital_distance": 0.055,
                "orbital_period": 4.0,
                "Mp": 1.2,
                "Ms": 0.95,
                "Rs": 0.93,
                "Rp": 1.14,
            }
        ]
        ttv_module = importlib.import_module("cmat.ttv_sim")
        rebound_module = importlib.import_module("cmat.simulation.rebound_ttv")
        created = []

        class FakeSimulation:
            def __init__(self):
                self.ri_whfast = type("WHFast", (), {"safe_mode": None})()
                self.t = 0.0
                self.particles = [
                    type("Particle", (), {"x": 0.0, "y": 0.0, "P": 1.0})()
                ]
                self.add_calls = []
                created.append(self)

            def add(self, **kwargs):
                self.add_calls.append(kwargs)
                self.particles.append(
                    type(
                        "Particle",
                        (),
                        {"x": 0.0, "y": 0.0, "P": kwargs.get("a", 1.0)},
                    )()
                )

            def move_to_com(self):
                return None

            def integrate(self, *_args, **_kwargs):
                raise rebound_module.rebound.Escape

        with patch(
            "cmat.simulation.rebound_ttv.rebound.Simulation", side_effect=FakeSimulation
        ):
            ttv_rebound = calculate_rebound_ttv(
                parameters=(1.5, 10.0),
                prop=prop,
                n_transit_simulations=2,
            )

        fake = created[-1]

        self.assertEqual(ttv_rebound.shape, (2,))
        self.assertAlmostEqual(fake.add_calls[0]["r"], prop[0]["Rs"] * ttv_module.rs_to_AU)
        self.assertAlmostEqual(fake.add_calls[1]["m"], prop[0]["Mp"] * ttv_module.mj_to_ms)
        self.assertAlmostEqual(
            fake.add_calls[1]["r"],
            prop[0]["Rp"] * ttv_module.rj_to_rs * ttv_module.rs_to_AU,
        )
        self.assertAlmostEqual(fake.add_calls[2]["m"], 10.0 * ttv_module.me_to_ms)

    def test_helper_returns_invalid_series_for_early_termination(self):
        ttv_rebound = calculate_rebound_ttv(
            parameters=(1.5, 10.0),
            prop=self.prop,
            n_transit_simulations=4,
        )

        self.assertEqual(ttv_rebound.shape, (4,))
        self.assertTrue(np.all(np.isfinite(ttv_rebound)))

    def test_helper_matches_legacy_invalid_series_handling(self):
        sim = ttv_sim(
            epochs=np.array([0, 1]),
            ttv_mcmc=np.array([0.0, 0.0]),
            ttv_err=np.ones(2),
            rs=np.array([1.5]),
            mp2s=np.array([10.0]),
            prop=self.prop,
        )
        rebound_module = importlib.import_module("cmat.simulation.rebound_ttv")

        class FakeSimulation:
            def __init__(self):
                self.ri_whfast = type("WHFast", (), {"safe_mode": None})()
                self.t = 0.0
                self.particles = [
                    type("Particle", (), {"x": 0.0, "y": 1.0, "P": 1.0})()
                ]

            def add(self, **kwargs):
                self.particles.append(
                    type(
                        "Particle",
                        (),
                        {"x": 0.0, "y": 1.0, "P": kwargs.get("a", 1.0)},
                    )()
                )

            def move_to_com(self):
                return None

            def integrate(self, *_args, **_kwargs):
                raise rebound_module.rebound.Escape

        with patch(
            "cmat.simulation.rebound_ttv.rebound.Simulation", side_effect=FakeSimulation
        ):
            helper = calculate_rebound_ttv(
                parameters=(1.5, 10.0),
                prop=self.prop,
                n_transit_simulations=2,
            )
            legacy = sim.calculate_rebound((1.5, 10.0))

        self.assertTrue(np.all(np.isnan(helper)))
        np.testing.assert_array_equal(legacy, helper)
