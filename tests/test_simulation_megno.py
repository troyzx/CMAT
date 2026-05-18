import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cmat.simulation.megno import calculate_megno
from cmat.ttv_sim import TTVSimulation


class SimulationMegnoTests(unittest.TestCase):
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

    def test_helper_matches_legacy_facade(self):
        sim = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([0.0, 0.0, 0.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.5]),
            mp2s=np.array([10.0]),
            prop=self.prop,
        )

        legacy = sim.simulation_m((1.5, 10.0))
        helper = calculate_megno(
            parameters=(1.5, 10.0),
            prop=self.prop,
            dt=sim.megno_dt,
            runtime=sim.megno_runtime,
        )

        self.assertLess(abs(legacy - helper), 0.02)

    def test_run_megno_supports_legacy_thread_override(self):
        sim = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=self.prop,
        )
        sim.worker_count = 8
        sim.start_method = "spawn"
        sim.show_progress = False

        context_calls = []

        class FakePool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, parameters):
                return iter([2.1 for _ in parameters])

        class FakeContext:
            def Pool(self, process_count):
                context_calls.append(process_count)
                return FakePool()

        with mock.patch("cmat.ttv_sim.get_context", return_value=FakeContext()):
            result = sim.run_megno(number_of_threads=4)

        self.assertEqual(context_calls, [4])
        self.assertEqual(result, [2.1])

    def test_run_megno_uses_configured_execution_controls(self):
        sim = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.0]),
            mp2s=np.array([10.0]),
            prop=self.prop,
        )
        sim.worker_count = 3
        sim.start_method = "forkserver"
        sim.show_progress = False

        context_calls = []

        class FakePool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, parameters):
                return iter([2.1 for _ in parameters])

        class FakeContext:
            def Pool(self, process_count):
                context_calls.append(process_count)
                return FakePool()

        with mock.patch("cmat.ttv_sim.get_context", return_value=FakeContext()) as get_context_mock, mock.patch(
            "cmat.ttv_sim.tqdm"
        ) as tqdm_mock:
            result = sim.run_megno()

        get_context_mock.assert_called_once_with("forkserver")
        tqdm_mock.assert_not_called()
        self.assertEqual(context_calls, [3])
        self.assertEqual(result, [2.1])

    def test_megno_cache_round_trip_is_unchanged(self):
        sim = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.5, 2.0]),
            mp2s=np.array([10.0, 20.0]),
            prop=self.prop,
        )
        sim.megno_results = [2.0, 2.1, 9.8, 10.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "megno_grid.npz"
            sim.save_megno_grid_cache(cache_path)
            payload = sim.load_megno_grid_cache(cache_path)

        np.testing.assert_array_equal(payload["period_ratios"], np.array([1.5, 2.0]))
        np.testing.assert_array_equal(payload["companion_masses"], np.array([10.0, 20.0]))
        np.testing.assert_array_equal(payload["megno_results"], np.array([2.0, 2.1, 9.8, 10.0]))

    def test_run_megno_can_reuse_cache_without_pool(self):
        sim = TTVSimulation(
            epochs=np.array([0, 1, 2]),
            ttv_mcmc=np.array([1.0, 1.0, 1.0]),
            ttv_err=np.ones(3),
            rs=np.array([1.5]),
            mp2s=np.array([10.0]),
            prop=self.prop,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "megno_grid.npz"
            sim.megno_results = [2.5]
            sim.save_megno_grid_cache(cache_path)

            with mock.patch("cmat.ttv_sim.get_context") as get_context_mock:
                result = sim.run_megno(use_cache=True, cache_path=cache_path)

        get_context_mock.assert_not_called()
        self.assertEqual(result, [2.5])
