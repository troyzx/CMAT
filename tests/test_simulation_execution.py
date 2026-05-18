import subprocess
import sys
import unittest

from cmat.simulation.execution import (
    build_mass_ratio_parameter_grid,
    maybe_run_in_pool,
    resolve_worker_count,
)


def _plus_one(value):
    return value + 1


class SimulationExecutionTests(unittest.TestCase):
    def test_build_mass_ratio_parameter_grid_preserves_legacy_order(self):
        self.assertEqual(
            build_mass_ratio_parameter_grid([1.5, 2.0], [10.0, 20.0]),
            [(1.5, 10.0), (2.0, 10.0), (1.5, 20.0), (2.0, 20.0)],
        )

    def test_resolve_worker_count_uses_configured_default(self):
        self.assertEqual(resolve_worker_count(None, worker_count=3), 3)
        self.assertEqual(resolve_worker_count(2, worker_count=3), 2)

    def test_resolve_worker_count_rejects_invalid_values(self):
        with self.assertRaises(TypeError):
            resolve_worker_count(2.5, worker_count=1)
        with self.assertRaises(ValueError):
            resolve_worker_count(0, worker_count=1)

    def test_maybe_run_in_pool_runs_serially_when_worker_count_is_one(self):
        result = maybe_run_in_pool(
            _plus_one,
            [1, 2, 3],
            worker_count=1,
            show_progress=False,
        )

        self.assertEqual(result, [2, 3, 4])

    def test_maybe_run_in_pool_uses_pool_when_requested(self):
        context_calls = []

        class FakePool:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def imap(self, func, parameters):
                return iter(func(item) for item in parameters)

        class FakeContext:
            def Pool(self, worker_count):
                context_calls.append(worker_count)
                return FakePool()

        result = maybe_run_in_pool(
            _plus_one,
            [1, 2, 3],
            worker_count=2,
            start_method="spawn",
            show_progress=False,
            get_context_fn=lambda _method: FakeContext(),
        )

        self.assertEqual(context_calls, [2])
        self.assertEqual(result, [2, 3, 4])

    def test_execution_module_import_is_rebound_and_matplotlib_safe(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import cmat, cmat.domain, cmat.simulation.execution, sys; "
                    "assert 'rebound' not in sys.modules; "
                    "assert 'matplotlib' not in sys.modules"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)
