import importlib
import subprocess
import sys
import unittest


class PackageImportTests(unittest.TestCase):
    def test_package_import_does_not_eagerly_import_light_curve_stack(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import cmat, sys; "
                    "assert cmat.__all__ == ["
                    "'Fitlpf','TransitFitWorkflow','ttv_sim','TTVSimulation',"
                    "'TargetConfig','FitControls','SimulationGrid','ExecutionConfig',"
                    "'BayesianScoringConfig','ScoringConfig','OutputConfig','RunConfig']; "
                    "assert callable(cmat.__getattr__); "
                    "heavy=('pytransit','rebound','matplotlib','astroquery','requests'); "
                    "assert not any(name in sys.modules for name in heavy)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)

    def test_top_level_config_export_is_available(self):
        cmat = importlib.import_module("cmat")

        target = cmat.TargetConfig("WASP-44 b")
        execution = cmat.ExecutionConfig()
        bayesian = cmat.BayesianScoringConfig()
        scoring = cmat.ScoringConfig()

        self.assertEqual(target.planet_name, "WASP-44 b")
        self.assertEqual(execution.start_method, "fork")
        self.assertEqual(
            bayesian.nuisance_parameters, ("epoch_shift", "baseline_offset", "jitter")
        )
        self.assertEqual(scoring.backend, "chi2_rms")

    def test_legacy_and_preferred_fit_exports_both_resolve(self):
        cmat = importlib.import_module("cmat")

        try:
            fitlpf = cmat.__getattr__("Fitlpf")
            workflow = cmat.__getattr__("TransitFitWorkflow")
        except Exception as exc:
            raise unittest.SkipTest(
                f"Fit workflow import is unavailable in this environment: {exc}"
            ) from exc

        self.assertTrue(callable(fitlpf))
        self.assertTrue(callable(workflow))
        self.assertIs(workflow, fitlpf)
        self.assertTrue(hasattr(workflow, "plot_ttv_residuals"))

    def test_preferred_simulation_name_remains_callable_after_submodule_import(self):
        cmat = importlib.import_module("cmat")

        importlib.import_module("cmat.ttv_sim")

        self.assertTrue(callable(cmat.TTVSimulation))


if __name__ == "__main__":
    unittest.main()
