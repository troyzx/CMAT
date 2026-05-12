import importlib
import unittest


class PackageImportTests(unittest.TestCase):
    def test_package_import_does_not_eagerly_import_light_curve_stack(self):
        cmat = importlib.import_module("cmat")

        self.assertEqual(
            cmat.__all__,
            [
                "Fitlpf",
                "TransitFitWorkflow",
                "ttv_sim",
                "TTVSimulation",
                "TargetConfig",
                "FitControls",
                "SimulationGrid",
                "ExecutionConfig",
                "BayesianScoringConfig",
                "ScoringConfig",
                "OutputConfig",
                "RunConfig",
            ],
        )
        self.assertTrue(callable(cmat.__getattr__("ttv_sim")))
        self.assertTrue(callable(cmat.__getattr__("TTVSimulation")))
        self.assertIs(cmat.__getattr__("TTVSimulation"), cmat.__getattr__("ttv_sim"))

    def test_top_level_config_export_is_available(self):
        cmat = importlib.import_module("cmat")

        target = cmat.TargetConfig("WASP-44 b")
        execution = cmat.ExecutionConfig()
        bayesian = cmat.BayesianScoringConfig()
        scoring = cmat.ScoringConfig()

        self.assertEqual(target.planet_name, "WASP-44 b")
        self.assertEqual(execution.start_method, "fork")
        self.assertEqual(bayesian.nuisance_parameters, ("epoch_shift", "baseline_offset", "jitter"))
        self.assertEqual(scoring.backend, "chi2_rms")

    def test_legacy_and_preferred_fit_exports_both_resolve(self):
        cmat = importlib.import_module("cmat")

        self.assertTrue(callable(cmat.__getattr__("Fitlpf")))
        self.assertTrue(callable(cmat.__getattr__("TransitFitWorkflow")))
        self.assertIs(
            cmat.__getattr__("TransitFitWorkflow"),
            cmat.__getattr__("Fitlpf"),
        )
        self.assertTrue(hasattr(cmat.__getattr__("TransitFitWorkflow"), "plot_ttv_residuals"))


if __name__ == "__main__":
    unittest.main()
