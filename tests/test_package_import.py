import importlib
import unittest


class PackageImportTests(unittest.TestCase):
    def test_package_import_does_not_eagerly_import_light_curve_stack(self):
        cmat = importlib.import_module("cmat")

        self.assertEqual(
            cmat.__all__,
            [
                "Fitlpf",
                "ttv_sim",
                "TargetConfig",
                "FitControls",
                "SimulationGrid",
                "OutputConfig",
                "RunConfig",
            ],
        )
        self.assertTrue(callable(cmat.ttv_sim))

    def test_top_level_config_export_is_available(self):
        cmat = importlib.import_module("cmat")

        target = cmat.TargetConfig("WASP-44 b")

        self.assertEqual(target.planet_name, "WASP-44 b")


if __name__ == "__main__":
    unittest.main()
