import importlib
import unittest


class PackageImportTests(unittest.TestCase):
    def test_package_import_does_not_eagerly_import_light_curve_stack(self):
        cmat = importlib.import_module("cmat")

        self.assertEqual(cmat.__all__, ["Fitlpf", "ttv_sim"])
        self.assertTrue(callable(cmat.ttv_sim))


if __name__ == "__main__":
    unittest.main()
