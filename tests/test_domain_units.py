import subprocess
import sys
import unittest

import numpy as np

from cmat.domain.units import (
    ME_TO_MS,
    MJ_TO_MS,
    RS_TO_AU,
    earth_mass_to_solar_mass,
    jupiter_mass_to_solar_mass,
    solar_radius_to_au,
)


class DomainUnitsTests(unittest.TestCase):
    def test_constants_match_domain_contract(self):
        self.assertEqual(MJ_TO_MS, 9.5e-4)
        self.assertEqual(ME_TO_MS, 3.0e-6)
        self.assertEqual(RS_TO_AU, 0.00465047)

    def test_unit_conversions_support_scalars_and_arrays(self):
        self.assertAlmostEqual(float(jupiter_mass_to_solar_mass(2.0)), 1.9e-3)
        self.assertAlmostEqual(float(earth_mass_to_solar_mass(3.0)), 9.0e-6)
        self.assertAlmostEqual(float(solar_radius_to_au(2.0)), 0.00930094)
        np.testing.assert_allclose(
            earth_mass_to_solar_mass(np.array([1.0, 2.0])),
            np.array([3.0e-6, 6.0e-6]),
        )

    def test_domain_modules_import_without_heavy_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import importlib, sys; "
                    "mods=['cmat.domain.units','cmat.domain.timing',"
                    "'cmat.domain.residuals','cmat.domain.mass_limits']; "
                    "[importlib.import_module(name) for name in mods]; "
                    "heavy=['pytransit','rebound','matplotlib','astroquery']; "
                    "assert not any(name in sys.modules for name in heavy)"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
