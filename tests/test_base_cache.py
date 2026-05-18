import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Patch llvmlite before importing cmat.base to avoid RuntimeError
# on environments where llvmlite.binding.initialize() is deprecated.
try:
    import llvmlite.binding
    llvmlite.binding.initialize = lambda *a, **kw: None
except ImportError:
    pass

# Mock out scipy.signal to avoid import errors from arviz/pytransit
# on environments with incompatible scipy versions.
sys.modules.setdefault("scipy.signal", MagicMock())

from cmat.base import Fitlpf


_MOCK_PROP = [{
    "transit_time": 1000.0,
    "transit_time_lower": 0.01,
    "transit_time_upper": 0.02,
    "orbital_period": 10.0,
    "orbital_period_lower": 0.001,
    "orbital_period_upper": 0.002,
    "Ms": 1.0,
    "Ms_unit": "M_sun",
    "Mp": 10.0,
    "Mp_unit": "M_earth",
    "orbital_period_unit": "days",
    "transit_time_unit": "BJD",
    "Mp_ref": "Test Ref",
}]


class GetParameterCacheTests(unittest.TestCase):
    """Tests for Fitlpf.get_parameter cache logic."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self._tmpdir, "planet_params.json")

    @patch("cmat.base.get_prop", return_value=_MOCK_PROP)
    @patch("cmat.base.get_id", return_value=12345)
    def test_first_run_saves_cache(self, mock_get_id, mock_get_prop):
        planet = Fitlpf("TestPlanet")
        planet.get_parameter(use_cache=True, cache_path=self.cache_file)

        self.assertEqual(mock_get_id.call_count, 1)
        self.assertEqual(mock_get_prop.call_count, 1)
        self.assertTrue(os.path.exists(self.cache_file))

        with open(self.cache_file, "r") as f:
            data = json.load(f)
        self.assertEqual(data["ticid"], 12345)

    @patch("cmat.base.get_prop", return_value=_MOCK_PROP)
    @patch("cmat.base.get_id", return_value=12345)
    def test_second_run_loads_from_cache(self, mock_get_id, mock_get_prop):
        # First run to populate the cache
        planet = Fitlpf("TestPlanet")
        planet.get_parameter(use_cache=True, cache_path=self.cache_file)

        # Reset mock counts
        mock_get_id.reset_mock()
        mock_get_prop.reset_mock()

        # Second run should read from cache, NOT call the API
        planet2 = Fitlpf("TestPlanet")
        planet2.get_parameter(use_cache=True, cache_path=self.cache_file)

        mock_get_id.assert_not_called()
        mock_get_prop.assert_not_called()
        self.assertEqual(planet2.ticid, 12345)
        self.assertAlmostEqual(planet2.period.n, 10.0)
        self.assertAlmostEqual(planet2.period.s, 0.002)
        self.assertAlmostEqual(planet2.zero_epoch.n, 1000.0 + 2.4e6 + 0.5)
        self.assertAlmostEqual(planet2.zero_epoch.s, 0.02)


class DownloadDataCacheTests(unittest.TestCase):
    """Tests for Fitlpf.download_data cache skip logic."""

    @patch("cmat.base.Observations")
    def test_skips_download_when_fits_exist(self, mock_observations):
        with tempfile.TemporaryDirectory() as tmpdir:
            planet = Fitlpf("TestPlanet", datadir=tmpdir + "/")

            # Create fake .fits to simulate already-downloaded data
            tess_dir = Path(tmpdir) / "TestPlanet" / "mastDownload" / "TESS" / "sector"
            tess_dir.mkdir(parents=True)
            (tess_dir / "lightcurve.fits").touch()

            result = planet.download_data(use_cache=True)

            self.assertIsNone(result)
            mock_observations.query_object.assert_not_called()

    @patch("cmat.base.Observations")
    def test_overwrite_cache_forces_download(self, mock_observations):
        with tempfile.TemporaryDirectory() as tmpdir:
            planet = Fitlpf("TestPlanet", datadir=tmpdir + "/")

            tess_dir = Path(tmpdir) / "TestPlanet" / "mastDownload" / "TESS" / "sector"
            tess_dir.mkdir(parents=True)
            (tess_dir / "lightcurve.fits").touch()

            planet.download_data(use_cache=True, overwrite_cache=True)

            # With overwrite_cache=True, the API should be called
            self.assertEqual(mock_observations.query_object.call_count, 1)


if __name__ == "__main__":
    unittest.main()
