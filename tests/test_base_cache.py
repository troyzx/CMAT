"""Unit tests for Fitlpf caching methods (get_parameter, download_data)."""

import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from cmat.base import Fitlpf
    _HAS_BASE = True
except Exception:
    _HAS_BASE = False

_SKIP_REASON = "cmat.base requires pytransit/numba stack unavailable in this environment"

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


@unittest.skipUnless(_HAS_BASE, _SKIP_REASON)
class GetParameterCacheTests(unittest.TestCase):
    """Tests for Fitlpf.get_parameter cache logic."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self._tmpdir, "planet_params.json")

    def test_import_path_uses_real_scipy_signal_module(self):
        scipy_signal = importlib.import_module("scipy.signal")

        self.assertTrue(hasattr(scipy_signal, "windows"))
        self.assertIs(Fitlpf, importlib.import_module("cmat.base").Fitlpf)

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
        self.assertEqual(data["cache_schema_version"], "1")
        self.assertEqual(data["planet_name"], "TestPlanet")
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

    @patch("cmat.base.get_prop", return_value=_MOCK_PROP)
    @patch("cmat.base.get_id", return_value=12345)
    def test_planet_name_mismatch_raises_clear_error(self, mock_get_id, mock_get_prop):
        planet = Fitlpf("TestPlanet")
        planet.get_parameter(use_cache=True, cache_path=self.cache_file)

        with open(self.cache_file, "r") as f:
            cached_data = json.load(f)
        cached_data["planet_name"] = "OtherPlanet"
        with open(self.cache_file, "w") as f:
            json.dump(cached_data, f, indent=2)

        with self.assertRaisesRegex(ValueError, "planet_name mismatch"):
            Fitlpf("TestPlanet").get_parameter(use_cache=True, cache_path=self.cache_file)


@unittest.skipUnless(_HAS_BASE, _SKIP_REASON)
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
