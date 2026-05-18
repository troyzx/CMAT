"""Unit tests for Fitlpf caching methods (get_parameter, download_data)."""

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import dill
import numpy as np

try:
    from cmat.base import Fitlpf
    _HAS_BASE = True
    _BASE_IMPORT_ERROR = None
except Exception as exc:
    _BASE_IMPORT_ERROR = exc
    try:
        sys.modules.pop("cmat.base", None)
        sys.modules.pop("cmat.singlefit", None)

        pytransit_module = types.ModuleType("pytransit")
        pytransit_lpf_module = types.ModuleType("pytransit.lpf")
        pytransit_tesslpf_module = types.ModuleType("pytransit.lpf.tesslpf")
        pytransit_orbits_module = types.ModuleType("pytransit.orbits")

        class _StubBaseLPF:
            pass

        class _StubTESSLPF:
            pass

        def _stub_fold(*args, **kwargs):
            return np.zeros(1)

        def _stub_downsample_time(*args, **kwargs):
            return np.zeros(1), np.zeros(1), np.zeros(1)

        def _stub_epoch(tc, zero_epoch, period):
            return int(np.rint((tc - zero_epoch) / period))

        pytransit_tesslpf_module.BaseLPF = _StubBaseLPF
        pytransit_tesslpf_module.TESSLPF = _StubTESSLPF
        pytransit_tesslpf_module.fold = _stub_fold
        pytransit_tesslpf_module.downsample_time = _stub_downsample_time
        pytransit_orbits_module.epoch = _stub_epoch

        sys.modules["pytransit"] = pytransit_module
        sys.modules["pytransit.lpf"] = pytransit_lpf_module
        sys.modules["pytransit.lpf.tesslpf"] = pytransit_tesslpf_module
        sys.modules["pytransit.orbits"] = pytransit_orbits_module

        from cmat.base import Fitlpf

        _HAS_BASE = True
        _BASE_IMPORT_ERROR = None
    except Exception as fallback_exc:
        _HAS_BASE = False
        _BASE_IMPORT_ERROR = fallback_exc

_SKIP_REASON = (
    "cmat.base requires pytransit/numba stack unavailable in this environment"
    if _BASE_IMPORT_ERROR is None
    else f"cmat.base import is unavailable: {_BASE_IMPORT_ERROR}"
)

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


class _FakeSeries:
    def __init__(self, mean_value, std_value):
        self._mean_value = mean_value
        self._std_value = std_value

    def mean(self):
        return self._mean_value

    def std(self):
        return self._std_value


class _FakeSingle:
    def __init__(self, tc_mean, tc_std):
        self._posterior = {"tc": _FakeSeries(tc_mean, tc_std)}

    def posterior_samples(self):
        return self._posterior


class _DummyProgress:
    def update(self, _count):
        return None


class _DummyTqdm:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return _DummyProgress()

    def __exit__(self, exc_type, exc, tb):
        return False


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


@unittest.skipUnless(_HAS_BASE, _SKIP_REASON)
class FitSinglesCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self.cache_file = os.path.join(self._tmpdir, "fit_singles.dill")

    def _make_fit(self, planet_name="TestPlanet"):
        fit = Fitlpf(planet_name)
        fit.zero_epoch = importlib.import_module("cmat.base").ufloat(100.0, 1e-3)
        fit.period = importlib.import_module("cmat.base").ufloat(2.0, 1e-4)
        fit.lpf = SimpleNamespace(times=[object(), object(), object()])
        return fit

    @staticmethod
    def _vectorize_factory(func):
        def _runner(indices, pbar):
            return np.array([func(index, pbar) for index in indices], dtype=object)

        return _runner

    def _fit_single_side_effect(self, index, _pbar):
        return _FakeSingle(110.0 + 2.0 * index, 0.01 + 0.01 * index)

    def test_valid_v2_cache_restores_results_without_refitting(self):
        fit = self._make_fit()
        payload = {
            "cache_schema_version": "2",
            "planet_name": "TestPlanet",
            "post_samples": [{"tc": _FakeSeries(110.0, 0.01)}],
            "tcs": [importlib.import_module("cmat.base").ufloat(110.0, 0.01)],
            "epochs": np.array([5]),
        }

        with open(self.cache_file, "wb") as f:
            dill.dump(payload, f)

        with patch.object(fit, "fit_single", side_effect=AssertionError("should not refit")):
            fit.fit_singles(use_cache=True, cache_path=self.cache_file)

        self.assertIsNone(fit.singles)
        np.testing.assert_array_equal(fit.epochs, payload["epochs"])
        self.assertEqual(len(fit.post_samples), 1)
        self.assertAlmostEqual(fit.post_samples[0]["tc"].mean(), 110.0)
        self.assertAlmostEqual(fit.post_samples[0]["tc"].std(), 0.01)
        self.assertAlmostEqual(fit.tcs[0].n, 110.0)
        self.assertAlmostEqual(fit.tcs[0].s, 0.01)

    def test_old_object_cache_is_ignored_and_recomputed(self):
        fit = self._make_fit()
        old_payload = {
            "singles": ["unsafe-pytransit-object"],
            "post_samples": ["stale"],
            "tcs": ["stale"],
            "epochs": np.array([999]),
        }

        with open(self.cache_file, "wb") as f:
            dill.dump(old_payload, f)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("cmat.base.np.vectorize", side_effect=self._vectorize_factory):
                with patch("cmat.base.tqdm", _DummyTqdm):
                    with patch.object(fit, "fit_single", side_effect=self._fit_single_side_effect) as mock_fit_single:
                        fit.fit_singles(use_cache=True, cache_path=self.cache_file)

        self.assertEqual(mock_fit_single.call_count, 3)
        self.assertTrue(
            any("Ignoring fit_singles cache" in str(item.message) for item in caught)
        )
        np.testing.assert_array_equal(fit.epochs, np.array([5, 6, 7]))

    def test_corrupted_dill_cache_is_ignored_and_recomputed(self):
        fit = self._make_fit()
        with open(self.cache_file, "wb") as f:
            f.write(b"not-a-dill-payload")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("cmat.base.np.vectorize", side_effect=self._vectorize_factory):
                with patch("cmat.base.tqdm", _DummyTqdm):
                    with patch.object(fit, "fit_single", side_effect=self._fit_single_side_effect) as mock_fit_single:
                        fit.fit_singles(use_cache=True, cache_path=self.cache_file)

        self.assertEqual(mock_fit_single.call_count, 3)
        self.assertTrue(
            any("Ignoring fit_singles cache" in str(item.message) for item in caught)
        )

    def test_planet_name_mismatch_is_rejected_and_recomputed(self):
        fit = self._make_fit()
        payload = {
            "cache_schema_version": "2",
            "planet_name": "OtherPlanet",
            "post_samples": [],
            "tcs": [],
            "epochs": np.array([], dtype=int),
        }

        with open(self.cache_file, "wb") as f:
            dill.dump(payload, f)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with patch("cmat.base.np.vectorize", side_effect=self._vectorize_factory):
                with patch("cmat.base.tqdm", _DummyTqdm):
                    with patch.object(fit, "fit_single", side_effect=self._fit_single_side_effect) as mock_fit_single:
                        fit.fit_singles(use_cache=True, cache_path=self.cache_file)

        self.assertEqual(mock_fit_single.call_count, 3)
        self.assertTrue(
            any("planet_name mismatch" in str(item.message) for item in caught)
        )

    def test_saved_cache_does_not_contain_singles(self):
        fit = self._make_fit()

        with patch("cmat.base.np.vectorize", side_effect=self._vectorize_factory):
            with patch("cmat.base.tqdm", _DummyTqdm):
                with patch.object(fit, "fit_single", side_effect=self._fit_single_side_effect):
                    fit.fit_singles(use_cache=True, cache_path=self.cache_file)

        with open(self.cache_file, "rb") as f:
            payload = dill.load(f)

        self.assertEqual(payload["cache_schema_version"], "2")
        self.assertEqual(payload["planet_name"], "TestPlanet")
        self.assertNotIn("singles", payload)

    def test_downstream_ttv_calculation_works_after_loading_cached_results(self):
        fit = self._make_fit()
        payload = {
            "cache_schema_version": "2",
            "planet_name": "TestPlanet",
            "post_samples": [
                {"tc": _FakeSeries(110.0, 0.01)},
                {"tc": _FakeSeries(112.0, 0.02)},
                {"tc": _FakeSeries(114.0, 0.03)},
            ],
            "tcs": [
                importlib.import_module("cmat.base").ufloat(110.0, 0.01),
                importlib.import_module("cmat.base").ufloat(112.0, 0.02),
                importlib.import_module("cmat.base").ufloat(114.0, 0.03),
            ],
            "epochs": np.array([5, 6, 7]),
        }

        with open(self.cache_file, "wb") as f:
            dill.dump(payload, f)

        with patch.object(fit, "fit_single", side_effect=AssertionError("should not refit")):
            fit.fit_singles(use_cache=True, cache_path=self.cache_file)

        fit.calculate_ttv()

        np.testing.assert_allclose(fit.ttv_mcmc, np.array([0.0, 0.0, 0.0]), atol=1e-8)
        np.testing.assert_allclose(
            fit.ttv_err,
            np.array([0.01, 0.0201, 0.0302]) * importlib.import_module("cmat.base").DAY_TO_SEC,
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
