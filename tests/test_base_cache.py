import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Patch llvmlite before importing cmat.base to avoid RuntimeError
import llvmlite.binding
original_init = llvmlite.binding.initialize
def dummy_init(*args, **kwargs):
    pass
llvmlite.binding.initialize = dummy_init

# Mock out scipy.signal to avoid the gaussian import error as well
import sys
from unittest.mock import MagicMock
sys.modules["scipy.signal"] = MagicMock()



from cmat.base import Fitlpf



@pytest.fixture
def temp_cache_dir(tmp_path):
    cache_dir = tmp_path / "cmat_cache"
    cache_dir.mkdir()
    return cache_dir


@patch("cmat.base.get_id")
@patch("cmat.base.get_prop")
def test_get_parameter_cache_save_and_load(mock_get_prop, mock_get_id, temp_cache_dir):
    # Mock the API responses
    mock_get_id.return_value = 12345
    mock_get_prop.return_value = [{
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
        "Mp_ref": "Test Ref"
    }]

    planet = Fitlpf("TestPlanet")
    cache_file = temp_cache_dir / "planet_params.json"

    # First run: should call the API and save the cache
    planet.get_parameter(use_cache=True, cache_path=str(cache_file))
    
    assert mock_get_id.call_count == 1
    assert mock_get_prop.call_count == 1
    assert cache_file.exists()
    
    with open(cache_file, "r") as f:
        data = json.load(f)
        assert data["ticid"] == 12345

    # Second run: should load from cache, API should not be called again
    planet2 = Fitlpf("TestPlanet")
    planet2.get_parameter(use_cache=True, cache_path=str(cache_file))
    
    assert mock_get_id.call_count == 1  # Still 1
    assert mock_get_prop.call_count == 1  # Still 1
    assert planet2.ticid == 12345
    assert planet2.period.n == 10.0
    assert planet2.period.s == 0.002
    assert planet2.zero_epoch.n == 1000.0 + 2.4e6 + 0.5
    assert planet2.zero_epoch.s == 0.02


@patch("cmat.base.Observations")
def test_download_data_cache_skip(mock_observations, tmp_path):
    planet = Fitlpf("TestPlanet", datadir=str(tmp_path) + "/")
    
    # Create fake fits file to simulate already downloaded data
    tess_dir = tmp_path / "TestPlanet" / "mastDownload" / "TESS" / "some_folder"
    tess_dir.mkdir(parents=True)
    (tess_dir / "data.fits").touch()

    # Call with cache enabled
    planet.download_data(use_cache=True)
    
    # Should skip download completely
    mock_observations.query_object.assert_not_called()

    # Call with overwrite enabled
    planet.download_data(use_cache=True, overwrite_cache=True)
    
    # Should call the API
    assert mock_observations.query_object.call_count == 1
