import pytest
import os
import sys
import xarray as xr
import numpy as np
import pandas as pd

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')
import functions_module
from functions_module import WindDataLoader

@pytest.fixture(scope="module")
def setup_dummy_netcdf_file():
    # Create a temporary directory for the dummy file
    temp_dir = '/Users/cinnamon/Downloads/Project03_46W38/tmp/test_winddataloader'
    os.makedirs(temp_dir, exist_ok=True)
    dummy_filepath = os.path.join(temp_dir, 'dummy_data.nc')

    # Create a simple xarray Dataset
    ds = xr.Dataset(
        {
            'u10': (('time', 'lat', 'lon'), np.random.rand(1, 1, 1)),
            'v10': (('time', 'lat', 'lon'), np.random.rand(1, 1, 1))
        },
        coords={
            'time': pd.to_datetime(['2000-01-01']),
            'lat': [0],
            'lon': [0]
        }
    )
    ds.to_netcdf(dummy_filepath)
    yield dummy_filepath
    # Cleanup: remove the dummy file and directory
    os.remove(dummy_filepath)
    os.rmdir(temp_dir)

def test_load_netcdf_file_success(setup_dummy_netcdf_file):
    dummy_filepath = setup_dummy_netcdf_file
    data_loader = WindDataLoader()

    loaded_ds = data_loader._load_netcdf_file(dummy_filepath)

    # Assertions
    assert isinstance(loaded_ds, xr.Dataset)
    assert 'u10' in loaded_ds
    assert 'v10' in loaded_ds
    # Correctly assert that the dataset is not empty by checking if it has data variables
    assert len(loaded_ds.data_vars) > 0

def test_load_netcdf_file_not_found():
    data_loader = WindDataLoader()
    non_existent_filepath = '/Users/cinnamon/Downloads/Project03_46W38/non/existent/path/to/file.nc'

    with pytest.raises(FileNotFoundError) as excinfo:
        data_loader._load_netcdf_file(non_existent_filepath)
    assert f"File not found: {non_existent_filepath}" in str(excinfo.value)
