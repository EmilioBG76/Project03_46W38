import pytest
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import shutil # For cleaning up temporary directories

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')
import functions_module
from functions_module import WindDataLoader
# Test loading data for a specific year
@pytest.fixture(scope="module")
# Define a fixture to set up temporary netCDF files for testing
def setup_dummy_netcdf_files():
    temp_dir = '/Users/cinnamon/Downloads/Project03_46W38/tmp/test_winddataloader_years'
    os.makedirs(temp_dir, exist_ok=True)

    file_ranges_mapping = {
        (1997, 1999): "1997-1999.nc",
        (2000, 2002): "2000-2002.nc",
        (2003, 2005): "2003-2005.nc",
        (2006, 2008): "2006-2008.nc"
    }

    for (start_year, end_year), filename in file_ranges_mapping.items():
        filepath = os.path.join(temp_dir, filename)
        times = pd.to_datetime([f'{y}-01-01' for y in range(start_year, end_year + 1)])
        lat = np.array([55.75])
        lon = np.array([7.75])

        # Create simple data for u10 and v10
        u10_data = np.random.rand(len(times), len(lat), len(lon)) * 10
        v10_data = np.random.rand(len(times), len(lat), len(lon)) * 10

        ds = xr.Dataset(
            {
                'u10': (('time', 'latitude', 'longitude'), u10_data),
                'v10': (('time', 'latitude', 'longitude'), v10_data)
            },
            coords={
                'time': times,
                'latitude': lat,
                'longitude': lon
            }
        )
        ds.to_netcdf(filepath)

    yield temp_dir

    # Cleanup: remove the temporary directory and its contents
    shutil.rmtree(temp_dir)
# Define the test function for loading data for a single year
def test_load_data_for_single_year(setup_dummy_netcdf_files):
    temp_dir = setup_dummy_netcdf_files
    data_loader = WindDataLoader(input_dir=temp_dir)

    target_year = 1998
    loaded_ds = data_loader.load_data_for_year(target_year)

    assert loaded_ds is not None
    assert isinstance(loaded_ds, xr.Dataset)
    assert 'u10' in loaded_ds.data_vars
    assert 'v10' in loaded_ds.data_vars
    # Check that only data for the target year is loaded
    assert len(loaded_ds['time']) == 1 # One time step per year in dummy files
    assert pd.to_datetime(str(loaded_ds['time'].dt.year.item())) == pd.to_datetime(str(target_year))
# Define the test function for loading data for a year with no matching files
def test_load_data_for_year_no_matching_file(setup_dummy_netcdf_files):
    temp_dir = setup_dummy_netcdf_files
    data_loader = WindDataLoader(input_dir=temp_dir)

    target_year = 2010 # A year not covered by any dummy file
    loaded_ds = data_loader.load_data_for_year(target_year)

    assert loaded_ds is None
# Define the test function for loading data when input directory does not exist
def test_load_data_for_year_non_existent_input_dir():
    non_existent_dir = '/Users/cinnamon/Downloads/Project03_46W38/non/existent/input/dir_123'
    data_loader = WindDataLoader(input_dir=non_existent_dir)

    target_year = 1998
    loaded_ds = data_loader.load_data_for_year(target_year)

    assert loaded_ds is None
