import pytest
import os
import sys
import pandas as pd
import xarray as xr
import numpy as np
import unittest.mock as mock # Import mock module explicitly
from unittest.mock import MagicMock, patch, call, ANY # Explicitly import ANY
import shutil

# Add the path to functions_module.py and main.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

# Import the run_analysis function from main.py
import main
from main import run_analysis, START_YEAR, END_YEAR, OUTPUT_BASE_DIR, INPUT_DIR, TARGET_HEIGHT, TARGET_LATITUDE, TARGET_LONGITUDE

# --- Fixtures for Mocking and Setup ---
# Fixture to create mock input data files
@pytest.fixture(scope="module")
def setup_mock_data_files():
    temp_input_dir = '/tmp/mock_input_data'
    os.makedirs(temp_input_dir, exist_ok=True)

    # Create dummy NetCDF files that WindDataLoader would find
    file_ranges_mapping = {
        (1997, 1999): "1997-1999.nc",
        (2000, 2002): "2000-2002.nc",
    }

    for (start_year, end_year), filename in file_ranges_mapping.items():
        filepath = os.path.join(temp_input_dir, filename)
        times = pd.to_datetime([f'{y}-01-01' for y in range(start_year, end_year + 1)])
        lat = np.array([55.75])
        lon = np.array([7.75])

        u10_data = np.random.rand(len(times), len(lat), len(lon)) * 10
        v10_data = np.random.rand(len(times), len(lat), len(lon)) * 10
        u100_data = np.random.rand(len(times), len(lat), len(lon)) * 15
        v100_data = np.random.rand(len(times), len(lat), len(lon)) * 15

        ds = xr.Dataset(
            {
                'u10': (('time', 'latitude', 'longitude'), u10_data),
                'v10': (('time', 'latitude', 'longitude'), v10_data),
                'u100': (('time', 'latitude', 'longitude'), u100_data),
                'v100': (('time', 'latitude', 'longitude'), v100_data),
            },
            coords={
                'time': times,
                'latitude': lat,
                'longitude': lon
            },
        )
        ds.to_netcdf(filepath)

    # Create dummy power curve CSVs for turbines
    nrel5mw_power_curve_path = os.path.join(temp_input_dir, 'NREL_Reference_5MW_126.csv')
    nrel15mw_power_curve_path = os.path.join(temp_input_dir, 'NREL_Reference_15MW_240.csv')

    data_5mw = {
        'wind_speed': [0, 3, 5, 10, 12, 25, 30],
        'power': [0, 0, 500, 3000, 5000, 5000, 0]
    }
    df_5mw = pd.DataFrame(data_5mw)
    df_5mw.to_csv(nrel5mw_power_curve_path, index=False)

    data_15mw = {
        'wind_speed': [0, 4, 6, 12, 15, 25, 30],
        'power': [0, 0, 1000, 8000, 15000, 15000, 0]
    }
    df_15mw = pd.DataFrame(data_15mw)
    df_15mw.to_csv(nrel15mw_power_curve_path, index=False)

    yield temp_input_dir # Provide the temp input directory to tests

    shutil.rmtree(temp_input_dir) # Cleanup after tests are done
# Fixture to create a mock output directory
@pytest.fixture(scope="function")
def mock_output_directory():
    temp_output_dir = 'Users/cinnamon/Downloads/Project03_46W38/tmp/mock_output_data'
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)
    yield temp_output_dir
    shutil.rmtree(temp_output_dir)
# --- Test Cases ---
@patch('main.WindDataLoader')
@patch('main.WindAnalysisPlotter')
@patch('main.NREL5MWWindTurbine')
@patch('main.NREL15MWWindTurbine')
@patch('main.WindResource')
@patch('pandas.DataFrame.to_csv') # Patch to_csv
@patch('os.path.exists') # Patch os.path.exists
# Define test run_analysis success case
def test_run_analysis_success(
    MockOsPathExists, # New mock argument
    MockToCsv,        # New mock argument
    MockWindResource,
    MockNREL15MWWindTurbine,
    MockNREL5MWWindTurbine,
    MockWindAnalysisPlotter,
    MockWindDataLoader,
    mock_output_directory,
    setup_mock_data_files
):
    # Set up mock instances and their return values
    mock_data_loader_instance = MockWindDataLoader.return_value
    mock_plotter_instance = MockWindAnalysisPlotter.return_value
    mock_nrel5mw_turbine_instance = MockNREL5MWWindTurbine.return_value
    mock_nrel15mw_turbine_instance = MockNREL15MWWindTurbine.return_value
    mock_wind_resource_instance = MockWindResource.return_value

    # Configure MockOsPathExists to return True for any path, simulating file creation
    MockOsPathExists.return_value = True

    # Configure WindDataLoader mock with enough data for Weibull fitting
    num_time_steps_for_weibull = 100 # Ensure enough data points for Weibull fitting
    dummy_dataset = xr.Dataset(
        {
            'u10': (('time', 'latitude', 'longitude'), np.random.rand(num_time_steps_for_weibull, 1, 1) * 10),
            'v10': (('time', 'latitude', 'longitude'), np.random.rand(num_time_steps_for_weibull, 1, 1) * 10)
        },
        coords={
            'time': pd.date_range(start='1997-01-01', periods=num_time_steps_for_weibull, freq='D'), # Use pd.date_range
            'latitude': [55.75],
            'longitude': [7.75]
        }
    )
    mock_data_loader_instance.load_data_for_year.return_value = dummy_dataset

    # Configure WindResource mock
    # Ensure the returned values are xarray.DataArray-like with .values attribute
    mock_wind_resource_instance.get_wind_data_at_height.return_value = (
        MagicMock(spec=xr.DataArray, values=np.array([10.0] * num_time_steps_for_weibull)),
        MagicMock(spec=xr.DataArray, values=np.array([270.0] * num_time_steps_for_weibull))
    )
    mock_wind_resource_instance.fit_weibull_distribution.return_value = (2.0, 10.0) # k, A

    # Configure Turbine mocks for AEP calculation
    mock_nrel5mw_turbine_instance.name = 'NREL 5 MW'
    mock_nrel5mw_turbine_instance.calculate_aep.return_value = 25000.0
    mock_nrel15mw_turbine_instance.name = 'NREL 15 MW'
    mock_nrel15mw_turbine_instance.calculate_aep.return_value = 75000.0

    # Patch global variables in main.py if necessary
    with (
        patch('main.INPUT_DIR', setup_mock_data_files),
        patch('main.OUTPUT_BASE_DIR', mock_output_directory),
        patch('main.START_YEAR', 1997),
        patch('main.END_YEAR', 2008),
        patch('main.TARGET_LATITUDE', 55.75),
        patch('main.TARGET_LONGITUDE', 7.75),
        patch('main.TARGET_HEIGHT', 90)
    ):

        run_analysis()

    # Assertions
    # Verify WindDataLoader and WindAnalysisPlotter were instantiated
    MockWindDataLoader.assert_called_once_with(input_dir=setup_mock_data_files)
    MockWindAnalysisPlotter.assert_called_once_with(output_dir=mock_output_directory)

    # Verify yearly directories were created
    mock_plotter_instance.create_yearly_directories.assert_called_once_with(1997, 2008)

    # Assert data_loader.load_data_for_year is called for each year
    expected_calls_load_data = [call(year) for year in range(1997, 2008 + 1)]
    mock_data_loader_instance.load_data_for_year.assert_has_calls(expected_calls_load_data, any_order=True)
    assert mock_data_loader_instance.load_data_for_year.call_count == (2008 - 1997 + 1)

    # Assert WindResource is instantiated for each year
    expected_calls_wind_resource = [call(mock_data_loader_instance.load_data_for_year.return_value, TARGET_LATITUDE, TARGET_LONGITUDE)
                                    for _ in range(1997, 2008 + 1)]
    MockWindResource.assert_has_calls(expected_calls_wind_resource, any_order=True)
    assert MockWindResource.call_count == (2008 - 1997 + 1)

    # Assert get_wind_data_at_height is called for each year
    expected_calls_get_wind_data = [call(TARGET_HEIGHT) for _ in range(1997, 2008 + 1)]
    mock_wind_resource_instance.get_wind_data_at_height.assert_has_calls(expected_calls_get_wind_data, any_order=True)
    assert mock_wind_resource_instance.get_wind_data_at_height.call_count == (2008 - 1997 + 1)

    # Assert fit_weibull_distribution is called for each year
    expected_calls_fit_weibull = [call(TARGET_HEIGHT) for _ in range(1997, 2008 + 1)]
    mock_wind_resource_instance.fit_weibull_distribution.assert_has_calls(expected_calls_fit_weibull, any_order=True)
    assert mock_wind_resource_instance.fit_weibull_distribution.call_count == (2008 - 1997 + 1)

    # Verify plotting calls are made
    expected_calls_plot_speed = [call(ANY, 2.0, 10.0, year, TARGET_HEIGHT) for year in range(1997, 2008 + 1)]
    mock_plotter_instance.plot_speed_distribution.assert_has_calls(expected_calls_plot_speed, any_order=True)
    assert mock_plotter_instance.plot_speed_distribution.call_count == (2008 - 1997 + 1)
    
    # Verify wind rose plotting calls
    expected_calls_plot_rose = [call(ANY, ANY, year, TARGET_HEIGHT) for year in range(1997, 2008 + 1)]
    mock_plotter_instance.plot_wind_rose.assert_has_calls(expected_calls_plot_rose, any_order=True)
    assert mock_plotter_instance.plot_wind_rose.call_count == (2008 - 1997 + 1)

    # Assert calculate_aep is called twice for each year (once for each turbine)
    expected_calls_aep_5mw = [call(2.0, 10.0) for _ in range(1997, 2008 + 1)]
    mock_nrel5mw_turbine_instance.calculate_aep.assert_has_calls(expected_calls_aep_5mw, any_order=True)
    assert mock_nrel5mw_turbine_instance.calculate_aep.call_count == (2008 - 1997 + 1)
    
    # Verify for 15 MW turbine
    expected_calls_aep_15mw = [call(2.0, 10.0) for _ in range(1997, 2008 + 1)]
    mock_nrel15mw_turbine_instance.calculate_aep.assert_has_calls(expected_calls_aep_15mw, any_order=True)
    assert mock_nrel15mw_turbine_instance.calculate_aep.call_count == (2008 - 1997 + 1)

    # Verify AEP summary file creation by checking mock_to_csv call
    aep_summary_path = os.path.join(mock_output_directory, f'aep_summary_{1997}-{2008}.csv')
    MockToCsv.assert_called_once_with(aep_summary_path, index=False)

# Define test run_analysis data loading failure case
@patch('main.WindDataLoader')
@patch('main.WindAnalysisPlotter')
@patch('main.NREL5MWWindTurbine')
@patch('main.NREL15MWWindTurbine')
@patch('main.WindResource')
@patch('pandas.DataFrame.to_csv')
# Removed patch('os.path.exists') as it's not needed for checking mock calls
def test_run_analysis_data_loading_failure(
    MockToCsv,
    MockWindResource,
    MockNREL15MWWindTurbine,
    MockNREL5MWWindTurbine,
    MockWindAnalysisPlotter,
    MockWindDataLoader,
    mock_output_directory,
    setup_mock_data_files
):
    mock_data_loader_instance = MockWindDataLoader.return_value
    mock_plotter_instance = MockWindAnalysisPlotter.return_value

    # Configure MockOsPathExists to return True for any path, simulating directory creation
    # This mock is actually unused in this test as os.path.exists is not called directly in run_analysis for output dir
    # creation, but rather by plotter.create_yearly_directories which we are mocking.
    # So removing MockOsPathExists from this test function's arguments.

    # Configure mock_data_loader_instance to return None for year 1998
    def side_effect_load_data(year):
        if year == 1998:
            return None
        else:
            # Ensure enough data points for Weibull fitting in successful years
            num_time_steps_for_weibull = 100
            dummy_dataset = xr.Dataset(
                {
                    'u10': (('time', 'latitude', 'longitude'), np.random.rand(num_time_steps_for_weibull, 1, 1) * 10),
                    'v10': (('time', 'latitude', 'longitude'), np.random.rand(num_time_steps_for_weibull, 1, 1) * 10)
                },
                coords={
                    'time': pd.date_range(start=f'{year}-01-01', periods=num_time_steps_for_weibull, freq='D'), # Use pd.date_range
                    'latitude': [55.75],
                    'longitude': [7.75]
                }
            )
            return dummy_dataset
    # Set the side effect for load_data_for_year
    mock_data_loader_instance.load_data_for_year.side_effect = side_effect_load_data

    # Configure WindResource mock (needed for years that don't fail)
    mock_wind_resource_instance = MockWindResource.return_value
    mock_wind_resource_instance.get_wind_data_at_height.return_value = (
        MagicMock(spec=xr.DataArray, values=np.array([10.0, 10.0])),
        MagicMock(spec=xr.DataArray, values=np.array([270.0, 270.0]))
    )
    # Configure fit_weibull_distribution mock
    mock_wind_resource_instance.fit_weibull_distribution.return_value = (2.0, 10.0)

    # Configure Turbine mocks (needed for years that don't fail)
    mock_nrel5mw_turbine_instance = MockNREL5MWWindTurbine.return_value
    mock_nrel5mw_turbine_instance.name = 'NREL 5 MW'
    mock_nrel5mw_turbine_instance.calculate_aep.return_value = 25000.0
    mock_nrel15mw_turbine_instance = MockNREL15MWWindTurbine.return_value
    mock_nrel15mw_turbine_instance.name = 'NREL 15 MW'
    mock_nrel15mw_turbine_instance.calculate_aep.return_value = 75000.0


    # Patch global variables in main.py if necessary
    with (
        patch('main.INPUT_DIR', setup_mock_data_files),
        patch('main.OUTPUT_BASE_DIR', mock_output_directory),
        patch('main.START_YEAR', 1997),
        patch('main.END_YEAR', 1998),
        patch('main.TARGET_LATITUDE', 55.75),
        patch('main.TARGET_LONGITUDE', 7.75),
        patch('main.TARGET_HEIGHT', 90)
    ):

        run_analysis()

    # Verify load_data_for_year was called for all years (1997, 1998)
    expected_calls_load_data = [call(1997), call(1998)]
    mock_data_loader_instance.load_data_for_year.assert_has_calls(expected_calls_load_data, any_order=True)
    assert mock_data_loader_instance.load_data_for_year.call_count == 2

    # Verify WindResource was NOT instantiated for the failed year (1998)
    assert MockWindResource.call_count == 1 # Only for 1997

    # Verify plotting and AEP calculation were NOT called for the failed year (1998)
    assert mock_plotter_instance.plot_speed_distribution.call_count == 1 # Only for 1997
    assert mock_plotter_instance.plot_wind_rose.call_count == 1 # Only for 1997
    assert mock_nrel5mw_turbine_instance.calculate_aep.call_count == 1 # Only for 1997
    assert mock_nrel15mw_turbine_instance.calculate_aep.call_count == 1 # Only for 1997

    # Check that aep_summary.csv was created with only one year's data
    aep_summary_path = os.path.join(mock_output_directory, 'aep_summary_1997-2008.csv') # Correct hardcoded filename
    MockToCsv.assert_called_once_with(aep_summary_path, index=False)

    # Verify no yearly output folders/files were created for 1998 if it failed
    # The plotter should not create files for 1998 as data loading failed
    assert mock_plotter_instance.plot_speed_distribution.call_count == 1
    assert mock_plotter_instance.plot_wind_rose.call_count == 1
