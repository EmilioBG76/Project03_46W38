import pytest
import numpy as np
import xarray as xr
import sys
import os
import pandas as pd # Added import for pandas

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

from functions_module import WindResource, compute_wind_speed_direction
# Fixture to create a WindResource instance with dummy data
@pytest.fixture
# Define a fixture to set up WindResource with dummy data
def setup_wind_resource():
    # Dummy data for a single time point, 2x2 spatial grid
    time_val = pd.to_datetime('2000-01-01T00:00:00')
    latitudes = np.array([55.5, 56.0])
    longitudes = np.array([7.5, 8.0])

    # Example u and v components at 10m and 100m
    # At 55.75N, 7.75E (center point of the grid)
    # For linear interpolation, we'll need values from all 4 corners.
    # Let's define values such that the interpolated value for u and v is simple.
    # For the target 55.75N, 7.75E:
    # u10 will be interpolated between (55.5, 7.5), (55.5, 8.0), (56.0, 7.5), (56.0, 8.0)
    # Assuming 55.75 is 0.5 way between 55.5 and 56.0
    # Assuming 7.75 is 0.5 way between 7.5 and 8.0
    # So the interpolated value will be the average of the four corners.

    # u10 values (interpolates to (2+3+4+5)/4 = 3.5 at target lat/lon)
    u10_data = np.array([[[2.0, 3.0], [4.0, 5.0]]])
    # v10 values (interpolates to (6+7+8+9)/4 = 7.5 at target lat/lon)
    v10_data = np.array([[[6.0, 7.0], [8.0, 9.0]]])
    # u100 values (interpolates to (10+11+12+13)/4 = 11.5 at target lat/lon)
    u100_data = np.array([[[10.0, 11.0], [12.0, 13.0]]])
    # v100 values (interpolates to (14+15+16+17)/4 = 15.5 at target lat/lon)
    v100_data = np.array([[[14.0, 15.0], [16.0, 17.0]]])

    dummy_dataset = xr.Dataset(
        {
            'u10': (('time', 'latitude', 'longitude'), u10_data),
            'v10': (('time', 'latitude', 'longitude'), v10_data),
            'u100': (('time', 'latitude', 'longitude'), u100_data),
            'v100': (('time', 'latitude', 'longitude'), v100_data),
        },
        coords={
            'time': [time_val],
            'latitude': latitudes,
            'longitude': longitudes,
        },
    )

    target_latitude = 55.75
    target_longitude = 7.75
    return WindResource(dummy_dataset, target_latitude, target_longitude)
# Define test functions for get_wind_data_at_10m
def test_get_wind_data_at_10m(setup_wind_resource):
    wind_resource = setup_wind_resource
    wind_speed, wind_direction = wind_resource.get_wind_data_at_height(10)

    # Expected interpolated u10 and v10 at 55.75N, 7.75E (midpoint interpolation)
    expected_u10 = 3.5 # (2+3+4+5)/4
    expected_v10 = 7.5 # (6+7+8+9)/4
    expected_speed_10m, expected_dir_10m = compute_wind_speed_direction(expected_u10, expected_v10)

    np.testing.assert_almost_equal(wind_speed.item(), expected_speed_10m.item(), decimal=5)
    np.testing.assert_almost_equal(wind_direction.item(), expected_dir_10m.item(), decimal=5)
# Define test functions for get_wind_data_at_100m
def test_get_wind_data_at_100m(setup_wind_resource):
    wind_resource = setup_wind_resource
    wind_speed, wind_direction = wind_resource.get_wind_data_at_height(100)

    # Expected interpolated u100 and v100 at 55.75N, 7.75E
    expected_u100 = 11.5 # (10+11+12+13)/4
    expected_v100 = 15.5 # (14+15+16+17)/4
    expected_speed_100m, expected_dir_100m = compute_wind_speed_direction(expected_u100, expected_v100)

    np.testing.assert_almost_equal(wind_speed.item(), expected_speed_100m.item(), decimal=5)
    np.testing.assert_almost_equal(wind_direction.item(), expected_dir_100m.item(), decimal=5)
# Define test functions for get_wind_data_at intermediate height (e.g., 50m)
def test_get_wind_data_between_10m_100m(setup_wind_resource):
    wind_resource = setup_wind_resource
    target_height = 50 # Example: 50m
    wind_speed, wind_direction = wind_resource.get_wind_data_at_height(target_height)

    # Linear interpolation between 10m and 100m values
    # u_50 = u10 + (u100 - u10) * ((50 - 10) / (100 - 10))
    expected_u10 = 3.5
    expected_v10 = 7.5
    expected_u100 = 11.5
    expected_v100 = 15.5

    frac = (target_height - 10) / (100 - 10) # 40/90
    expected_u_50 = expected_u10 + (expected_u100 - expected_u10) * frac
    expected_v_50 = expected_v10 + (expected_v100 - expected_v10) * frac
    expected_speed_50m, expected_dir_50m = compute_wind_speed_direction(expected_u_50, expected_v_50)

    np.testing.assert_almost_equal(wind_speed.item(), expected_speed_50m.item(), decimal=5)
    np.testing.assert_almost_equal(wind_direction.item(), expected_dir_50m.item(), decimal=5)
# Define test functions for get_wind_data_at height above 100m (e.g., 120m)
def test_get_wind_data_above_100m_extrapolation(setup_wind_resource):
    wind_resource = setup_wind_resource
    target_height = 120 # Example: 120m
    wind_speed, wind_direction = wind_resource.get_wind_data_at_height(target_height)

    # Extrapolation uses power law from 100m data with alpha = 0.14
    alpha = 0.14
    expected_u100 = 11.5
    expected_v100 = 15.5
    ws_100, dir_100 = compute_wind_speed_direction(expected_u100, expected_v100)

    expected_speed_120m = ws_100 * (target_height / 100)**alpha
    expected_dir_120m = dir_100 # Direction assumed to be same as 100m

    np.testing.assert_almost_equal(wind_speed.item(), expected_speed_120m.item(), decimal=5)
    np.testing.assert_almost_equal(wind_direction.item(), expected_dir_120m.item(), decimal=5)
# Define test functions for get_wind_data_at zero wind speeds
def test_get_wind_data_zero_wind():
    # Test with zero wind speeds at all levels
    time_val = pd.to_datetime('2000-01-01T00:00:00')
    latitudes = np.array([55.5, 56.0])
    longitudes = np.array([7.5, 8.0])

    zero_data = np.array([[[0.0, 0.0], [0.0, 0.0]]])

    dummy_dataset_zero_wind = xr.Dataset(
        {
            'u10': (('time', 'latitude', 'longitude'), zero_data),
            'v10': (('time', 'latitude', 'longitude'), zero_data),
            'u100': (('time', 'latitude', 'longitude'), zero_data),
            'v100': (('time', 'latitude', 'longitude'), zero_data),
        },
        coords={
            'time': [time_val],
            'latitude': latitudes,
            'longitude': longitudes,
        },
    )

    target_latitude = 55.75
    target_longitude = 7.75
    wind_resource_zero = WindResource(dummy_dataset_zero_wind, target_latitude, target_longitude)

    # Test at arbitrary height
    target_height = 90
    wind_speed, wind_direction = wind_resource_zero.get_wind_data_at_height(target_height)

    np.testing.assert_almost_equal(wind_speed.item(), 0.0, decimal=5)
    np.testing.assert_almost_equal(wind_direction.item(), 0.0, decimal=5)
