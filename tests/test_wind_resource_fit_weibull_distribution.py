import pytest
import numpy as np
import xarray as xr
import sys
import os
import pandas as pd
import scipy.stats

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

from functions_module import WindResource, compute_wind_speed_direction, fit_weibull_parameters

@pytest.fixture
def setup_wind_resource_for_weibull():
    # Create a dummy xarray.Dataset with multiple time steps for Weibull fitting
    num_time_steps = 1000 # Increased time steps for better fitting
    time_coords = pd.date_range('2000-01-01', periods=num_time_steps, freq='h')
    latitudes = np.array([55.5, 56.0])
    longitudes = np.array([7.5, 8.0])

    # Generate slightly varying u and v components at 10m and 100m
    # To ensure a distribution that can be fitted.
    np.random.seed(0) # For reproducibility
    u10_data = np.random.normal(loc=2, scale=1, size=(num_time_steps, len(latitudes), len(longitudes)))
    v10_data = np.random.normal(loc=3, scale=1, size=(num_time_steps, len(latitudes), len(longitudes)))
    u100_data = np.random.normal(loc=8, scale=2, size=(num_time_steps, len(latitudes), len(longitudes)))
    v100_data = np.random.normal(loc=10, scale=2, size=(num_time_steps, len(latitudes), len(longitudes)))

    dummy_dataset = xr.Dataset(
        {
            'u10': (('time', 'latitude', 'longitude'), u10_data),
            'v10': (('time', 'latitude', 'longitude'), v10_data),
            'u100': (('time', 'latitude', 'longitude'), u100_data),
            'v100': (('time', 'latitude', 'longitude'), v100_data),
        },
        coords={
            'time': time_coords,
            'latitude': latitudes,
            'longitude': longitudes,
        },
    )

    target_latitude = 55.75
    target_longitude = 7.75
    return WindResource(dummy_dataset, target_latitude, target_longitude)

def test_fit_weibull_parameters_at_target_height(setup_wind_resource_for_weibull):
    wind_resource = setup_wind_resource_for_weibull
    target_height = 90

    k_fitted, A_fitted = wind_resource.fit_weibull_distribution(target_height)

    assert k_fitted is not None
    assert A_fitted is not None
    assert k_fitted > 0
    assert A_fitted > 0

    # To make assertions robust, re-fit Weibull on the generated wind speeds from the fixture
    wind_speed_series, _ = wind_resource.get_wind_data_at_height(target_height)
    speeds_for_fit = wind_speed_series.values[wind_speed_series.values > 0]

    if len(speeds_for_fit) >= 2:
        # Fit Weibull to the actual generated speeds to get expected k and A
        k_expected, _, A_expected = scipy.stats.weibull_min.fit(speeds_for_fit, floc=0)
        np.testing.assert_almost_equal(k_fitted, k_expected, decimal=1)
        np.testing.assert_almost_equal(A_fitted, A_expected, decimal=1)
    else:
        pytest.fail("Insufficient valid wind speed data points for fitting in fixture")

def test_fit_weibull_parameters_for_zero_wind_speeds():
    num_time_steps = 100
    time_coords = pd.date_range('2000-01-01', periods=num_time_steps, freq='h')
    # Use a single point to represent the target location
    latitudes = np.array([55.75])
    longitudes = np.array([7.75])

    zero_data = np.zeros((num_time_steps, len(latitudes), len(longitudes)))

    dummy_dataset_zero_wind = xr.Dataset(
        {
            'u10': (('time', 'latitude', 'longitude'), zero_data),
            'v10': (('time', 'latitude', 'longitude'), zero_data),
            'u100': (('time', 'latitude', 'longitude'), zero_data),
            'v100': (('time', 'latitude', 'longitude'), zero_data),
        },
        coords={
            'time': time_coords,
            'latitude': latitudes,
            'longitude': longitudes,
        },
    )

    target_latitude = 55.75
    target_longitude = 7.75
    with pytest.warns(RuntimeWarning): # Expect scipy RuntimeWarning
        wind_resource_zero = WindResource(dummy_dataset_zero_wind, target_latitude, target_longitude)
        target_height = 90

        k, A = wind_resource_zero.fit_weibull_distribution(target_height)

        assert k is None
        assert A is None
