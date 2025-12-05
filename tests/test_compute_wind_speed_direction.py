import numpy as np
import xarray as xr
import sys

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')
from functions_module import compute_wind_speed_direction

# Test cases for compute_wind_speed_direction function
def test_positive_uv_components():
    # Test case 1: Positive u and v components (Quadrant I)
    u = 3.0
    v = 4.0
    wind_speed, wind_direction = compute_wind_speed_direction(u, v)
    np.testing.assert_almost_equal(wind_speed, 5.0)
    np.testing.assert_almost_equal(wind_direction, 216.8698976, decimal=5) # arctan2(3,4) is 36.87 deg from north to east, meteorological from is 180 + 36.87 = 216.87
    np.testing.assert_almost_equal(wind_direction, 216.8698976, decimal=5) # arctan2(3,4) is 36.87 deg from north to east, meteorological from is 180 + 36.87 = 216.87

def test_negative_uv_components():
    # Test case 2: Negative u and v components (Quadrant III)
    u = -3.0
    v = -4.0
    wind_speed, wind_direction = compute_wind_speed_direction(u, v)
    np.testing.assert_almost_equal(wind_speed, 5.0)
    np.testing.assert_almost_equal(wind_direction, 36.8698976, decimal=5) # arctan2(-3,-4) is -143.13 deg from north to east, meteorological from is 180 + (-143.13) = 36.87

def test_mixed_uv_components_quadrant_ii():
    # Test case 3: Mixed u and v components (Quadrant II: u negative, v positive)
    # arctan2(-3,4) is -36.87 deg from north to west, meteorological from is 180 + (-36.87) = 143.13
    u = -3.0
    v = 4.0
    wind_speed, wind_direction = compute_wind_speed_direction(u, v)
    np.testing.assert_almost_equal(wind_speed, 5.0)
    np.testing.assert_almost_equal(wind_direction, 143.1301023, decimal=5) # arctan2(-3,4) is -36.87 deg from north to west, meteorological from is 180 + (-36.87) = 143.13
    # Test case 4: Mixed u and v components (Quadrant IV: u positive, v negative)
    # arctan2(3,-4) is 143.13 deg from north to west, meteorological from is 180 + 143.13 = 323.13 

def test_mixed_uv_components_quadrant_iv():
    # Test case 4: Mixed u and v components (Quadrant IV: u positive, v negative)
    u = 3.0
    v = -4.0
    wind_speed, wind_direction = compute_wind_speed_direction(u, v)
    np.testing.assert_almost_equal(wind_speed, 5.0)
    np.testing.assert_almost_equal(wind_direction, 323.1301023, decimal=5) # arctan2(3,-4) is 143.13 deg from north to west, meteorological from is 180 + 143.13 = 323.13

def test_zero_u_positive_v():
    # Test case 5: Zero u, positive v (North wind)
    u = 0.0
    v = 5.0
    wind_speed, wind_direction = compute_wind_speed_direction(u, v)
    np.testing.assert_almost_equal(wind_speed, 5.0)
    np.testing.assert_almost_equal(wind_direction, 180.0, decimal=5) # arctan2(0,5) is 0 deg, meteorological from is 180 + 0 = 180

def test_positive_u_zero_v():
    # Test case 6: Positive u, zero v (East wind)
    u = 5.0
    v = 0.0
    wind_speed, wind_direction = compute_wind_speed_direction(u, v)
    np.testing.assert_almost_equal(wind_speed, 5.0)
    np.testing.assert_almost_equal(wind_direction, 270.0, decimal=5) # arctan2(5,0) is 90 deg, meteorological from is 180 + 90 = 270

def test_zero_uv_components():
    # Test case 7: Both u and v are zero (no wind)
    u = 0.0
    v = 0.0
    wind_speed, wind_direction = compute_wind_speed_direction(u, v)
    np.testing.assert_almost_equal(wind_speed, 0.0)
    np.testing.assert_almost_equal(wind_direction, 0.0, decimal=5) # Conventionally 0 for no wind

def test_array_input():
    # Test case 8: Array inputs
    u_arr = np.array([3.0, -3.0, 0.0, 5.0])
    v_arr = np.array([4.0, -4.0, 5.0, 0.0])
    expected_speeds = np.array([5.0, 5.0, 5.0, 5.0])
    expected_directions = np.array([216.8698976, 36.8698976, 180.0, 270.0])
    wind_speed, wind_direction = compute_wind_speed_direction(u_arr, v_arr)
    np.testing.assert_almost_equal(wind_speed, expected_speeds, decimal=5)
    np.testing.assert_almost_equal(wind_direction, expected_directions, decimal=5)

def test_xarray_input():
    # Test case 9: xarray DataArray inputs
    u_xr = xr.DataArray(np.array([3.0, -3.0]), dims=['time'])
    v_xr = xr.DataArray(np.array([4.0, -4.0]), dims=['time'])
    expected_speeds = np.array([5.0, 5.0])
    expected_directions = np.array([216.8698976, 36.8698976])
    wind_speed_xr, wind_direction_xr = compute_wind_speed_direction(u_xr, v_xr)
    np.testing.assert_almost_equal(wind_speed_xr.values, expected_speeds, decimal=5)
    np.testing.assert_almost_equal(wind_direction_xr.values, expected_directions, decimal=5)
    assert isinstance(wind_speed_xr, xr.DataArray)
    assert isinstance(wind_direction_xr, xr.DataArray)
