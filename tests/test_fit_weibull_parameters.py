import pytest
import numpy as np
import scipy.stats
import sys
import os

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')
from functions_module import fit_weibull_parameters

def test_known_distribution():
    # Test with a known wind speed distribution (e.g., synthetic data from a Weibull distribution)
    k_true = 2.5
    A_true = 8.0
    # Generate some synthetic wind speed data from a Weibull distribution
    np.random.seed(42) # for reproducibility
    wind_speeds = scipy.stats.weibull_min.rvs(k_true, loc=0, scale=A_true, size=10000)
    
    k_fitted, A_fitted = fit_weibull_parameters(wind_speeds)
    
    # Assert that fitted parameters are close to true parameters
    # Use a larger tolerance for fitted parameters due to estimation
    np.testing.assert_almost_equal(k_fitted, k_true, decimal=1)
    np.testing.assert_almost_equal(A_fitted, A_true, decimal=1)

def test_insufficient_data():
    # Test with an array containing very few data points (less than 2)
    wind_speeds_single = np.array([5.0])
    k, A = fit_weibull_parameters(wind_speeds_single)
    assert k is None
    assert A is None
    
    wind_speeds_empty = np.array([])
    k, A = fit_weibull_parameters(wind_speeds_empty)
    assert k is None
    assert A is None

def test_zero_wind_speeds():
    # Test with all zero wind speeds
    wind_speeds = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    k, A = fit_weibull_parameters(wind_speeds)
    # For all zero speeds, fitting might return small non-zero values or fail
    # Based on current implementation, it might fail or return k=0, A=0. For simplicity, check for None or near-zero
    # The function handles speeds >= 0, so if all are zero, it should return None, None
    assert k is None or np.isclose(k, 0, atol=1e-5)
    assert A is None or np.isclose(A, 0, atol=1e-5)

def test_constant_non_zero_wind_speed():
    # Test with a constant non-zero wind speed
    wind_speeds = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    k, A = fit_weibull_parameters(wind_speeds)
    # For constant speeds, k tends to be very large and A equal to the constant speed
    # scipy.stats.weibull_min.fit might return a high k and A very close to the constant value
    assert k is not None
    assert A is not None
    np.testing.assert_almost_equal(A, 10.0, decimal=5)
    # k for constant speed should be very large, typically limited by float precision, or very high value
    assert k > 10.0 # Expect k to be a large value, effectively infinity for perfect constant

def test_mixed_data():
    # Test with a mix of zero and non-zero wind speeds
    wind_speeds = np.array([0.0, 2.0, 5.0, 7.0, 10.0])
    k, A = fit_weibull_parameters(wind_speeds)
    assert k is not None
    assert A is not None
    # Specific values are hard to predict, just ensure it fits without error
    assert k > 0
    assert A > 0
