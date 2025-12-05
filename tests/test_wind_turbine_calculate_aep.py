import pytest
import numpy as np
import pandas as pd
import os
import sys
import scipy.stats
import scipy.integrate

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')
import functions_module
from functions_module import WindTurbine, NREL5MWWindTurbine
# Fixtures for setting up WindTurbine instances with known power curves
@pytest.fixture
# Define a simple WindTurbine with a known power curve for AEP testing
def setup_simple_turbine_for_aep():
    input_dir = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'
    power_curve_filepath = 'simple_aep_power_curve.csv'
    nrel_simple_aep_power_curve_path = os.path.join(input_dir, power_curve_filepath)

    # Create a simplified dummy power curve for AEP testing
    # Power is 1000 kW between 5m/s and 10m/s, and linearly decreases from 10m/s to 15m/s where it becomes 0.
    data = {
        'wind_speed': [0, 4.9, 5.0, 10.0, 10.1, 25],
        'power': [0, 0, 1000, 1000, 0, 0] # Power curve has cut-in at 5m/s and cut-out at 10m/s for power generation
    }
    df = pd.DataFrame(data)
    df.to_csv(nrel_simple_aep_power_curve_path, index=False)

    return WindTurbine(name='Simple Test Turbine', hub_height=90, power_curve_filepath=power_curve_filepath, input_dir=input_dir)
# Fixture for setting up NREL5MWWindTurbine instance
@pytest.fixture
def setup_nrel5mw_turbine():
    input_dir = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'
    # The NREL 5 MW power curve is expected to be created by previous steps
    return NREL5MWWindTurbine(input_dir=input_dir)
# Fixtures and tests for WindTurbine.calculate_aep method
def test_calculate_aep_with_valid_parameters(setup_simple_turbine_for_aep):
    turbine = setup_simple_turbine_for_aep
    k_test = 2.0
    A_test = 7.0

    # Manual calculation for expected AEP to verify
    class WindTurbineVerification:
        def __init__(self, power_curve_data):
            self.power_curve = pd.DataFrame(power_curve_data)
            self.power_curve.columns = ['wind_speed', 'power']

        def get_power_output(self, wind_speed):
            return np.interp(wind_speed,
                            self.power_curve['wind_speed'],
                            self.power_curve['power'],
                            left=0, right=0)

    simple_power_curve_data = {
        'wind_speed': [0, 4.9, 5.0, 10.0, 10.1, 25],
        'power': [0, 0, 1000, 1000, 0, 0]
    }
    verification_turbine = WindTurbineVerification(simple_power_curve_data)

    def weibull_pdf(u, k_val, A_val):
        return scipy.stats.weibull_min.pdf(u, k_val, loc=0, scale=A_val)

    def integrand_verifier(u, k_val, A_val, turbine_obj):
        power_output_kw = turbine_obj.get_power_output(u)
        return power_output_kw * weibull_pdf(u, k_val, A_val)

    # The cut-in/cut-out speeds are now determined by the WindTurbine class itself
    # For this simple curve, cut_in_speed should be 5.0 and cut_out_speed should be 10.0
    lower_bound = turbine.cut_in_speed
    upper_bound = turbine.cut_out_speed

    integral_val, _ = scipy.integrate.quad(integrand_verifier, lower_bound, upper_bound, args=(k_test, A_test, verification_turbine))
    expected_aep = 1.0 * 8760 * (integral_val / 1000)

    calculated_aep = turbine.calculate_aep(k_test, A_test)
    np.testing.assert_almost_equal(calculated_aep, expected_aep, decimal=2)
# Define tests for AEP calculation when k parameter is None
def test_calculate_aep_k_is_none(setup_simple_turbine_for_aep):
    turbine = setup_simple_turbine_for_aep
    k_test = None
    A_test = 7.0
    calculated_aep = turbine.calculate_aep(k_test, A_test)
    assert np.isclose(calculated_aep, 0.0)
# Define test for AEP calculation when A parameter is None
def test_calculate_aep_a_is_none(setup_simple_turbine_for_aep):
    turbine = setup_simple_turbine_for_aep
    k_test = 2.0
    A_test = None
    calculated_aep = turbine.calculate_aep(k_test, A_test)
    assert np.isclose(calculated_aep, 0.0)
# Define test for AEP zero power curve range
def test_calculate_aep_zero_power_curve_range(setup_simple_turbine_for_aep):
    turbine = setup_simple_turbine_for_aep
    # These parameters result in very low wind speeds, mostly outside the power-producing range of 5-10m/s
    k_test = 5.0
    A_test = 1.0
    calculated_aep = turbine.calculate_aep(k_test, A_test)
    assert np.isclose(calculated_aep, 0.0, atol=1e-3) # Use np.isclose instead of assert_almost_equal with atol
# Define test for AEP calculation using NREL5MWWindTurbine with known parameters
def test_calculate_aep_using_nrel5mw_turbine(setup_nrel5mw_turbine):
    turbine = setup_nrel5mw_turbine
    # These k and A values are from the 1997 main.py output
    k_test = 2.17
    A_test = 10.67
    calculated_aep = turbine.calculate_aep(k_test, A_test)
    # The expected value calculated using the dummy power curve for NREL_Reference_5MW_126.csv
    np.testing.assert_almost_equal(calculated_aep, 23284.64, decimal=2)

