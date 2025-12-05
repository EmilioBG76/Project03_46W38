import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')
import functions_module
from functions_module import WindTurbine
# Fixtures and tests for WindTurbine.get_power_output method
@pytest.fixture
# Define a WindTurbine with a known power curve for testing get_power_output
def setup_turbine():
    input_dir = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'
    power_curve_filepath = 'NREL_Reference_5MW_126.csv'
    # Ensure the dummy power curve exists with specific values for testing
    nrel5mw_power_curve_path = os.path.join(input_dir, power_curve_filepath)

    # Power curve data points as specified in the instructions
    data_5mw = {
        'wind_speed': [0, 3, 5, 10, 12, 25, 30],
        'power': [0, 0, 500, 3000, 5000, 5000, 0]
    }
    df_5mw = pd.DataFrame(data_5mw)
    df_5mw.to_csv(nrel5mw_power_curve_path, index=False)

    return WindTurbine(name='Test Turbine', hub_height=90, power_curve_filepath=power_curve_filepath, input_dir=input_dir)
# Define tests for WindTurbine.get_power_output at cut-in speed
def test_power_output_at_cut_in(setup_turbine):
    turbine = setup_turbine
    # Cut-in speed for the dummy curve is 3 m/s
    assert np.isclose(turbine.get_power_output(2.9), 0) # Below cut-in
    assert np.isclose(turbine.get_power_output(3.0), 0)  # At cut-in
    assert np.isclose(turbine.get_power_output(3.1), 25.0, atol=1e-6) # Just above cut-in (linear interpolation between 3m/s (0kW) and 5m/s (500kW))
# Define tests for WindTurbine.get_power_output at cut-out speed
def test_power_output_at_cut_out(setup_turbine):
    turbine = setup_turbine
    # Cut-out speed for the dummy curve is 25 m/s for full power, then tapers to 30 m/s for 0 power.
    assert np.isclose(turbine.get_power_output(25.0), 5000) # At 25 m/s
    assert np.isclose(turbine.get_power_output(25.1), 4900.0, atol=1e-6) # Just above 25 m/s (linear interpolation between 25m/s (5000kW) and 30m/s (0kW))
    assert np.isclose(turbine.get_power_output(29.9), 100.0, atol=1e-6) # Just before 30 m/s (linear interpolation between 25m/s (5000kW) and 30m/s (0kW))
# Define test power output interpolation between defined points
def test_power_output_interpolation(setup_turbine):
    turbine = setup_turbine
    # Test linear interpolation between 5m/s (500kW) and 10m/s (3000kW)
    # At 7.5 m/s, power = 500 + (7.5 - 5) / (10 - 5) * (3000 - 500) = 500 + 2.5 / 5 * 2500 = 500 + 0.5 * 2500 = 500 + 1250 = 1750 kW
    assert np.isclose(turbine.get_power_output(7.5), 1750.0)
# Define test power output at rated speed range
def test_power_output_rated_speed(setup_turbine):
    turbine = setup_turbine
    # Rated speed range is 12m/s to 25m/s where power is 5000kW
    assert np.isclose(turbine.get_power_output(12.0), 5000) # At rated speed start
    assert np.isclose(turbine.get_power_output(18.0), 5000) # Within rated speed range
    assert np.isclose(turbine.get_power_output(24.9), 5000) # Just below rated speed end
# Define test power output outside operating range
def test_power_output_outside_operating_range(setup_turbine):
    turbine = setup_turbine
    # Well below cut-in speed
    assert np.isclose(turbine.get_power_output(0.0), 0) # Zero wind speed
    assert np.isclose(turbine.get_power_output(1.0), 0) # Below cut-in
    # At the final point of the curve where power drops to 0
    assert np.isclose(turbine.get_power_output(30.0), 0) 
    # Well above the last defined point in the power curve
    assert np.isclose(turbine.get_power_output(35.0), 0)
