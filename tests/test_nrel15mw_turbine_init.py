import pytest
import os
import sys
import pandas as pd

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

from functions_module import NREL15MWWindTurbine
# Define the test function for NREL15MWWindTurbine initialization
def test_nrel15mw_turbine_initialization():
    input_dir = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'

    # Instantiate NREL15MWWindTurbine
    turbine = NREL15MWWindTurbine(input_dir=input_dir)

    # Assert that the name attribute is 'NREL 15 MW'
    assert turbine.name == 'NREL 15 MW'

    # Assert that the hub_height attribute is 150
    assert turbine.hub_height == 150

    # Assert that the power_curve_filepath ends with 'NREL_Reference_15MW_240.csv'
    assert turbine.power_curve_filepath.endswith('NREL_Reference_15MW_240.csv')

    # Assert that the power_curve attribute is a pandas DataFrame and is not empty
    assert isinstance(turbine.power_curve, pd.DataFrame)
    assert not turbine.power_curve.empty
    assert 'wind_speed' in turbine.power_curve.columns
    assert 'power' in turbine.power_curve.columns

    # Assert that cut_in_speed, cut_out_speed, and rated_power are correctly determined
    # Based on the dummy NREL 15 MW power curve: [0, 4, 6, 12, 15, 25, 30] m/s for wind_speed
    # and [0, 0, 1000, 8000, 15000, 15000, 0] kW for power
    # Cut-in speed should be the first wind speed where power > 0, which is 4 m/s (from CSV).
    # Cut-out speed should be the last wind speed where power > 0, which is 25 m/s.
    # Rated power should be the maximum power, which is 15000 kW.

    assert turbine.cut_in_speed == 4.0  
    assert turbine.cut_out_speed == 25.0
    assert turbine.rated_power == 15000.0
