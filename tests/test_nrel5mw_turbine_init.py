import pytest
import os
import sys
import pandas as pd

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

from functions_module import NREL5MWWindTurbine

def test_nrel5mw_turbine_initialization():
    input_dir = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'
    
    # Instantiate NREL5MWWindTurbine
    turbine = NREL5MWWindTurbine(input_dir=input_dir)

    # 6. Assert that the name attribute is 'NREL 5 MW'
    assert turbine.name == 'NREL 5 MW'

    # 7. Assert that the hub_height attribute is 90
    assert turbine.hub_height == 90

    # 8. Assert that the power_curve_filepath ends with 'NREL_Reference_5MW_126.csv'
    assert turbine.power_curve_filepath.endswith('NREL_Reference_5MW_126.csv')

    # 9. Assert that the power_curve attribute is a pandas DataFrame and is not empty
    assert isinstance(turbine.power_curve, pd.DataFrame)
    assert not turbine.power_curve.empty
    assert 'wind_speed' in turbine.power_curve.columns
    assert 'power' in turbine.power_curve.columns

    # 10. Assert that cut_in_speed, cut_out_speed, and rated_power are correctly determined
    # Based on the dummy NREL 5 MW power curve: [0, 3, 5, 10, 12, 25, 30] m/s for wind_speed
    # and [0, 0, 500, 3000, 5000, 5000, 0] kW for power
    # Cut-in speed should be the first wind speed where power > 0, which is 5 m/s.
    # Cut-out speed should be the last wind speed where power > 0, which is 25 m/s.
    # Rated power should be the maximum power, which is 5000 kW.

    assert turbine.cut_in_speed == 5.0
    assert turbine.cut_out_speed == 25.0
    assert turbine.rated_power == 5000.0
