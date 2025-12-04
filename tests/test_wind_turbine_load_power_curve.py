import pytest
import os
import sys
import pandas as pd

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

from functions_module import WindTurbine

def test_load_power_curve_success():
    # Use an existing dummy power curve file for testing
    input_dir = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'
    valid_power_curve_filepath = 'NREL_Reference_5MW_126.csv'
    
    # Instantiate WindTurbine, which calls _load_power_curve during __init__
    turbine = WindTurbine(name='Test Turbine', hub_height=90, power_curve_filepath=valid_power_curve_filepath, input_dir=input_dir)
    
    # Assertions for successful loading
    assert isinstance(turbine.power_curve, pd.DataFrame)
    assert not turbine.power_curve.empty
    assert 'wind_speed' in turbine.power_curve.columns
    assert 'power' in turbine.power_curve.columns
    assert turbine.cut_in_speed is not None
    assert turbine.cut_out_speed is not None
    assert turbine.rated_power is not None

def test_load_power_curve_file_not_found():
    # Use a non-existent power curve file for testing FileNotFoundError
    input_dir = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'
    non_existent_power_curve_filepath = 'non_existent_curve.csv'
    
    # Expect FileNotFoundError when instantiating WindTurbine with a non-existent file
    with pytest.raises(FileNotFoundError):
        WindTurbine(name='Failing Turbine', hub_height=90, power_curve_filepath=non_existent_power_curve_filepath, input_dir=input_dir)
