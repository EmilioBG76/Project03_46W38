import pytest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import shutil

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

from functions_module import WindAnalysisPlotter

@pytest.fixture(scope="function")
def setup_plotter_and_output_dir():
    temp_output_dir = '/Users/cinnamon/Downloads/Project03_46W38/tmp/test_plot_speed_distribution_output'
    # Ensure the directory is clean before each test
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)
    
    plotter = WindAnalysisPlotter(output_dir=temp_output_dir)
    yield plotter, temp_output_dir
    
    # Cleanup after the test
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

def test_plot_speed_distribution_creation(setup_plotter_and_output_dir):
    plotter, temp_output_dir = setup_plotter_and_output_dir
    
    year = 2000
    height = 90
    
    # Call create_yearly_directories to ensure the year-specific folder exists
    plotter.create_yearly_directories(year, year)
    
    # Define dummy wind speed data and Weibull parameters
    np.random.seed(42) # for reproducibility
    wind_speed_data = np.random.normal(loc=10, scale=3, size=1000)
    k_weibull = 2.0
    A_weibull = 10.0
    
    # Call the plotting method
    plotter.plot_speed_distribution(wind_speed_data, k_weibull, A_weibull, year, height)
    
    # Assert that the plot file exists
    expected_file_path = os.path.join(temp_output_dir, str(year), f'wind_speed_distribution_{height}m_{year}.png')
    assert os.path.exists(expected_file_path)
    

def test_plot_speed_distribution_with_none_weibull_params(setup_plotter_and_output_dir):
    plotter, temp_output_dir = setup_plotter_and_output_dir
    
    year = 2001
    height = 100
    
    # Call create_yearly_directories to ensure the year-specific folder exists
    plotter.create_yearly_directories(year, year)
    
    # Define dummy wind speed data (Weibull params will be None)
    np.random.seed(43) # for reproducibility
    wind_speed_data = np.random.normal(loc=8, scale=2, size=500)
    
    # Call the plotting method with None for k and A
    plotter.plot_speed_distribution(wind_speed_data, None, None, year, height)
    
    # Assert that the plot file still exists
    expected_file_path = os.path.join(temp_output_dir, str(year), f'wind_speed_distribution_{height}m_{year}.png')
    assert os.path.exists(expected_file_path)
