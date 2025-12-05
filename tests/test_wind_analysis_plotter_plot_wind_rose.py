import pytest
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import shutil # for directory cleanup

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

from functions_module import WindAnalysisPlotter
# Test for WindAnalysisPlotter's plot_wind_rose method
@pytest.fixture(scope="function")
# Define a fixture to set up and tear down the WindAnalysisPlotter instance and output directory
def setup_plotter_and_output_dir():
    temp_output_dir = '/Users/cinnamon/Downloads/Project03_46W38/tmp/test_plot_wind_rose_output'
    # Ensure the directory is clean before each test
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)

    plotter = WindAnalysisPlotter(output_dir=temp_output_dir)
    yield plotter, temp_output_dir

    # Cleanup after the test
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
# Define the actual test function for wind rose plotting
def test_plot_wind_rose_creation(setup_plotter_and_output_dir):
    plotter, temp_output_dir = setup_plotter_and_output_dir

    year = 2000
    height = 90

    # Call create_yearly_directories to ensure the year-specific folder exists
    plotter.create_yearly_directories(year, year)

    # Define dummy wind speed data and wind direction data
    np.random.seed(42) # for reproducibility
    wind_speed_data = np.random.rand(100) * 15 # Wind speeds between 0 and 15 m/s
    wind_direction_data = np.random.rand(100) * 360 # Directions between 0 and 360 degrees

    # Call the plotting method
    plotter.plot_wind_rose(wind_speed_data, wind_direction_data, year, height)

    # Assert that the plot file exists
    expected_file_path = os.path.join(temp_output_dir, str(year), f'wind_rose_{height}m_{year}.png')
    assert os.path.exists(expected_file_path)
