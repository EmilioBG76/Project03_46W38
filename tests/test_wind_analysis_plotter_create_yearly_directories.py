import pytest
import os
import sys
import shutil

# Add the path to functions_module.py to sys.path
sys.path.insert(0, '/Users/cinnamon/Downloads/Project03_46W38/src')

from functions_module import WindAnalysisPlotter

@pytest.fixture(scope="function")
def setup_output_directory():
    temp_output_dir = '/Users/cinnamon/Downloads/Project03_46W38/tmp/test_output_yearly_dirs'
    # Ensure the directory is clean before each test
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)
    os.makedirs(temp_output_dir)
    yield temp_output_dir
    # Clean up after the test
    if os.path.exists(temp_output_dir):
        shutil.rmtree(temp_output_dir)

def test_create_yearly_directories(setup_output_directory):
    base_output_dir = setup_output_directory
    plotter = WindAnalysisPlotter(output_dir=base_output_dir)

    start_year = 2000
    end_year = 2002

    # Call the method to create yearly directories
    plotter.create_yearly_directories(start_year, end_year)

    # Assert that each yearly subdirectory exists
    for year in range(start_year, end_year + 1):
        expected_dir = os.path.join(base_output_dir, str(year))
        assert os.path.isdir(expected_dir), f"Directory for year {year} was not created."
