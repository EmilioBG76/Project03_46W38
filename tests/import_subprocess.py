import subprocess

# Run pytest tests in the /Users/cinnamon/Downloads/Project03_46W38/tests/ directory
## IMPORTANT: You can uncomment the specific test you want to run!
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_compute_wind_speed_direction.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_fit_weibull_parameters.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_nrel5mw_turbine_init.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_analysis_plotter_create_yearly_directories.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_analysis_plotter_plot_wind_rose.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_data_load_data_for_year.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_data_load_netcdf_file.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_resource_fit_weibull_distribution.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_resource_get_wind_data_at_height.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_turbine_calculate_aep.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_turbine_get_power_output.py']
#test_command = ['pytest', '/Users/cinnamon/Downloads/Project03_46W38/tests/test_wind_turbine_load_power_curve.py']
process = subprocess.run(test_command, capture_output=True, text=True)

print(process.stdout)
if process.stderr:
    print(process.stderr)

if process.returncode == 0:
    print("All tests passed successfully.")
else:
    print("Some tests failed. Check the output above for details.")