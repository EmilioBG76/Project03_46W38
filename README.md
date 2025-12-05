# Project03_46W38
This is the third and final programming project for the 46W38 course.
This is the third (final) programming project of 46W38 course. The final project 
chosen is WRA based on reanalysis data. 
The project objective is to develop a Python Module for Wind Resource Assessment 
at a specified site using ERA5 reanalysis data. Analyze multi-year hourly wind 
data to estimate key wind energy metrics.

## Package Architecture (UNDER CONSTRUCTION)
My final project have the following structure including required files as shown 
in the following diagram (under construction yet):
   ```
   Project03_46W38
   ├── inputs/
   │   ├── wind data consisting in multiple NetCDF4 files from period 1997-2008
   │   ├── NREL 5MW wind turbine reference power curve provided
   │   ├── NREL 15MW wind turbine reference power curve provided
   │   └── simple power curve for testing
   ├── outputs/
   │   ├── wind roses and wind speed distributions (yearly sorted)
   │   ├── AEP summary calculation results for the data period provided (1997-2008)
   │   └── initial plotting results (will delete later)
   ├── src/
   │   ├── main.py 
   │   ├── functions_module
   │   └── __init__.py
   ├── tests/
   │   ├── tests for functions in the functions_module
   │   └── test for main script
   ├── examples/
   │   ├── main.py (will run in evaluation)
   │   └── functions_module.py containing all necessary functions to run main.py
   ├── .gitignore
   ├── LICENSE
   ├── README.md
   ├── pyproject03_46w38.toml
   ├── environment yml file to be completed  
   └── to be completed
   ```
The `main.py` script inside the `examples` folder demonstrates, in a
clear and structured manner, how the required functions are called and 
executed.
The `functions_module.py` script contains all the functions definition.
The `inputs` folder...
The `outputs` folder...
The `src` folder...
The `tests` folder...
....

## Package Overview (UNDER CONSTRUCTION)
* My package contains main.py and functions_module.py scripts

* The classes defined in `src/functions_module.py` are:
- WindTurbine: represents a generic wind turbine containing all the necessary
information and functionalities required to calculate AEP. WindTurbine class
includes attributes and methods.
- NREL5MWWindTurbine and NREL15MWWindTurbine: both represent a specific type of wind turbine, NREL 5MW and 15MW reference wind turbines. Includes functionalities, attributes and wind turbine characteristics.
- WindResource: Works with the wind conditions at a specific location, taking raw
wind data and provides wind speed and direction for a chosen height. In addition fit Weibull distribution. Enables interpolation and vertical extrapolation needed. Also includes attributes and methods. 
- WindDataLoader:  Necessary for data management, reading and organizing netCDF4 files for a specific year. Providing all wind data for the chosen year. 
Allows easy access and preparation to wind data provided&stored in netCDF4 format.
Also includes attributes and methods.
- WindAnalysisPlotter: Is the class for visualization of the wind data analized.
Create folder structure for storing the results in a proper way. Generate the output plotting obtained for wind speed distributions and wind roses too.
Includes attributes and methods.

* The `src/main.py`script imports and uses the detailed classes. It contains the 
required WRA, doing the required tasks as I detail here:
- Setting up the entire analysis: location, height and years to study.
- Managing data: loading raw data provided in netCDF4 files for each provided year.
- Analyzing wind behaviour: calculating wind speed and wind direction. Fitting
Weibull distribution for each provided year.
- Study wind turbines: Using provided NREL reference wind turbines data calculates
AEP for every single year for the period provided (1997-2008).
- Visualizing: Plotting wind speed distributions and wind roses at a specific height for every single year for the period provided (1997-2008).
- Saving output data summary: Providing all AEP results in a csv file for every single year for the period provided (1997-2008).
 

## Installation Instructions
* Not necessary specific intructions needed apart from using Github repository and its directories created for this project.

## Testing work developed
* I have also tested coverage of the package and it is higher than 70%. It has been evaluated used, as indicated, `pytest-cov` on the `src` folder.
It can be seen how I have created separated test files in order to fully check
the `src/functions_module.py` and `src/main.py` scripts correct functioning.
All the test stored in `tests/` directory are:
- test_compute_wind_speed_direction: Verification of the compute_wind _speed_direction_function which is found in `src/functions_module.py`. Ensuring that this function converts properly u and v wind components into wind speed and wind direction necessary for obtaining the correct results.
- test_fit_weibull_parameters: Testing the fit_weibull_parameters function.
Ensuring this functions properly calculates Weibull parameters(A and k) taking into account the provided wind speed data. It has been reviewed several scenarios for that purpose.
- test_wind_analysis_plotter_create_yearly_directories: Testing the create_yearly_directories method for the WindAnalysisPlotter class. Ensuring that it creates properly the output directory sorted by subfolder years (1997-2008).
- test_wind_analysis_plotter_plot_speed_distribution: Validates the plot_speed_distribution method for the WindAnalysisPlotter class. Ensuring that this method generates and saves properly the required plots.
- test_wind_analysis_plotter_plot_wind_rose: under work
- test_wind_data_loader_load_data_for_year: under work
- test_wind_data_loader_load_netcdf_file: under work
- test_wind_resource_fit_weibull_distribution: under work
- test_wind_resource_get_wind_data_at_height: under work
- test_wind_turbine_calculate_aep: under work
- test_wind_turbine_get_power_output: under work
- test_wind_turbine_load_power_curve: under work
- test_main_script: under work





