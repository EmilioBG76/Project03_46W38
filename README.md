# Project03_46W38
This is the third and final programming project for the 46W38 course.
This is the third (final) programming project of 46W38 course. The final project 
chosen is WRA based on reanalysis data. 
The project objective is to develop a Python Module for Wind Resource Assessment 
at a specified site using ERA5 reanalysis data. Analyze multi-year hourly wind 
data to estimate key wind energy metrics.

My final project have the following structure including required files as shown 
in the following diagram (under construction yet):
   ```
   Project03_46W38
   ├── inputs/
   │   ├── wind data consisting in multiple NetCDF4 files from period 1997-2008
   │   ├── NREL 5MW wind turbine reference power curve provided
   │   └── NREL 15MW wind turbine reference power curve provided
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
   │   └── functions_module.py containing all the necessary functions to run main.py
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

## Package Overview (UNDER CONSTRUCTION)
* My package contains main.py and functions_module.py scripts
* A description of the class(es) you have implemented in your package, with
clear reference to the file name of the code inside `src`.

## Module/Package Architecture (UNDER CONSTRUCTION)
A description of the module/package architecture, with **at least one diagram**. 

## Installation Instructions (UNDER CONSTRUCTION)
* (Optional) Installation instructions if you have packaged your code.

## Other (UNDER CONSTRUCTION)
* Test coverage of the module/package should be higher than 70%, as evaluated using
`pytest-cov` on the `src` folder, by running:
   ```
   pytest --cov=src tests/
   ```