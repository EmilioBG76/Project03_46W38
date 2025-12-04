# Project03_46W38
This is the third (final) programming project of 46W38 course. The final project 
chosen is WRA based on reanalysis data. 
The project objective is to develop a Python Module for Wind Resource Assessment 
at a specified site using ERA5 reanalysis data. Analyze multi-year hourly wind 
data to estimate key wind energy metrics.

1. My final project have the following structure including required files as shown in 
the following diagram (under construction yet):
   ```
   Project03_46W38
   ├── inputs/
   │   ├── wind data in multiple NetCDF4 files provided (1997-2008)
   │   ├── NREL 5MW wind turbine reference power curve provided
   │   ├── NREL 15MW wind turbine reference power curve provided
   ├── outputs/
   │   └── wind roses and wind speed distributions (yearly sorted)
   │   └── AEP summary calculation results for the data period provided (1997-2008)
   │   └── initial plotting results (will delete later)
   ├── src/
   │   ├── main.py 
   │   └── functions_module
   │   └── __init__.py
   ├── tests/
   │   └── functions_module functions complete tests 
   │   └── main test
   ├── examples/
   │   └── main.py (will run in evaluation)
   ├── .gitignore
   ├── LICENSE
   ├── README.md
   ├── to be completed
   ├── to be completed  
   └── to be completed
   ```