# This is the main script to run the analysis.

import os                                          # For file path manipulations
import pandas as pd                                # For data handling and saving results
import functions_module                            # Import the entire functions module  
from functions_module import (                     # Import specific classes and functions  
    WindDataLoader,                                # Import WindDataLoader class 
    WindResource,                                  # Import WindResource class
    WindAnalysisPlotter,                           # Import WindAnalysisPlotter class
    NREL5MWWindTurbine,                            # Import NREL 5MW WindTurbine class
    NREL15MWWindTurbine,                           # Import NREL 15MW WindTurbine class
    compute_wind_speed_direction,                  # Import compute wind speed direction function
    fit_weibull_parameters                         # Import fit Weibull parameters function
)

# --- Configuration Parameters ---
TARGET_LATITUDE = 55.75                            # Example: Latitude for Horns Rev 1
TARGET_LONGITUDE = 7.75                            # Example: Longitude for Horns Rev 1
TARGET_HEIGHT = 90                                 # Example: Target height for analysis (e.g., NREL 5 MW hub height)
START_YEAR = 1997                                  # Start year for analysis
END_YEAR = 2008                                    # End year for analysis
INPUT_DIR = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'
OUTPUT_BASE_DIR = '/Users/cinnamon/Downloads/Project03_46W38/outputs'

# --- Main Analysis Logic ---
# Define main analysis function    
def run_analysis():
    print(f"Starting wind resource analysis for {TARGET_LATITUDE}°N, {TARGET_LONGITUDE}°E at {TARGET_HEIGHT}m...")
    
    # Instantiate WindDataLoader and WindAnalysisPlotter
    data_loader = WindDataLoader(input_dir=INPUT_DIR)
    plotter = WindAnalysisPlotter(output_dir=OUTPUT_BASE_DIR)

    # Create yearly output directories
    plotter.create_yearly_directories(START_YEAR, END_YEAR)

    aep_results = [] # To store AEP results for all years and turbines

    # Initialize turbine models once
    nrel_5mw_turbine = NREL5MWWindTurbine(input_dir=INPUT_DIR)
    nrel_15mw_turbine = NREL15MWWindTurbine(input_dir=INPUT_DIR)
# --- Loop through each year for analysis ---
    for year in range(START_YEAR, END_YEAR + 1):
        print() # Add a blank line for readability
        print(f"--- Processing Year: {year} ---")
        # Load and parse multiple provided netCDF4 files for the current year
        print(f"Loading wind data for {year}...")
        # Load data only for the current year to ensure Weibull fit is specific to that year
        combined_dataset_year = data_loader.load_data_for_year(year)

        if combined_dataset_year is None:
            print(f"Failed to load wind data for year {year}. Skipping AEP calculation for this year.")
            continue
        
        print(f"Wind data loaded successfully for {year}.")
        # Initialize WindResource for the target location with data for the current year
        wind_resource = WindResource(combined_dataset_year, TARGET_LATITUDE, TARGET_LONGITUDE)

        # Compute wind speed and wind direction time series at target height
        print(f"Computing wind speed and direction at {TARGET_HEIGHT}m for {year}...")
        wind_speed_series, wind_direction_series = wind_resource.get_wind_data_at_height(TARGET_HEIGHT)

        # Fit Weibull distribution for wind speed at the given location and height for the current year
        print(f"Fitting Weibull distribution for {year}...")
        k_weibull, A_weibull = wind_resource.fit_weibull_distribution(TARGET_HEIGHT)
        
        if k_weibull is not None and A_weibull is not None:
            print(f"Weibull parameters fitted for {year}: k={k_weibull:.2f}, A={A_weibull:.2f}")

            # 6. Plot wind speed distribution (histogram vs. fitted Weibull distribution)
            plotter.plot_speed_distribution(wind_speed_series, k_weibull, A_weibull, year, TARGET_HEIGHT)

            # 7. Plot wind rose diagram
            plotter.plot_wind_rose(wind_speed_series, wind_direction_series, year, TARGET_HEIGHT)

            # 8. Compute AEP for both turbines
            print(f"Calculating AEP for {year}...")
            aep_5mw = nrel_5mw_turbine.calculate_aep(k_weibull, A_weibull)
            aep_15mw = nrel_15mw_turbine.calculate_aep(k_weibull, A_weibull)
            
            aep_results.append({'Year': year, 'Turbine': nrel_5mw_turbine.name, 'AEP (MWh)': aep_5mw})
            aep_results.append({'Year': year, 'Turbine': nrel_15mw_turbine.name, 'AEP (MWh)': aep_15mw})

            print(f"NREL 5 MW AEP for {year}: {aep_5mw:.2f} MWh")
            print(f"NREL 15 MW AEP for {year}: {aep_15mw:.2f} MWh")

        else:
            print(f"Weibull fitting failed for year {year}, skipping distribution plot and AEP calculation.")

    # Save all AEP results to a single table
    if aep_results:
        aep_df = pd.DataFrame(aep_results)
        aep_output_path = os.path.join(OUTPUT_BASE_DIR, 'aep_summary_1997-2008.csv')
        aep_df.to_csv(aep_output_path, index=False)
        print() # Add a blank line for readability
        print(f"All AEP results saved to {aep_output_path}")
        print() # Add a blank line for readability
        print("AEP Summary:")
        print(aep_df)
    else:
        print("No AEP results to save.")

    print() # Add a blank line for readability
    print("Overall analysis complete.")
# Run the main analysis function
if __name__ == "__main__":
    run_analysis()