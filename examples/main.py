# This is the main script to run the analysis.

import os                                          # For file path manipulations
import pandas as pd                                # For data handling and saving results
import functions_module
from functions_module import (
    load_wind_data,
    WindResource,
    create_yearly_output_directories,
    plot_wind_speed_distribution,
    plot_wind_rose,
    calculate_aep,
    NREL5MWWindTurbine,
    NREL15MWWindTurbine
)

# --- Configuration Parameters ---
TARGET_LATITUDE = 55.75  # Example: Latitude for Horns Rev 1
TARGET_LONGITUDE = 7.75  # Example: Longitude for Horns Rev 1
TARGET_HEIGHT = 90           # Example: Target height for analysis (e.g., NREL 5 MW hub height)
START_YEAR = 1997
END_YEAR = 2008
INPUT_DIR = '/Users/cinnamon/Downloads/Project03_46W38/inputs/'
OUTPUT_BASE_DIR = '/Users/cinnamon/Downloads/Project03_46W38/outputs'

# --- Main Analysis Logic ---
    
def run_analysis():
    print(f"Starting wind resource analysis for {TARGET_LATITUDE}°N, {TARGET_LONGITUDE}°E at {TARGET_HEIGHT}m...")
    
    # 0. Create yearly output directories
    create_yearly_output_directories(OUTPUT_BASE_DIR, START_YEAR, END_YEAR)

    aep_results = [] # To store AEP results for all years and turbines

    # Initialize turbine models once
    nrel_5mw_turbine = NREL5MWWindTurbine(input_dir=INPUT_DIR)
    nrel_15mw_turbine = NREL15MWWindTurbine(input_dir=INPUT_DIR)

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"\n--- Processing Year: {year} ---")
        # Load and parse multiple provided netCDF4 files for the current year
        print(f"Loading wind data for {year}...")
        # Load data only for the current year to ensure Weibull fit is specific to that year
        combined_dataset_year = load_wind_data(start_year=year, end_year=year, input_dir=INPUT_DIR)

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
            plot_title_weibull = f'Wind Speed Distribution at {TARGET_LATITUDE}°N, {TARGET_LONGITUDE}°E, {TARGET_HEIGHT}m ({year})'
            weibull_plot_filename = f'wind_speed_distribution_{TARGET_HEIGHT}m_{year}.png'
            weibull_save_path = os.path.join(OUTPUT_BASE_DIR, str(year), weibull_plot_filename)
            plot_wind_speed_distribution(wind_speed_series, k_weibull, A_weibull, plot_title_weibull, weibull_save_path)

            # 7. Plot wind rose diagram
            plot_title_windrose = f'Wind Rose at {TARGET_LATITUDE}°N, {TARGET_LONGITUDE}°E, {TARGET_HEIGHT}m ({year})'
            windrose_plot_filename = f'wind_rose_{TARGET_HEIGHT}m_{year}.png'
            windrose_save_path = os.path.join(OUTPUT_BASE_DIR, str(year), windrose_plot_filename)
            plot_wind_rose(wind_speed_series, wind_direction_series, plot_title_windrose, windrose_save_path)

            # 8. Compute AEP for both turbines
            print(f"Calculating AEP for {year}...")
            aep_5mw = calculate_aep(k_weibull, A_weibull, nrel_5mw_turbine)
            aep_15mw = calculate_aep(k_weibull, A_weibull, nrel_15mw_turbine)
            
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
        print(f"\nAll AEP results saved to {aep_output_path}")
        print("\nAEP Summary:")
        print(aep_df)
    else:
        print("No AEP results to save.")

    print("\nOverall analysis complete.")

if __name__ == "__main__":
    run_analysis()   
