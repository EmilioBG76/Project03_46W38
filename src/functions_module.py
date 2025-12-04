# This module will contain the functions for wind data processing.

import xarray as xr
import os
import numpy as np
import pandas as pd
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
import seaborn as sns
# Import WindroseAxes for dedicated wind rose plotting
from windrose import WindroseAxes

# Define Wind Turbine classes
class WindTurbine:
    """
    Base class for wind turbines.
    """
    def __init__(self, name, hub_height, power_curve_filepath, input_dir='/Users/cinnamon/Downloads/Project03_46W38/inputs'):
        self.name = name
        self.hub_height = hub_height
        self.power_curve_filepath = os.path.join(input_dir, power_curve_filepath)
        self.power_curve = self._load_power_curve()
        self.cut_in_speed = self.power_curve['wind_speed'].min() if not self.power_curve.empty else None
        self.cut_out_speed = self.power_curve['wind_speed'].max() if not self.power_curve.empty else None
        self.rated_power = self.power_curve['power'].max() if not self.power_curve.empty else None
# Define method to load power curve
    def _load_power_curve(self):
        """
        Loads the turbine's power curve from a CSV file.
        """
        if not os.path.exists(self.power_curve_filepath):
            raise FileNotFoundError(f"Power curve file not found: {self.power_curve_filepath}")
        # Assuming the CSV has columns 'wind_speed' and 'power' and potentially an extra index column
        power_curve_df = pd.read_csv(self.power_curve_filepath, usecols=[0, 1]) # Explicitly select first two columns
        power_curve_df.columns = ['wind_speed', 'power'] # Ensure column names are consistent
        return power_curve_df
# Define method to get power output
    def get_power_output(self, wind_speed):
        """
        Calculates power output for a given wind speed using linear interpolation.
        """
        if not self.power_curve.empty:
            # Interpolate power output for the given wind speed
            power_output = np.interp(wind_speed,
                                    self.power_curve['wind_speed'],
                                    self.power_curve['power'],
                                    left=0, right=0) # Power is 0 outside cut-in/cut-out
            return power_output
        return 0
# Define specific NREL turbine models
# Define NREL 5 MW turbine class
class NREL5MWWindTurbine(WindTurbine):
    """
    Represents the NREL 5 MW reference wind turbine.
    Hub height: 90 m
    Power curve: NREL_Reference_5MW_126.csv
    """
    def __init__(self, input_dir='/Users/cinnamon/Downloads/Project03_46W38/inputs'):
        super().__init__(
            name='NREL 5 MW',
            hub_height=90,
            power_curve_filepath='NREL_Reference_5MW_126.csv',
            input_dir=input_dir
        )
# Define NREL 15 MW turbine class
class NREL15MWWindTurbine(WindTurbine):
    """
    Represents the NREL 15 MW reference wind turbine.
    Hub height: 150 m
    Power curve: NREL_Reference_15MW_240.csv
    """
    def __init__(self, input_dir='/Users/cinnamon/Downloads/Project03_46W38/inputs'):
        super().__init__(
            name='NREL 15 MW',
            hub_height=150,
            power_curve_filepath='NREL_Reference_15MW_240.csv',
            input_dir=input_dir
        )
# Define WindResource class
class WindResource:
    """
    Manages wind resource data for a specific location and height.
    Handles spatial interpolation and vertical extrapolation.
    """
    def __init__(self, dataset, latitude, longitude):
        self.dataset = dataset
        self.latitude = latitude
        self.longitude = longitude
        
        # Interpolate u and v components at 10m and 100m for the given lat/lon
        self.u10_interp = self.dataset['u10'].interp(latitude=latitude, longitude=longitude, method='linear')
        self.v10_interp = self.dataset['v10'].interp(latitude=latitude, longitude=longitude, method='linear')
        self.u100_interp = self.dataset['u100'].interp(latitude=latitude, longitude=longitude, method='linear')
        self.v100_interp = self.dataset['v100'].interp(latitude=latitude, longitude=longitude, method='linear')
# Define method to get wind data at target height
    def get_wind_data_at_height(self, target_height):
        """
        Computes wind speed and direction time series at a target height.
        Uses interpolation for 10m and 100m data, and power law for extrapolation.

        Args:
            target_height (float): The desired height in meters.

        Returns:
            tuple: (wind_speed_series, wind_direction_series) at the target height.
        """
        if target_height == 10:
            wind_speed, wind_direction = compute_wind_speed_direction(self.u10_interp, self.v10_interp)
        elif target_height == 100:
            wind_speed, wind_direction = compute_wind_speed_direction(self.u100_interp, self.v100_interp)
        elif target_height > 10 and target_height < 100:
            # Linear interpolation for heights between 10m and 100m
            u_component = self.u10_interp + (self.u100_interp - self.u10_interp) * ((target_height - 10) / (100 - 10))
            v_component = self.v10_interp + (self.v100_interp - self.v10_interp) * ((target_height - 10) / (100 - 10))
            wind_speed, wind_direction = compute_wind_speed_direction(u_component, v_component)
        else:
            # Extrapolate using power law profile from 100m data
            # Assume alpha = 0.14 as a typical value for offshore wind, if not specified
            # More advanced usage could involve calculating alpha from 10m and 100m data
            # For now, we'll use 100m as the reference height for extrapolation above 100m
            alpha = 0.14 # A common default for power law exponent
            
            # First, get wind speed at 100m
            ws_100, _ = compute_wind_speed_direction(self.u100_interp, self.v100_interp)
            
            # Apply power law: u(z) = u(zr) * (z/zr)^alpha
            wind_speed = ws_100 * (target_height / 100)**alpha
            
            # For wind direction, assume it's the same as 100m for heights around/above 100m
            _, wind_direction = compute_wind_speed_direction(self.u100_interp, self.v100_interp)
            
        return wind_speed, wind_direction
# Define method to fit Weibull distribution
    def fit_weibull_distribution(self, target_height):
        """
        Fits a Weibull distribution to the wind speed data at the specified height
        for the location managed by this WindResource instance.

        Args:
            target_height (float): The height at which to fit the Weibull distribution.

        Returns:
            tuple: (k, A) - Weibull shape and scale parameters. Returns (None, None) if fitting fails.
        """
        wind_speed_series, _ = self.get_wind_data_at_height(target_height)
        k, A = fit_weibull_parameters(wind_speed_series)
        return k, A

# Mapping of file ranges to filenames
file_ranges = {
    (1997, 1999): "1997-1999.nc",
    (2000, 2002): "2000-2002.nc",
    (2003, 2005): "2003-2005.nc",
    (2006, 2008): "2006-2008.nc"
}
# Function to load a single NetCDF file
def load_netcdf_file(filepath):
    """
    Loads a single NetCDF4 file into an xarray Dataset.
    Args:
        filepath (str): The path to the NetCDF4 file.
    Returns:
        xr.Dataset: The loaded xarray Dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return xr.open_dataset(filepath, engine='netcdf4')

# Function to load and concatenate multiple NetCDF files based on year range
def load_wind_data(start_year=1997, end_year=2008, input_dir='/Users/cinnamon/Downloads/Project03_46W38/inputs'):
    """
    Loads and concatenates wind data from multiple NetCDF4 files within a specified year range.

    Args:
        start_year (int): The starting year for data loading (inclusive).
        end_year (int): The ending year for data loading (inclusive).
        input_dir (str): The directory where NetCDF4 files are located.

    Returns:
        xr.Dataset: A concatenated xarray Dataset containing wind data for the specified period.
    """
    all_datasets = []
    # NetCDF files are named 'XXXX-YYYY.nc', meaning data from year XXXX to YYYY
    # The provided files are: 1997-1999.nc, 2000-2002.nc, 2003-2005.nc, 2006-2008.nc
    
    files_to_load = []
    for (f_start, f_end), filename in file_ranges.items():
        # Check if the file's range overlaps with the requested range
        if max(start_year, f_start) <= min(end_year, f_end):
            files_to_load.append(os.path.join(input_dir, filename))

    if not files_to_load:
        print(f"No NetCDF files found for the years {start_year}-{end_year} in {input_dir}")
        return None

    for file_path in files_to_load:
        print(f"Loading data from: {file_path}")
        try:
            ds = xr.open_dataset(file_path, engine='netcdf4')
            all_datasets.append(ds)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if all_datasets:
        # Concatenate along the 'time' dimension
        combined_dataset = xr.concat(all_datasets, dim='time')
        # Filter by requested years if the loaded files contain data outside the range
        combined_dataset = combined_dataset.sel(time=slice(str(start_year), str(end_year)))
        return combined_dataset
    else:
        return None
# Define functions for wind data processing related to wind speed and direction
def compute_wind_speed_direction(u_component, v_component):
    """
    Computes wind speed and wind direction from u and v components.

    Args:
        u_component (xarray.DataArray or numpy.ndarray): The u-component (eastward) of wind speed.
        v_component (xarray.DataArray or numpy.ndarray): The v-component (northward) of wind speed.

    Returns:
        tuple: A tuple containing:
            - wind_speed (xarray.DataArray or numpy.ndarray): Computed wind speed in m/s.
            - wind_direction (xarray.DataArray or numpy.ndarray): Computed wind direction in degrees (0-360, standard meteorological convention: direction FROM which).
    """
    wind_speed = np.sqrt(u_component**2 + v_component**2)

    wind_direction_rad = np.arctan2(u_component, v_component)
    wind_direction_deg = np.degrees(wind_direction_rad)

    # Ensures values are 0 to 360 (meteorological convention: direction FROM which)
    # np.arctan2 returns values in [-pi, pi], which is [-180, 180] degrees.
    # Adding 180 shifts it to [0, 360], then % 360 handles any negative results correctly.
    wind_direction = (wind_direction_deg + 180 + 360) % 360
    
    # Handle cases where wind speed is exactly zero. Conventionally, direction is 0 for zero wind.
    # Use np.where to apply this condition element-wise without using .item().
    wind_direction = np.where(wind_speed == 0, 0.0, wind_direction)
    
    return wind_speed, wind_direction
# Function to fit Weibull distribution and return parameters
def fit_weibull_parameters(wind_speed_data):
    """
    Fits a Weibull distribution to wind speed data and returns the shape (k) and scale (A) parameters.

    Args:
        wind_speed_data (numpy.ndarray or xarray.DataArray): Time series of wind speed.

    Returns:
        tuple: (k, A) - Weibull shape and scale parameters. Returns (None, None) if fitting fails.
    """
    speeds = wind_speed_data.values if hasattr(wind_speed_data, 'values') else np.asarray(wind_speed_data)
    speeds = speeds[speeds >= 0] # Ensure data is non-negative

    if len(speeds) < 2:
        print("Not enough data points to fit Weibull distribution.")
        return None, None

    try:
        # Fit Weibull_min distribution (2-parameter Weibull), fixing location (loc) to 0.
        # weibull_min.fit returns (shape, loc, scale)
        shape, loc, scale = scipy.stats.weibull_min.fit(speeds, floc=0)
        return shape, scale
    except Exception as e:
        print(f"Error fitting Weibull distribution: {e}")
        return None, None
# Function to create yearly output directories
def create_yearly_output_directories(base_output_dir, start_year, end_year):
    """
    Creates yearly subdirectories within a base output directory.

    Args:
        base_output_dir (str): The main output directory (e.g., '/Users/cinnamon/Downloads/Project03_46W38/outputs').
        start_year (int): The starting year for which to create a subdirectory.
        end_year (int): The ending year for which to create a subdirectory.
    """
    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(base_output_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        print(f"Created output directory: {year_dir}")
# Function to plot wind speed distribution with fitted Weibull
def plot_wind_speed_distribution(wind_speed_data, k, A, title, save_path):
    """
    Plots the histogram of wind speed data and overlays the fitted Weibull distribution.

    Args:
        wind_speed_data (numpy.ndarray or xarray.DataArray): Time series of wind speed.
        k (float): Weibull shape parameter.
        A (float): Weibull scale parameter.
        title (str): Title for the plot.
        save_path (str): Full path to save the plot image (e.g., /content/outputs/2000/weibull_plot.png').
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(wind_speed_data, bins=30, kde=False, stat='density', color='skyblue', label='Wind Speed Histogram')

    # Plot fitted Weibull PDF
    if k is not None and A is not None:
        # Extract scalar value from xarray DataArray for np.linspace to avoid dimension mismatch
        # Using .max().item() is appropriate here as it reduces the DataArray to a single max scalar value.
        x = np.linspace(0, wind_speed_data.max().item(), 100)
        pdf = scipy.stats.weibull_min.pdf(x, k, loc=0, scale=A)
        plt.plot(x, pdf, color='red', linewidth=2, label=f'Fitted Weibull (k={k:.2f}, A={A:.2f})')

    plt.title(title)
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"Saved wind speed distribution plot to {save_path}")
# Function to plot wind rose diagram
def plot_wind_rose(wind_speed_data, wind_direction_data, title, save_path, 
                bins=np.arange(0, 20, 2), 
                direction_bins=16): # Added direction_bins for windrose
    """
    Plots a wind rose diagram using the windrose library.

    Args:
        wind_speed_data (numpy.ndarray or xarray.DataArray): Time series of wind speed.
        wind_direction_data (numpy.ndarray or xarray.DataArray): Time series of wind direction in degrees.
        title (str): Title for the plot.
        save_path (str): Full path to save the plot image.
        bins (array-like): Bins for wind speed categories.
        direction_bins (int): Number of direction bins for the wind rose (default 16).
    """
    # Convert to numpy arrays if they are xarray DataArrays
    speeds = wind_speed_data.values if hasattr(wind_speed_data, 'values') else np.asarray(wind_speed_data)
    directions = wind_direction_data.values if hasattr(wind_direction_data, 'values') else np.asarray(wind_direction_data)

    # Filter out NaN values
    valid_indices = ~np.isnan(speeds) & ~np.isnan(directions)
    speeds = speeds[valid_indices]
    directions = directions[valid_indices]

    fig = plt.figure(figsize=(10, 10))
    # Create a WindroseAxes object
    ax = WindroseAxes.from_ax(fig=fig)

    # Plot the wind rose
    ax.bar(directions, speeds, 
        normed=True,  # Normalize frequencies
        opening=0.8,  # Width of the bars
        edgecolor='black', 
        bins=bins, 
        nsector=direction_bins) # Number of direction bins
    
    # Set title and legend
    ax.set_title(title, va='bottom', fontsize=16)
    ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1.15, 0.9))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() # Close plot to free memory
    print(f"Saved wind rose diagram to {save_path}")
# Function to calculate Annual Energy Production (AEP)
def calculate_aep(k, A, turbine, efficiency=1.0, hours_in_year=8760):
    """
    Calculates the Annual Energy Production (AEP) for a given turbine and Weibull parameters.

    Args:
        k (float): Weibull shape parameter.
        A (float): Weibull scale parameter.
        turbine (WindTurbine): An instance of a WindTurbine class (e.g., NREL5MWWindTurbine).
        efficiency (float): Turbine availability/efficiency (default 1.0).
        hours_in_year (int): Number of hours in a year (default 8760).

    Returns:
        float: Annual Energy Production (AEP) in MWh.
    """
    if k is None or A is None or not isinstance(turbine, WindTurbine):
        return 0.0

    # Define the Weibull PDF function
    def weibull_pdf(u, k_val, A_val):
        return scipy.stats.weibull_min.pdf(u, k_val, loc=0, scale=A_val)

    # Define the integrand: Power(u) * Weibull_PDF(u)
    def integrand(u, k_val, A_val, turbine_obj):
        power_output_kw = turbine_obj.get_power_output(u) # Power curve often in kW
        return power_output_kw * weibull_pdf(u, k_val, A_val)

    # Perform the numerical integration
    # Integral limits are the turbine's cut-in and cut-out speeds
    lower_bound = turbine.cut_in_speed if turbine.cut_in_speed is not None else 0
    upper_bound = turbine.cut_out_speed if turbine.cut_out_speed is not None else 25 # A reasonable upper limit if cut-out is missing

    integral_result, _ = scipy.integrate.quad(integrand, lower_bound, upper_bound, args=(k, A, turbine))

    # AEP = efficiency * hours_in_year * integral_result (which is in kW)
    # Convert kW to MW (divide by 1000)
    aep_mwh = efficiency * hours_in_year * (integral_result / 1000)
    
    return aep_mwh




