# src/functions_module.py
# This module contains classes and functions for wind resource analysis,
# including data loading, wind resource management, turbine modeling,
# Weibull fitting, AEP calculation, and plotting.
# Imports
import xarray as xr               # For handling NetCDF files and multi-dimensional arrays
import os                         # For file path manipulations
import numpy as np                # For numerical computations
import pandas as pd               # For data handling and saving results
import scipy.stats                # For statistical distributions and fitting
import scipy.integrate            # For numerical integration
import matplotlib.pyplot as plt   # For plotting
import seaborn as sns             # For enhanced statistical plotting
# Import WindroseAxes for dedicated wind rose plotting
from windrose import WindroseAxes # For wind rose diagrams

# Define Wind Turbine classes
class WindTurbine:
    """Base class for wind turbines."""
    def __init__(self, name, hub_height, power_curve_filepath, input_dir='/Users/cinnamon/Downloads/Project03_46W38/inputs'):
        self.name = name
        self.hub_height = hub_height
        self.power_curve_filepath = os.path.join(input_dir, power_curve_filepath)
        self.power_curve = self._load_power_curve()
        # Determine cut-in and cut-out speeds more accurately
        if not self.power_curve.empty:
            # Find cut-in speed: first wind speed where power output is greater than 0
            power_on_indices = self.power_curve[self.power_curve['power'] > 0]['wind_speed'].index
            if not power_on_indices.empty:
                self.cut_in_speed = self.power_curve.loc[power_on_indices[0], 'wind_speed']
            else:
                self.cut_in_speed = None # No power ever produced

            # Find cut-out speed: last wind speed where power output is greater than 0
            power_off_indices = self.power_curve[self.power_curve['power'] > 0].index
            if not power_off_indices.empty:
                self.cut_out_speed = self.power_curve.loc[power_off_indices[-1], 'wind_speed']
            else:
                self.cut_out_speed = None # No power ever produced
        else:
            self.cut_in_speed = None
            self.cut_out_speed = None
        self.rated_power = self.power_curve['power'].max() if not self.power_curve.empty else None
# Load power curve from CSV
    def _load_power_curve(self):
        """Loads the turbine's power curve from a CSV file."""
        if not os.path.exists(self.power_curve_filepath):
            raise FileNotFoundError(f"Power curve file not found: {self.power_curve_filepath}")
        # Assuming the CSV has columns 'wind_speed' and 'power' and potentially an extra index column
        power_curve_df = pd.read_csv(self.power_curve_filepath, usecols=[0, 1]) # Explicitly select first two columns
        power_curve_df.columns = ['wind_speed', 'power'] # Ensure column names are consistent
        return power_curve_df
# Calculate power output for a given wind speed
    def get_power_output(self, wind_speed):
        """Calculates power output for a given wind speed using linear interpolation."""
        if not self.power_curve.empty:
            # Interpolate power output for the given wind speed
            power_output = np.interp(wind_speed,
                                    self.power_curve['wind_speed'],
                                    self.power_curve['power'],
                                    left=0, right=0) # Power is 0 outside cut-in/cut-out
            return power_output
        return 0
# Calculate Annual Energy Production (AEP)
    def calculate_aep(self, k, A, efficiency=1.0, hours_in_year=8760):
        """Calculates the Annual Energy Production (AEP) for a given turbine and Weibull parameters.

        Args:
            k (float): Weibull shape parameter.
            A (float): Weibull scale parameter.
            efficiency (float): Turbine availability/efficiency (default 1.0).
            hours_in_year (int): Number of hours in a year (default 8760).

        Returns:
            float: Annual Energy Production (AEP) in MWh.
        """
        if k is None or A is None:
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
        lower_bound = self.cut_in_speed if self.cut_in_speed is not None else 0
        upper_bound = self.cut_out_speed if self.cut_out_speed is not None else 25 # A reasonable upper limit if cut-out is missing

        integral_result, _ = scipy.integrate.quad(integrand, lower_bound, upper_bound, args=(k, A, self))

        # AEP = efficiency * hours_in_year * integral_result (which is in kW)
        # Convert kW to MW (divide by 1000)
        aep_mwh = efficiency * hours_in_year * (integral_result / 1000)

        return aep_mwh

# Define specific NREL turbine models
# Define NREL 5 MW and 15 MW turbine classes
class NREL5MWWindTurbine(WindTurbine):
    """Represents the NREL 5 MW reference wind turbine.
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

class NREL15MWWindTurbine(WindTurbine):
    """Represents the NREL 15 MW reference wind turbine.
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
    """Manages wind resource data for a specific location and height.
    Handles spatial interpolation and vertical extrapolation."""
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
        """Computes wind speed and direction time series at a target height.
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
        """Fits a Weibull distribution to the wind speed data at the specified height
        for the location managed by this WindResource instance.

        Args:
            target_height (float): The height at which to fit the Weibull distribution.

        Returns:
            tuple: (k, A) - Weibull shape and scale parameters. Returns (None, None) if fitting fails.
        """
        wind_speed_series, _ = self.get_wind_data_at_height(target_height)
        k, A = fit_weibull_parameters(wind_speed_series)
        return k, A

# Define WindDataLoader class
class WindDataLoader:
    """Encapsulates logic for loading and managing wind data from NetCDF files."""
    def __init__(self, input_dir='/content/inputs'):
        self.input_dir = input_dir
        self.file_ranges = {
            (1997, 1999): "1997-1999.nc",
            (2000, 2002): "2000-2002.nc",
            (2003, 2005): "2003-2005.nc",
            (2006, 2008): "2006-2008.nc"
        }
# Define method to load a single NetCDF file
    def _load_netcdf_file(self, filepath):
        """Loads a single NetCDF4 file into an xarray Dataset.
        Args:
            filepath (str): The path to the NetCDF4 file.
        Returns:
            xr.Dataset: The loaded xarray Dataset.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return xr.open_dataset(filepath, engine='netcdf4')
# Define method to load data for a specific year
    def load_data_for_year(self, year):
        """Loads and concatenates wind data from relevant NetCDF4 files for a specific year.

        Args:
            year (int): The year for which to load data.

        Returns:
            xr.Dataset: An xarray Dataset containing wind data for the specified year.
        """
        all_datasets = []
        files_to_load = []

        for (f_start, f_end), filename in self.file_ranges.items():
            if f_start <= year <= f_end:
                files_to_load.append(os.path.join(self.input_dir, filename))

        if not files_to_load:
            print(f"No NetCDF files found for the year {year} in {self.input_dir}")
            return None

        for file_path in files_to_load:
            print(f"Loading data from: {file_path}")
            try:
                ds = self._load_netcdf_file(file_path)
                all_datasets.append(ds)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if all_datasets:
            combined_dataset = xr.concat(all_datasets, dim='time')
            # Filter by the exact year requested
            combined_dataset = combined_dataset.sel(time=slice(str(year), str(year)))
            return combined_dataset
        else:
            return None

# Define WindAnalysisPlotter class
class WindAnalysisPlotter:
    """Encapsulates logic for creating yearly output directories and generating wind analysis plots."""
    def __init__(self, output_dir='/Users/cinnamon/Downloads/Project03_46W38/outputs'):
        self.output_dir = output_dir

    def create_yearly_directories(self, start_year, end_year):
        """Creates yearly subdirectories within the base output directory.

        Args:
            start_year (int): The starting year for which to create a subdirectory.
            end_year (int): The ending year for which to create a subdirectory.
        """
        for year in range(start_year, end_year + 1):
            year_dir = os.path.join(self.output_dir, str(year))
            os.makedirs(year_dir, exist_ok=True)
            print(f"Created output directory: {year_dir}")
# Define method to plot wind speed distribution
    def plot_speed_distribution(self, wind_speed_data, k, A, year, height):
        """Plots the histogram of wind speed data and overlays the fitted Weibull distribution.

        Args:
            wind_speed_data (numpy.ndarray or xarray.DataArray): Time series of wind speed.
            k (float): Weibull shape parameter.
            A (float): Weibull scale parameter.
            year (int): The year of the data.
            height (int): The height of the wind speed data.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(wind_speed_data, bins=30, kde=False, stat='density', color='skyblue', label='Wind Speed Histogram')

        # Plot fitted Weibull PDF
        if k is not None and A is not None:
            x = np.linspace(0, wind_speed_data.max().item(), 100)
            pdf = scipy.stats.weibull_min.pdf(x, k, loc=0, scale=A)
            plt.plot(x, pdf, color='red', linewidth=2, label=f'Fitted Weibull (k={k:.2f}, A={A:.2f})')

        plt.title(f'Wind Speed Distribution at {height}m for {year}')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, str(year), f'wind_speed_distribution_{height}m_{year}.png')
        plt.savefig(save_path)
        plt.close() # Close plot to free memory
        print(f"Saved wind speed distribution plot to {save_path}")
# Define method to plot wind rose
    def plot_wind_rose(self, wind_speed_data, wind_direction_data, year, height,
                        bins=np.arange(0, 20, 2),
                        direction_bins=16):
        """Plots a wind rose diagram using the windrose library.

        Args:
            wind_speed_data (numpy.ndarray or xarray.DataArray): Time series of wind speed.
            wind_direction_data (numpy.ndarray or xarray.DataArray): Time series of wind direction in degrees.
            year (int): The year of the data.
            height (int): The height of the wind speed data.
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
        ax.set_title(f'Wind Rose at {height}m for {year}', va='bottom', fontsize=16)
        ax.set_legend(title='Wind Speed (m/s)', bbox_to_anchor=(1.15, 0.9))

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, str(year), f'wind_rose_{height}m_{year}.png')
        plt.savefig(save_path)
        plt.close() # Close plot to free memory
        print(f"Saved wind rose diagram to {save_path}")


# Define functions for wind data processing related to wind speed and direction
def compute_wind_speed_direction(u_component, v_component):
    """Computes wind speed and wind direction from u and v components.

    Args:
        u_component (xarray.DataArray or numpy.ndarray): The u-component (eastward) of wind speed.
        v_component (xarray.DataArray or numpy.ndarray): The v-component (northward) of wind speed.

    Returns:
        tuple: A tuple containing:
            - wind_speed (xarray.DataArray or numpy.ndarray): Computed wind speed in m/s.
            - wind_direction (xarray.DataArray or numpy.ndarray): Computed wind direction in degrees (0-360, standard meteorological convention: direction FROM which).
    """
    wind_speed = np.sqrt(u_component**2 + v_component**2)

    # Convert xarray DataArrays to numpy arrays for explicit calculation, if they are DataArrays
    u_vals = u_component.values if isinstance(u_component, xr.DataArray) else u_component
    v_vals = v_component.values if isinstance(v_component, xr.DataArray) else v_component
    wind_speed_vals = wind_speed.values if isinstance(wind_speed, xr.DataArray) else wind_speed

    wind_direction_rad = np.arctan2(u_vals, v_vals)
    wind_direction_deg = np.degrees(wind_direction_rad)

    # Compute direction and handle the 0-360 range
    calculated_wind_direction = (wind_direction_deg + 180 + 360) % 360

    # Apply condition for zero wind speed
    # This ensures 'wind_direction' is always associated with a value before being used here.
    final_wind_direction = np.where(wind_speed_vals == 0, 0.0, calculated_wind_direction)

    # If original components were xarray DataArrays, ensure the output direction is also a DataArray
    if isinstance(u_component, xr.DataArray):
        # Create a new DataArray with the same coordinates and dimensions as the original inputs
        final_wind_direction = xr.DataArray(final_wind_direction,
                                            coords=wind_speed.coords,
                                            dims=wind_speed.dims)
# Return wind speed and direction
    return wind_speed, final_wind_direction

# Function to fit Weibull distribution and return parameters
def fit_weibull_parameters(wind_speed_data):
    """Fits a Weibull distribution to wind speed data and returns the shape (k) and scale (A) parameters.

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

    # Explicitly handle the case where all wind speeds are zero
    if np.all(speeds == 0):
        return None, None

    try:
        # Fit Weibull_min distribution (2-parameter Weibull), fixing location (loc) to 0.
        # weibull_min.fit returns (shape, loc, scale)
        shape, loc, scale = scipy.stats.weibull_min.fit(speeds, floc=0)
        return shape, scale
    except Exception as e:
        print(f"Error fitting Weibull distribution: {e}")
        return None, None
