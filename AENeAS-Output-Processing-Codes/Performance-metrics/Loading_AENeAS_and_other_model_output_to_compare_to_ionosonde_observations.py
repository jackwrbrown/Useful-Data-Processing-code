# Import key modules that will be used throughout loading in and analysing AENeAS output and comparing to data and different models
from glob import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
from datetime import datetime, timedelta
import iri2020 # This one requires a bit extra work

# Load in all the AENeAS output datasets that have been loaded previously

# Path to the directory with the .pkl files
directory = "/rds/projects/t/themendr-j-brown/aeneas_2.0/July2017Storm_Ionosonde_data/global July 2017 AENeASoutput dataframes/" 

# Get a list of all .pkl files that contain "df_summary"
pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl") and "df_summary" in f]

# Loop through and load each one
for file in pkl_files:
    file_path = os.path.join(directory, file)
    # Strip off the ".pkl"
    base = file.replace(".pkl", "")
    # Remove leading digits and the following underscore
    var_name = re.sub(r"^[0-9]+_", "", base)
    # Remove any parentheses (e.g. the "(ensemble_collapsed)" suffix becomes "_ensemble_collapsed")
    var_name = var_name.replace("(", "_").replace(")", "")
    # (Optional) collapse any double-underscores
    var_name = re.sub(r"__+", "_", var_name)
    # Load and assign
    globals()[var_name] = pd.read_pickle(file_path)
    print(f"Loaded {file} → `{var_name}`")

====================================================================================================================================================================
# Make dataset groups of AENeAS runs for comparison
====================================================================================================================================================================

# Make groups of datasets so that functions later on can be called easily for each dataset and comparison is easier
all_datasets = {
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2": July2017_cov_1_5_F107_6_scale_3x_std_dev_Ti_Te_0_2_df_summary,
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2, O1 N2 = 0.05": July2017_cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_O1_N2_0_05_df_summary,
    "Factory Settings": July2017_factory_setting_df_summary,
    "Factory Settings, No Assim": July2017_factory_no_data_assim_df_summary,
    "Cov 3": July2017_cov_3_df_summary,
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, O1 N2 = 0.1": July2017_cov_1_5_F107_6_scale_3x_std_dev_O1_N2_0_1_df_summary,
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2, poten = 0.1": July2017_cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_summary,
}

optimal_datasets = {
    "AENeAS FS": all_datasets["Factory Settings"],
    "AENeAS FS (No DA)": all_datasets["Factory Settings, No Assim"],
    "AENeAS Optimised": all_datasets["Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2"],
    "AENeAS Optimised, poten = 0.1": all_datasets["Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2, poten = 0.1"],
}

test_datasets = {
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2": July2017_cov_1_5_F107_6_scale_3x_std_dev_Ti_Te_0_2_df_summary,
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2, O1 N2 = 0.05": all_datasets["Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2, O1 N2 = 0.05"],
}
    
dataset_colors = {
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2": "cyan",
    "AENeAS Optimised": "cyan",
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, Ti Te = 0.2, O1 N2 = 0.05": "cyan",
    "Factory Settings": "blue",
    "AENeAS FS": "indigo",
    "Factory Settings, No Assim": "lightblue",
    "AENeAS FS (No DA)": "lightblue",
    "Cov 3": "orange",
    "Cov 1.5, F10.7 6 scale, 3x Std Dev, O1 N2 = 0.1": "purple",
}

# Make dataset groups dictionary
dataset_groups = {
    "all_datasets": all_datasets,
    "optimal_datasets": optimal_datasets,
    "test_datasets": test_datasets,
}
====================================================================================================================================================================
# Set hmF2 values of 180 km to NaN in AENeAS dataframes as this is the min searching value when loading the data
====================================================================================================================================================================

# Function to replace 180.0 in hmF2 column with NaN
def set_hmF2_to_nan(df):
    df.loc[df["hmF2 (km)"] == 180.0] = np.nan
    return df

# Apply to all datasets
for name, df in all_datasets.items():
    all_datasets[name] = set_hmF2_to_nan(df)

====================================================================================================================================================================
# Load in other model data that had been processed by ECG for comparison to AENeAS
====================================================================================================================================================================
# TIEGCM DATA

# Loading TIEGCM data from ECG
import glob

# Path to the directory of TIEGCM data
data_dir = '/rds/homes/j/jxb1605/themendr-j-brown/aeneas_2.0/July2017Storm_Ionosonde_data/July2017_TIEGCM_station_data/'
csv_files = glob.glob(os.path.join(data_dir, 'Interpolation_Results_*.csv'))

dfs = []

for file in csv_files:
    station_name = os.path.basename(file).replace('Interpolation_Results_', '').replace('.csv', '').upper()
    df = pd.read_csv(file)
    df['station'] = station_name
    df.columns = df.columns.str.strip()
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Remove brackets and convert to float
    for col in ['hmf2 (km)', 'nmf2 (e/m^3)']:
        df[col] = df[col].astype(str).str.strip("[]").astype(float)

    # Calculate foF2 (MHz) from NmF2
    df['foF2 (MHz)'] = (df['nmf2 (e/m^3)'] / 1.24e10) ** 0.5

    # Reorder columns
    df = df[['DateTime', 'hmf2 (km)', 'nmf2 (e/m^3)', 'foF2 (MHz)', 'station']]
    dfs.append(df)

combined_tiegcm_df = pd.concat(dfs, ignore_index=True)

# WACCM-X DATA

# Directory path to WACCM-X data
data_dir = '/rds/homes/j/jxb1605/themendr-j-brown/aeneas_2.0/July2017Storm_Ionosonde_data/2017_WACCMX_station_data/'

# Find WACCMX files
csv_files = glob.glob(os.path.join(data_dir, 'WACCMX_*_2017.csv'))

# Define date range to keep
start_date = pd.Timestamp('2017-07-14')
end_date = pd.Timestamp('2017-07-18 00:00:00')  # include end of the 18th

dfs = []

for file in csv_files:
    # Extract station name (middle part)
    filename = os.path.basename(file)
    parts = filename.split('_')
    station_name = parts[1]

    # Load CSV
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    # Add station name
    df['station'] = station_name

    # Convert DateTime column
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Filter to date range
    df = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date)]

    # Remove brackets and convert to float
    for col in ['hmf2 (km)', 'nmf2 (e/m^3)', 'foF2 (MHz)']:
        df[col] = df[col].astype(str).str.strip("[]").astype(float)

    # Select columns
    df = df[['DateTime', 'hmf2 (km)', 'nmf2 (e/m^3)', 'foF2 (MHz)', 'station']]
    dfs.append(df)

# Combine all into one DataFrame
combined_waccmx_df = pd.concat(dfs, ignore_index=True)

# Set hmF2 values above 500 km to NaN
combined_waccmx_df.loc[combined_waccmx_df["hmf2 (km)"] > 500.0] = np.nan

# NeQUICK DATA
# The NeQuick version used in AENeAS can be found in the AENeAS files nequick.py
# ATM it's the 2UoB version but might change

# Add AENeAS tools to the path
sys.path.append('/rds/projects/t/themendr-j-brown/aeneas_2.0/aeneas')
from aeneas.nequick import NeQuick

# Need to change these for the dates being looked at
# Taken from indexdb.nc file
f107_by_date = {
    '2017-07-14': 93.9,
    '2017-07-15': 91.6,
    '2017-07-16': 86.5,
    '2017-07-17': 85.6,
    '2017-07-18': 78.2,
}

# Change for the stations you're looking at
ionosonde_stations = [
    {"name": "ALPENA", "lat": 45.07, "lon": -83.56},
    {"name": "EIELSON", "lat": 64.66, "lon": -147.07},
    {"name": "FAIRFORD", "lat": 51.70, "lon": -1.50},
    {"name": "IDAHO", "lat": 43.81, "lon": -112.68},
    {"name": "JULIUSRUH", "lat": 54.60, "lon": 13.40},
    {"name": "MOSCOW", "lat": 55.49, "lon": 37.29},
    {"name": "TROMSO", "lat": 69.60, "lon": 19.20},
]

# Change depending on period being looked at
mth = 7
version = '2UoB'
grid = True
h = np.linspace(100, 600, num=101)  # heights in km
UT_times = np.arange(0, 24, 0.25)  # every 15 minutes to match AENeAS

results = {
    'DateTime': [],
    'foF2 (MHz)': [],
    'hmf2 (km)': [],
    'station': []
}

for station in ionosonde_stations:
    along = station["lon"]
    alat = station["lat"]
    station_name = station["name"]
    
    for day_str, flx in f107_by_date.items():
        day_int = int(day_str[-2:])
        base_date = datetime.strptime(day_str, "%Y-%m-%d")
        
        for UT in UT_times:
            neq = NeQuick(along, alat, h, mth, flx, UT,
                          grid=grid, version=version, day=day_int,
                          hdf='/rds/homes/j/jxb1605/themendr-j-brown/aeneas_2.0/aeneas/aeneas/nequickData.h5',
                          botTaper=3,
                          verbose=False)
            
            # Find foF2 and hmF2 directly
            nmf2 = np.max(neq)
            fof2 = (nmf2 / 1.24e10) ** 0.5
            idx_max = np.argmax(neq)
            hmf2 = h[idx_max]
            
            dt = base_date + timedelta(hours=UT)
            
            results['DateTime'].append(dt)
            results['foF2 (MHz)'].append(fof2)
            results['hmf2 (km)'].append(hmf2)
            results['station'].append(station_name)

nequick_df = pd.DataFrame(results)

# IRI DATA

# Define the date range
day_start = datetime(2017, 7, 14)
day_end = datetime(2017, 7, 18)


# Define ionosonde station locations
ionosonde_stations = [
    {"name": "ALPENA", "lat": 45.07, "lon": -83.56},
    {"name": "EIELSON", "lat": 64.66, "lon": -147.07},
    {"name": "FAIRFORD", "lat": 51.70, "lon": -1.50},
    {"name": "IDAHO", "lat": 43.81, "lon": -112.68},
    {"name": "JULIUSRUH", "lat": 54.60, "lon": 13.40},
    {"name": "MOSCOW", "lat": 55.49, "lon": 37.29},
    {"name": "TROMSO", "lat": 69.60, "lon": 19.20},
]

# List to hold all DataFrames
iri_df_list = []

# Loop through each station and compute IRI
for station in ionosonde_stations:
    name = station['name']
    lat = station['lat']
    lon = station['lon'] if station['lon'] >= 0 else station['lon'] + 360  # Convert to [0, 360]

    # Run IRI model
    iri_output = iri2020.timeprofile(
        tlim=(day_start, day_end),
        dt=timedelta(minutes=15),
        altkmrange=[100, 600, 5],
        glat=lat,
        glon=lon
    )

    # Extract values
    times = iri_output['time'].values
    fof2 = iri_output['foF2'].values
    hmf2 = iri_output['hmF2'].values
    nmf2 = iri_output['NmF2'].values  # Already in e/m^3

    # Make DataFrame
    df = pd.DataFrame({
        'DateTime': times,
        'foF2 (MHz)': fof2,
        'hmf2 (km)': hmf2,
        'nmf2 (e/m^3)': nmf2,
        'station': name
    })

    iri_df_list.append(df)

# Combine all into one DataFrame
iri_df = pd.concat(iri_df_list, ignore_index=True)

# E-CHAIM DATA
# You have to run this on MatLab and then save the data into a .csv file and load into Python for analysis
import pandas as pd

# Path to the CSV file
file_path = '/rds/projects/t/themendr-j-brown/E-CHAIM_updated_version/Release_Matlab_CDB-4.2.1/July2017_ECHAIM_fof2_hmf2.csv'

# Load the CSV using default comma separator
echaim_df = pd.read_csv(file_path)

# Convert the 'DateTime' column to datetime objects
echaim_df['DateTime'] = pd.to_datetime(echaim_df['DateTime'], dayfirst=True)

=============================================================================================================================================================
# Load in the ionosonde data to compare to
=============================================================================================================================================================

# Path to the ionosonde data folder
ionosonde_dir = "/rds/projects/t/themendr-j-brown/aeneas_2.0/July2017Storm_Ionosonde_data/"

# Find all .txt files in the folder
ionosonde_files = glob.glob(os.path.join(ionosonde_dir, "*.txt"))

# List to hold all processed DataFrames
all_data = []

# Column names based on your file format
columns = [
    "date", "doy", "time", "C_score", 
    "foF2", "foF1", "foE", 
    "hmF2", "hmF1", "hmE", "scale_F2"
]

# Read each file into a DataFrame
for file_path in ionosonde_files:
    station_name = os.path.splitext(os.path.basename(file_path))[0]  # e.g., "Tromso"
    
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        names=columns,
        header=1,           # Skip the first row (column description)
        na_values=["---"]
    )
    
    # Combine date and time into a datetime column
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S", errors='coerce')
    
    # Drop rows with invalid datetime
    df = df.dropna(subset=["datetime"])
    
    # Add station name as a column
    df["station"] = station_name

    # Reset index (keep datetime as a column)
    df.reset_index(drop=True, inplace=True)
    
    # Append to list
    all_data.append(df)

# Combine all data into a single DataFrame
ionosonde_data_df = pd.concat(all_data, ignore_index=True)

# For the rest of the code to work you need the station names to all be exactly the same so this just forces them to be

# Create a mapping from station name to lat/lon
station_coords = {
    station["name"].upper(): (station["lat"], (station["lon"] + 360) % 360)
    for station in ionosonde_stations
}

# Ensure station names in the dataframe are uppercase for matching
ionosonde_data_df['station'] = ionosonde_data_df['station'].str.upper()

# Replace specific mismatched station names
ionosonde_data_df['station'] = ionosonde_data_df['station'].replace({
    'IDAHO NATIONAL LAB': 'IDAHO'
})

# Standardize 'Station' column in all datasets to uppercase
for df in all_datasets.values():
    df['Station'] = df['Station'].str.upper()

# Add latitude and longitude columns by mapping from station name
ionosonde_data_df['latitude'] = ionosonde_data_df['station'].map(lambda s: station_coords.get(s, (None, None))[0])
ionosonde_data_df['longitude'] = ionosonde_data_df['station'].map(lambda s: station_coords.get(s, (None, None))[1])

