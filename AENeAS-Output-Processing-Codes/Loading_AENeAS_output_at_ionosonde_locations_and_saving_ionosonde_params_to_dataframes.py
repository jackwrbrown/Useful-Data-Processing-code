# Import modules
import sys
import os
import h5py
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# You need to have INSTALLED AENeAS before using this code

# Add AENeAS tools to the path
# These will be used to process and format the output data
# MIGHT NEED CHANGING WITH 2.0
sys.path.append('/rds/projects/t/themendr-j-brown/aeneas_2.0/aeneas') # change to your path where AENeAS tools etc. are stored
from aeneas.denGrid import aeneas2dg, dengrid

# ----------------------------
# Station locations: Examples given here but choose your own that you are comparing to
# Go to GIRO website for station locations: https://giro.uml.edu/ionoweb/
# ----------------------------
ionosonde_stations = [
    {"name": "ALPENA", "lat": 45.07, "lon": -83.56},
    {"name": "EIELSON", "lat": 64.66, "lon": -147.07},
    {"name": "FAIRFORD", "lat": 51.70, "lon": -1.50},
    {"name": "IDAHO", "lat": 43.81, "lon": -112.68},
    {"name": "JULIUSRUH", "lat": 54.60, "lon": 13.40},
    {"name": "MOSCOW", "lat": 55.49, "lon": 37.29},
    {"name": "TROMSO", "lat": 69.60, "lon": 19.20},
]

# ----------------------------
# File locations and time setup
# These are all examples for July 2017 period but change accordingly
# ----------------------------
run_prefix = "july2017storm" 
# Path to the folder containing the AENeAS outputs
base_path = "/rds/projects/t/themendr-j-brown/aeneas_2.0/aeneas" 
# Actual folder containing the outputs
output_base = os.path.join(base_path, run_prefix, "cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1")  # Change this to correct directory

# Start date of the outputs you want
start_date = datetime(2017, 7, 14)
num_days = 4 # number of days of output you want

# Takes regular 15 minute time intervals as these are the output times of AENeAS
times = [f"{hour:02d}{minute:02d}" for hour in range(24) for minute in [0, 15, 30, 45]]

# Choose an altitude range you want to get AENeAS model output for
alt = np.arange(100, 600, 5)  # Altitudes from 100 km to 595 km

# Store the outputs
summary_results = []
profile_records = []

# ----------------------------
# Main processing function
# ----------------------------
def process_source(data, label, alon, alat, aalt, combined_time):
    """Process a single NE dataset for all ionosonde stations."""
    # dengrid is in the AENeAS denGrid.py file
    # it makes a grid of electron densities from AENeAS output files
    electron_densities = dengrid(data, alon, alat, aalt,
                                 name='electron_densities',
                                 longname='Electron density grid',
                                 units='em-3')

    for station in ionosonde_stations:
        station_name = station["name"]
        lon_iono = station["lon"]
        lat_iono = station["lat"]

        try:
            # getColumn is another built in AENeAS function which gets a column of electron densities
            # Good for getting vertical electron density profiles
            ne_profile = [electron_densities.getColumn(lon_iono, lat_iono, h, order=3)[0] for h in alt] # Use order 3 here but can change to 1 or 2
        except Exception as e:
            print(f"⚠️ Could not interpolate for {station_name} at {combined_time}: {e}")
            continue

        ne_profile = np.array(ne_profile)

        for h, ne in zip(alt, ne_profile):
            profile_records.append({
                "Datetime": combined_time,
                "Source": label,
                "Station": station_name,
                "Latitude": lat_iono,
                "Longitude": lon_iono,
                "Altitude (km)": h,
                "Electron Density (el/m³)": ne
            })

        # F2 peak metrics (only consider heights ≥ 180 km)
        alt_threshold = 180
        valid_indices = alt >= alt_threshold
        ne_profile_above_180 = ne_profile[valid_indices]
        alt_above_180 = alt[valid_indices]
        
        if len(ne_profile_above_180) == 0:
            # No data above 180 km, skip
            print(f"  ⚠️ No NE data above {alt_threshold} km at {combined_time} for {label}")
            return
        
        peak_idx = np.argmax(ne_profile_above_180)
        hmF2 = alt_above_180[peak_idx]
        NmF2 = ne_profile_above_180[peak_idx]
        foF2 = (NmF2 / 1.24e10) ** 0.5

        # NEED TO ADD IN: set any values where hmF2 = 180 km to NaN as this is the min height so isn't actually the F2 peak

        summary_results.append({
            "Datetime": combined_time,
            "Source": label,
            "Station": station_name,
            "Latitude": lat_iono,
            "Longitude": lon_iono,
            "hmF2 (km)": hmF2,
            "NmF2 (el/m³)": NmF2,
            "foF2 (MHz)": foF2
        })

# ----------------------------
# Loop through files
# The files should still have the y####/d###/output_files structure as this code loops through those folders
# ----------------------------
for day_offset in range(num_days):
    current_date = start_date + timedelta(days=day_offset)
    year_str = f"y{current_date.year}"
    doy_str = f"d{current_date.timetuple().tm_yday:03d}"
    date_str = current_date.strftime("%Y%m%d")
    folder_path = os.path.join(output_base, year_str, doy_str)

    for time in times:
        file_name = f"AENeAS_{date_str}_{time}.hdf5"
        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Processing {file_name}...")

        try:
            combined_time = datetime.strptime(f"{date_str}_{time}", "%Y%m%d_%H%M")

            # THIS BIT MIGHT NEED CHANGING WITH 2.0 
            # This gets the analysis mean data
            analysis_mean = aeneas2dg(file_path, "NE").dengrid
            alon = aeneas2dg(file_path, "longitude").dengrid
            alat = aeneas2dg(file_path, "latitude").dengrid
            aalt = aeneas2dg(file_path, "ZG").dengrid

            # This gets the background mean electron density output and analysis standard deviation between ensemble members
            with h5py.File(file_path, "r") as f:
                background_mean = f["background/mean/NE"][:]
                analysis_std = f["analysis/std/NE"][:]

            process_source(analysis_mean, "Analysis Mean", alon, alat, aalt, combined_time)
            process_source(background_mean, "Background Mean", alon, alat, aalt, combined_time)
            process_source(analysis_std, "Analysis Std", alon, alat, aalt, combined_time)

            This processes all the individual ensemble members so they can be plotted etc in the future as well
            with h5py.File(file_path, "r") as f:
                for member_index in range(32):
                    member_path = f"analysis/member{member_index}/grids/NE"
                    if member_path in f:
                        member_ne = f[member_path][:]
                        label = f"Member {member_index}"
                        process_source(member_ne, label, alon, alat, aalt, combined_time)
                    else:
                        print(f"  ⚠️ {member_path} not found in {file_name}")

        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")

# ----------------------------
# Create output DataFrames
# Example names are given here but can save them into summary and profile dataframes
# ----------------------------
cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_summary = pd.DataFrame(summary_results)
cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_profiles = pd.DataFrame(profile_records)

print("Summary (F2 layer features):")
print(cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_summary.head())

print("\nElectron Density Profiles:")
print(cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_profiles.head())

===================================================================================================================================================================
# This next code saves the dataframes to a folder
===================================================================================================================================================================
# Define the folder path
folder = 'global July 2017 AENeASoutput dataframes' # Change name as needed

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

# Define file paths
summary_path = os.path.join(folder, 'July2017_cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_summary.pkl') # Change to name of df_summary
profiles_path = os.path.join(folder, 'July2017_cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_profiles.pkl') # Change to name of df_profiles

# Save the DataFrames
cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_summary.to_pickle(summary_path) # Change name again
cov_1_5_F10_7_6_scale_3x_std_dev_Ti_Te_0_2_poten_0_1_df_profiles.to_pickle(profiles_path) # Change name again


