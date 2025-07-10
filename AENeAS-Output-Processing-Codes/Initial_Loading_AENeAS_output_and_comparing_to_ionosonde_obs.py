# Code used to load in AENeAS outputat different stations to compare to ionosonde data

import sys
import os
import h5py
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Add AENeAS tools to the path
sys.path.append('/rds/projects/t/themendr-j-brown/aeneas_2.0/aeneas')
from aeneas.denGrid import aeneas2dg, dengrid

# ----------------------------
# Station locations
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
# ----------------------------
run_prefix = "july2017storm"
base_path = "/rds/projects/t/themendr-j-brown/aeneas_2.0/aeneas"
output_base = os.path.join(base_path, run_prefix, "output") # change this if files move to new location

start_date = datetime(2017, 7, 14)
num_days = 3

times = [f"{hour:02d}{minute:02d}" for hour in range(24) for minute in [0, 15, 30, 45]]
alt = np.arange(100, 600, 5)  # Altitudes from 100 km to 595 km

summary_results = []
profile_records = []

# ----------------------------
# Main processing function
# ----------------------------
def process_source(data, label, alon, alat, aalt, combined_time):
    """Process a single NE dataset for all ionosonde stations."""
    electron_densities = dengrid(data, alon, alat, aalt,
                                 name='electron_densities',
                                 longname='Electron density grid',
                                 units='em-3')

    for station in ionosonde_stations:
        station_name = station["name"]
        lon_iono = station["lon"]
        lat_iono = station["lat"]

        try:
            ne_profile = [electron_densities.getColumn(lon_iono, lat_iono, h, order=3)[0] for h in alt]
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

        peak_idx = np.argmax(ne_profile)
        hmF2 = alt[peak_idx]
        NmF2 = ne_profile[peak_idx]
        foF2 = (NmF2 / 1.24e10) ** 0.5

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

            analysis_mean = aeneas2dg(file_path, "NE").dengrid
            alon = aeneas2dg(file_path, "longitude").dengrid
            alat = aeneas2dg(file_path, "latitude").dengrid
            aalt = aeneas2dg(file_path, "ZG").dengrid

            with h5py.File(file_path, "r") as f:
                background_mean = f["background/mean/NE"][:]
                analysis_std = f["analysis/std/NE"][:]

            process_source(analysis_mean, "Analysis Mean", alon, alat, aalt, combined_time)
            process_source(background_mean, "Background Mean", alon, alat, aalt, combined_time)
            process_source(analysis_std, "Analysis Std", alon, alat, aalt, combined_time)

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
# ----------------------------
df_summary = pd.DataFrame(summary_results)
df_profiles = pd.DataFrame(profile_records)

print("Summary (F2 layer features):")
print(df_summary.head())

print("\nElectron Density Profiles:")
print(df_profiles.head())

######################################################################################################
# Loading in ionosonde data in the format DT sent me
import pandas as pd
import os
from glob import glob

# Path to the ionosonde data folder
ionosonde_dir = "/rds/projects/t/themendr-j-brown/aeneas_2.0/July2017Storm_Ionosonde_data/" # change depending on path

# Find all .txt files in the folder
ionosonde_files = glob(os.path.join(ionosonde_dir, "*.txt"))

# Dictionary to hold each DataFrame
ionosonde_data = {}

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
    df.set_index("datetime", inplace=True)
    
    ionosonde_data[station_name] = df

# Example: print the first few rows from one station
for station, df in ionosonde_data.items():
    print(f"\n📡 {station} station data:")
    print(df.head())
    break

#################################################################################################################
# Plotting the fof2 of the loaded in AENeAS output against the ionosonde data

import matplotlib.pyplot as plt
import os

def plot_fof2_for_station(df_summary, ionosonde_files, columns):
    """
    Plot foF2 for model analysis, background, ensemble members, and optionally ionosonde data.

    Parameters:
    - ff_df_summary: DataFrame with model foF2 data
    - ionosonde_files: list of ionosonde data file paths
    - columns: column names for ionosonde data file
    """
    
    # Ask user for station name
    station_name_input = input("Enter the name of the station you want to plot (e.g., FAIRFORD): ").upper()

    # Filter model data for that station
    df_station = df_summary[df_summary["Station"] == station_name_input]

    if df_station.empty:
        print(f"No model data found for station: {station_name_input}")
        return

    df_analysis = df_station[df_station["Source"] == "Analysis Mean"]
    df_background = df_station[df_station["Source"] == "Background Mean"]

    # Plot
    plt.figure(figsize=(12, 6))

    # Main model lines
    plt.plot(df_analysis["Datetime"], df_analysis["foF2 (MHz)"], marker="o", linestyle="-", color="royalblue", label="foF2 (Analysis)")
    plt.plot(df_background["Datetime"], df_background["foF2 (MHz)"], marker="s", linestyle="--", color="orange", label="foF2 (Background)")

    # Ionsonde overlay if available
    matching_files = [file for file in ionosonde_files if station_name_input.lower() in os.path.basename(file).lower()]
    if not matching_files:
        print(f"No ionosonde file found for {station_name_input}")
    else:
        file_path = matching_files[0]
        print(f"Loading ionosonde data from {file_path}...")
        df = pd.read_csv(
            file_path,
            sep=r"\s+",
            names=columns,
            header=1,
            na_values=["---"]
        )
        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M:%S", errors="coerce")
        df.set_index("datetime", inplace=True)
        plt.plot(df.index, df["foF2"], label=f"{station_name_input} Ionosonde", color="red", linewidth=2)

    # Ensemble members
    for i in range(32):
        member_label = f"Member {i}"
        df_member = df_station[df_station["Source"] == member_label]
        plt.plot(df_member["Datetime"], df_member["foF2 (MHz)"],
                 color='lightblue', linewidth=1, alpha=0.3)

    # Final styling
    plt.title(f"foF2 Over Time at {station_name_input}", fontsize=16)
    plt.xlabel("Time (UTC)", fontsize=14)
    plt.ylabel("foF2 (MHz)", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

########################################################################################################################

# Looking at contour plots of electron density with altitude and time on the axes
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Convert the electron density into a scalar value as it's initially stored as an array
df_profiles["Electron Density (el/m³)"] = df_profiles["Electron Density (el/m³)"].apply(lambda x: float(x[0]) if isinstance(x, (list, np.ndarray)) else float(x))

# Pivot for each data type, so you can make a contour plot
pivot_analysis = df_profiles[df_profiles["Source"] == "Analysis Mean"].pivot(
    index="Altitude (km)", columns="Datetime", values="Electron Density (el/m³)"
)

pivot_background = df_profiles[df_profiles["Source"] == "Background Mean"].pivot(
    index="Altitude (km)", columns="Datetime", values="Electron Density (el/m³)"
)

pivot_std = df_profiles[df_profiles["Source"] == "Analysis Std"].pivot(
    index="Altitude (km)", columns="Datetime", values="Electron Density (el/m³)"
)

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Set up shared parameters
cmap = "jet"
date_formatter = mdates.DateFormatter('%H:%M')

# Plot: Analysis Mean
im1 = axs[0].contourf(pivot_analysis.columns, pivot_analysis.index, pivot_analysis.values, levels=5000, cmap=cmap)
axs[0].set_title("Electron Density (Analysis Mean)")
axs[0].set_ylabel("Altitude (km)")
fig.colorbar(im1, ax=axs[0], label="el/m³")

# Plot: Background Mean
im2 = axs[1].contourf(pivot_background.columns, pivot_background.index, pivot_background.values, levels=5000, cmap=cmap)
axs[1].set_title("Electron Density (Background Mean)")
axs[1].set_ylabel("Altitude (km)")
fig.colorbar(im2, ax=axs[1], label="el/m³")

# Plot: Standard Deviation
im3 = axs[2].contourf(pivot_std.columns, pivot_std.index, pivot_std.values, levels=5000, cmap=cmap)
axs[2].set_title("Electron Density (Standard Deviation)")
axs[2].set_ylabel("Altitude (km)")
axs[2].set_xlabel("Time (UTC)")
fig.colorbar(im3, ax=axs[2], label="el/m³")

# Format x-axis as time
for ax in axs:
    ax.xaxis.set_major_formatter(date_formatter)
    ax.grid(True)

plt.tight_layout()
plt.show()

##############################################################################################################

# Performing a rank histogram on the output ensemble
import numpy as np
import matplotlib.pyplot as plt

# Initialize rank bins: 33 because for 32 members, we have 33 rank intervals
rank_counts = np.zeros(33, dtype=int)

# Go through each time in the observation (might need to redo this for the July 2017 storm)
for t in day_data['datetime']:
    # Get the observation (make sure it's a scalar)
    obs_row = day_data[day_data['datetime'] == t]
    if obs_row.empty:
        continue
    obs_foF2 = obs_row["foF2"].values[0]  # Ensure this is a scalar value

    # Get ensemble member foF2 values at this time
    ensemble_vals = []
    for i in range(32):
        label = f"Member {i}"
        member_data = df_summary[(df_summary["Source"] == label) & (df_summary["Datetime"] == t)]
        if not member_data.empty:
            # Extract the scalar value from the array in 'foF2 (MHz)'
            ensemble_vals.append(member_data["foF2 (MHz)"].values[0][0])  # Extract the first element of the array

    # Only proceed if we have all 32 values
    if len(ensemble_vals) != 32:
        continue

    ensemble_vals = np.array(ensemble_vals)
    sorted_vals = np.sort(ensemble_vals)

    # Find the rank position of the observation
    rank = np.searchsorted(sorted_vals, obs_foF2)

    # Update the rank counts
    rank_counts[rank] += 1

# Plot the rank histogram
plt.figure(figsize=(10, 6))
plt.bar(range(33), rank_counts, edgecolor='black')
plt.title("Rank Histogram of fof2 values across 20250208", fontsize=16)
plt.xlabel("Rank of Observation Among Ensemble Members", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.show()
