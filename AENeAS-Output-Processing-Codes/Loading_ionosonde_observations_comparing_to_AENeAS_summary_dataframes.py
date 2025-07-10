# Loading in ionosonde data in the format DT sent me

# Import modules
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
# You need to load in one of the summary dataframes that you loaded in from the: 
# Loading_AENeAS_output_at_ionosonde_locations_and_saving_ionosonde_params_to_dataframes.py

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

