import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Folder containing the output files
folder_path = "/rds/projects/t/themendr-j-brown/aeneas/2wk_spinup_20250208_DT_GNSS_data/factory_setting_run/"

# Function to find the peak electron density (NmF2) and corresponding altitude (hmF2) very basic could get a better one
def findPeakElectronDensity(Ne, alt, hmF2_min=200.0):
    if Ne.shape != alt.shape:
        raise ValueError(f"Ne shape {Ne.shape} doesn't match alt shape {alt.shape} in findPeakElectronDensity")

    Ne_masked = np.where(alt > hmF2_min, Ne, np.nan)
    peak_indices = np.nanargmax(Ne_masked, axis=2)

    idx0, idx1 = np.indices(peak_indices.shape)
    peak_NmF2 = Ne[idx0, idx1, peak_indices]
    peak_hmF2 = alt[idx0, idx1, peak_indices]

    return peak_hmF2, peak_NmF2

# Generic extractor (use for both background and analysis)
def extract_foF2_all_members(file_path, lon_index, lat_index, group='analysis', hmF2_min=200.0):
    foF2_members = []
    hmF2_members = []

    with h5py.File(file_path, "r") as f:
        for member_idx in range(32):
            Ne_path = f"{group}/member{member_idx}/grids/NE"
            ZG_path = f"altitudes/member{member_idx}/ZG"

            Ne = f[Ne_path][:]
            ZG = f[ZG_path][:]

            hmF2, NmF2 = findPeakElectronDensity(Ne, ZG, hmF2_min)
            foF2 = (NmF2 / 1.24e10) ** 0.5 # This is a used formula in the literature

            foF2_members.append(foF2[lon_index, lat_index])
            hmF2_members.append(hmF2[lon_index, lat_index])

    foF2_members = np.array(foF2_members)
    hmF2_members = np.array(hmF2_members)

    mean_foF2 = np.mean(foF2_members)
    mean_hmF2 = np.mean(hmF2_members)

    return mean_hmF2, mean_foF2, hmF2_members, foF2_members

# Latitude and longitude indices (corrected)
lat_index = 29  # This is the lat for Fairford, UK change according to where you are analysing
lon_index = 36  # This is the lon for Fairford, UK change according to where you are analysing

# Time strings and file names (15-min interval steps for one day)
time_strings = [f"{hour:02d}{minute:02d}" for hour in range(24) for minute in [0, 15, 30, 45]]
file_names = [f"AENeAS_20250208_{ts}.hdf5" for ts in time_strings]
timestamps = [datetime.strptime(f"2025-02-08 {ts[:2]}:{ts[2:]}:00", "%Y-%m-%d %H:%M:%S") for ts in time_strings]

# Preallocate arrays for analysis
analysis_mean_foF2 = []
analysis_mean_hmF2 = []
analysis_foF2_members = []
analysis_hmF2_members = []

# Preallocate arrays for background
background_mean_foF2 = []
background_mean_hmF2 = []
background_foF2_members = []
background_hmF2_members = []

# Valid times
valid_times = []

# Loop through files
for timestamp, file_name in zip(timestamps, file_names):
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        # Analysis
        a_hmF2, a_foF2, a_hmF2_members, a_foF2_members = extract_foF2_all_members(
            file_path, lon_index, lat_index, group="analysis"
        )

        # Background
        b_hmF2, b_foF2, b_hmF2_members, b_foF2_members = extract_foF2_all_members(
            file_path, lon_index, lat_index, group="background"
        )

        # Store analysis
        analysis_mean_hmF2.append(a_hmF2)
        analysis_mean_foF2.append(a_foF2)
        analysis_hmF2_members.append(a_hmF2_members)
        analysis_foF2_members.append(a_foF2_members)

        # Store background
        background_mean_hmF2.append(b_hmF2)
        background_mean_foF2.append(b_foF2)
        background_hmF2_members.append(b_hmF2_members)
        background_foF2_members.append(b_foF2_members)

        valid_times.append(timestamp)

# Convert to numpy arrays
valid_times = np.array(valid_times)  # Contains date times 

analysis_mean_foF2 = np.array(analysis_mean_foF2) # The FoF2 values of the analysis mean at a given lat and lon for 1 day
analysis_mean_hmF2 = np.array(analysis_mean_hmF2) # The HmF2 values of the analysis mean at a given lat and lon for 1 day
analysis_foF2_members = np.array(analysis_foF2_members) # The FoF2 values of the analysis ensemble members at a given lat and lon for 1 day
analysis_hmF2_members = np.array(analysis_hmF2_members) # The HmF2 values of the analysis ensemble members at a given lat and lon for 1 day

# The same arrays as above but for the background members, you have to have saved the background when running AENeAS to get these
background_mean_foF2 = np.array(background_mean_foF2)
background_mean_hmF2 = np.array(background_mean_hmF2)
background_foF2_members = np.array(background_foF2_members)
background_hmF2_members = np.array(background_hmF2_members)
