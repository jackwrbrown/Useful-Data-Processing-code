# This script is used to scan through the UCAR RO database and get RO measurements from all the RO satellites in a given timeframe

# Import modules
import os
import requests
import tarfile
from tqdm import tqdm
import shutil
import re
import numpy as np
import pandas as pd
import xarray as xr
import h5py
from datetime import datetime, timedelta

# URL to the COSMIC UCAR website
base_url = "https://data.cosmic.ucar.edu/gnss-ro/"

# All satellites in the database
satellites = [
    "champ", "cnofs", "cosmic1", "cosmic2", "fid", "geoopt",
    "gpsmet", "gpsmetas", "grace", "kompsat5", "metopa", "metopb",
    "metopc", "paz", "planetiq", "sacc", "spire", "tdx", "tsx"
]

# Days of the year you want data for, you have to do it out of 365
doys = [195, 196, 197, 198]
year = 2017

# Name the output directory
download_dir = "gnss_ro_data"
os.makedirs(download_dir, exist_ok=True)

# Function that scans the wesbite and downloads RO data
# It looks for the podTec data files which include the TEC data from LEO sats
def check_and_download(sat, proc_type, doy):
    fname = f"podTec_{proc_type}_{year}_{doy:03d}.tar.gz"
    url = f"{base_url}{sat}/{proc_type}/level1b/{year}/{doy:03d}/{fname}"
    local_path = os.path.join(download_dir, sat, proc_type, "level1b", f"{year}", f"{doy:03d}")
    os.makedirs(local_path, exist_ok=True)
    full_local_path = os.path.join(local_path, fname)

    try:
        r = requests.head(url, timeout=10)
        if r.status_code != 200:
            return False
    except Exception as e:
        print(f"Failed to access {url}: {e}")
        return False

    # Download the file
    print(f"Downloading {url}")
    try:
        r = requests.get(url, stream=True)
        with open(full_local_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

    # Extract it
    try:
        print(f"Extracting {full_local_path}")
        with tarfile.open(full_local_path, "r:gz") as tar:
            tar.extractall(path=local_path)
        os.remove(full_local_path)  # Remove .tar.gz after extraction
    except Exception as e:
        print(f"Failed to extract {full_local_path}: {e}")
        return False

    return True

# Main loop with satellite-level success tracking
for sat in satellites:
    satellite_root_dir = os.path.join(download_dir, sat)
    sat_had_data = False  # Track whether *any* data was downloaded for this satellite

    # It scans for reprocessed data first as this is usually better quality
    for doy in doys:
        target_folder = os.path.join(satellite_root_dir, "repro2021", "level1b", f"{year}", f"{doy:03d}")
        success = check_and_download(sat, "repro2021", doy)

    # If it can't find reprocessed data it looks for the postprocessed data instead
        if not success:
            print(f"No repro for {sat} DOY {doy}, trying postProc...")
            success = check_and_download(sat, "postProc", doy)
            target_folder = os.path.join(satellite_root_dir, "postProc", "level1b", f"{year}", f"{doy:03d}")

        if success:
            sat_had_data = True
        elif os.path.exists(target_folder) and not os.listdir(target_folder):
            print(f"Deleting empty folder: {target_folder}")
            shutil.rmtree(target_folder)

    # Delete satellite folder if it had no data at all
    if not sat_had_data and os.path.exists(satellite_root_dir):
        print(f"Deleting satellite folder (no data found): {satellite_root_dir}")
        shutil.rmtree(satellite_root_dir)

# Once we have downloaded the data we can format it into a format that AENeAS can use

print("Moving to process RO data into AENeAS format")

# --- Base directory where all the satellite data is stored ---
data_dir = 'gnss_ro_data'  # Replace with actual path if needed
output_dir = '/rds/projects/t/themendr-j-brown/aeneas_2.0/Madrigal_downloaded_data_20170714_20170718/July2017_full_RO_formatted_data' # Change to where you want to store the formatted data

# Change the times to suit your interval
start_time = pd.Timestamp('2017-07-14 00:00:00')
end_time = pd.Timestamp('2017-07-18 00:00:00')
bin_delta = pd.Timedelta(minutes=15)

# --- Create output directory ---
os.makedirs(output_dir, exist_ok=True)

# --- Recursively find all _nc files ---
nc_files = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('_nc'):
            nc_files.append(os.path.join(root, file))

print(f"Found {len(nc_files)} '_nc' files.")

# --- Function to process each file ---
def process_cosmic_ro_file(file_path):
    filename = os.path.basename(file_path)

    # This is what the filename is split into
    pattern = r'^podTec_(?P<recID>[^.]+)\.(?P<year>\d{4})\.(?P<doy>\d{3})\.(?P<hour>\d{2})\.(?P<minute>\d{2})\.\d+\.(?P<gnss_id>G\d+)\.\d+'
    match = re.match(pattern, filename)
    if not match:
        print(f"Skipping file (filename mismatch): {filename}")
        return None

    recID = match.group("recID")
    year = int(match.group("year"))
    doy = int(match.group("doy"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    gnss_id = match.group("gnss_id")
    satID = int(gnss_id[1:])

    start_time = datetime(year, 1, 1) + timedelta(days=doy - 1, hours=hour, minutes=minute)

    # The LEO satellite acts as the receiver and the GPS sat as the normal satellite
    try:
        ds = xr.open_dataset(file_path)
        dt = np.array([datetime(1980, 1, 6) + timedelta(seconds=float(t)) for t in ds['time'].values]) # 1980 is when they start counting GPS time
        df = pd.DataFrame({
            'DateTime': dt,
            'STEC': ds['TEC'].values,
            'X_rec': ds['x_LEO'].values, # To format in the way AENeAS needs you call it X_rec
            'Y_rec': ds['y_LEO'].values,
            'Z_rec': ds['z_LEO'].values,
            'X_sat': ds['x_GPS'].values,
            'Y_sat': ds['y_GPS'].values,
            'Z_sat': ds['z_GPS'].values,
            'recID': recID,
            'satID': satID,
            'FileStartTime': start_time
        })
        df['STEC_std'] = 3.5 # Set a standard error to 3.5 TECU (DT says this is appropriate)
        return df
    except Exception as e:
        print(f"Error in {filename}: {e}")
        return None

# --- Process all files ---
all_dfs = []
for file in tqdm(sorted(nc_files)):
    df = process_cosmic_ro_file(file)
    if df is not None:
        all_dfs.append(df)

if not all_dfs:
    print("No valid _nc files processed.")
    exit()

big_df = pd.concat(all_dfs, ignore_index=True)
print(f"Total rows loaded: {len(big_df)}")

# --- Bin and save into 15-minute files ---
current_start = start_time
while current_start < end_time:
    current_end = current_start + bin_delta
    mask = (big_df['DateTime'] >= current_start) & (big_df['DateTime'] < current_end)
    binned_df = big_df.loc[mask]

    if not binned_df.empty:
        filename = f"data_{current_end.strftime('%Y%m%d_%H%M')}.h5"
        output_path = os.path.join(output_dir, filename)

        with h5py.File(output_path, 'w') as hdf:
            hdf.create_dataset('recID', data=binned_df['recID'].values.astype('S'))
            hdf.create_dataset('rxLoc', data=binned_df[['X_rec', 'Y_rec', 'Z_rec']].values)
            hdf.create_dataset('satID', data=binned_df['satID'].values.astype('int32'))
            hdf.create_dataset('STEC', data=binned_df['STEC'].values.astype('float64'))
            hdf.create_dataset('satLoc', data=binned_df[['X_sat', 'Y_sat', 'Z_sat']].values)
            hdf.create_dataset('STEC_std', data=binned_df['STEC_std'].values.astype('float64'))

            hdf.attrs['Date'] = current_start.strftime('%Y-%m-%d')
            hdf.attrs['Hour'] = current_start.hour
            hdf.attrs['TimeWindow'] = f"{current_start} - {current_end}"

        print(f"Saved {len(binned_df)} rows to {filename}")

    current_start = current_end
