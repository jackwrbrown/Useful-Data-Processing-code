# This code automatically pulls all handscaled GIRO data from worldwide stations for a given time period

# It then automatically makes AENeAS data files in the documented format for ionosonde data

# Import modules
import pandas as pd
import numpy as np
import requests
import sys
import time
import os
import h5py

sys.path.append('/rds/projects/t/themendr-j-brown/aeneas/aeneas')
from coordinates import Coords

# ====================================================
# 1. Load station metadata & compute ECEF coordinates
# ====================================================
station_csv = "/rds/projects/t/themendr-j-brown/aeneas_2.0/GIRO_ionosonde_data/ionosonde_layerpeak_information/GIRO_station_locations.csv"
stations_df = pd.read_csv(station_csv, sep=",", engine="python")
stations_df.columns = stations_df.columns.str.strip()  # Clean headers

def geodetic_to_ecef(row):
    lon = row["Lon"]
    lat = row["Lat"]
    if lon > 180:
        lon -= 360
    coords = Coords([lon, lat, 0], 'GEODET')
    ecef = coords.convert('ECEF').data.squeeze()
    return pd.Series(ecef, index=["X", "Y", "Z"])

stations_df[["X", "Y", "Z"]] = stations_df.apply(geodetic_to_ecef, axis=1)

# ====================================================
# 2. GIRO query parameters
# ====================================================
start_date = "2017/07/14 00:00:00"
end_date   = "2017/07/18 00:00:00"
base_url   = "https://lgdc.uml.edu/common/DIDBGetValues"

# GIRO response parser
def parse_giro_response(response_text):
    lines = response_text.splitlines()

    # Locate start of data block
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("201"):
            data_start = i
            break
    if data_start is None:
        return pd.DataFrame()  # No data

    # GIRO column structure
    columns = ["time", "CS", "foF2", "foF2_QD", "foF1", "foF1_QD",
               "foE", "foE_QD", "hmE", "hmE_QD", "hmF2", "hmF2_QD", "hmF1", "hmF1_QD"]

    df = pd.DataFrame([line.split() for line in lines[data_start:]], columns=columns)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Convert numeric cols
    numeric_cols = ["foF2", "foF1", "foE", "hmE", "hmF2", "hmF1"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate peak electron densities [m^-3]
    df["NmF2"] = 1.24e10 * np.square(df["foF2"])
    df["NmF1"] = 1.24e10 * np.square(df["foF1"])
    df["NmE"]  = 1.24e10 * np.square(df["foE"])

    return df

# ====================================================
# 3. Loop over stations & gather rows
# ====================================================
all_rows = []
failed_stations = []

for idx, station in stations_df.iterrows():
    ursi_code = station["URSI code"]
    xyz = station[["X", "Y", "Z"]].values

    # Query GIRO API
    params = {
        "ursiCode": ursi_code,
        "charName": "foF2,foF1,foE,hmF2,hmF1,hmE",
        "DMUF": "3000",
        "fromDate": start_date,
        "toDate": end_date
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        station_df = parse_giro_response(response.text)
    except Exception as e:
        print(f"⚠️ Failed {ursi_code}: {e}")
        failed_stations.append(ursi_code)
        continue

    # If no valid data, skip
    if station_df.empty:
        print(f"⚠️ No data for {ursi_code}, skipping")
        failed_stations.append(ursi_code)
        continue

    # Collect F2, F1, and E peaks
    for _, row in station_df.iterrows():
        if not pd.isna(row["NmF2"]) and not pd.isna(row["hmF2"]):
            all_rows.append([row["time"], row["NmF2"], row["hmF2"], *xyz])
        if not pd.isna(row["NmF1"]) and not pd.isna(row["hmF1"]):
            all_rows.append([row["time"], row["NmF1"], row["hmF1"], *xyz])
        if not pd.isna(row["NmE"]) and not pd.isna(row["hmE"]):
            all_rows.append([row["time"], row["NmE"], row["hmE"], *xyz])

    print(f"✅ Processed {ursi_code}: {len(station_df)} timestamps")
    time.sleep(0.5)  # Be polite to API

# ====================================================
# 4. Build dataframe & bin into 15-min windows
# ====================================================
final_df = pd.DataFrame(all_rows, columns=["time", "elecDen", "ionoAlt", "X", "Y", "Z"])
final_df.set_index("time", inplace=True)
final_df.sort_index(inplace=True)

# Create output dir
output_dir = "/rds/projects/t/themendr-j-brown/aeneas_2.0/GIRO_ionosonde_data/July2017_AENeAS_formatted_GIRO_data"
os.makedirs(output_dir, exist_ok=True)

# Group into 15-min bins
grouped = final_df.groupby(pd.Grouper(freq="15min"))

for bin_start, df_bin in grouped:
    if df_bin.empty:
        continue

    end_time = bin_start + pd.Timedelta(minutes=15)
    output_filename = os.path.join(output_dir, f"data_{end_time.strftime('%Y%m%d_%H%M')}.h5")

    # Prepare arrays
    elecDen = df_bin["elecDen"].to_numpy().reshape(-1, 1)  # (n, m=1)
    ionoAlt = df_bin["ionoAlt"].to_numpy().reshape(-1, 1)  # (n, m=1)
    ionoLoc = df_bin[["X", "Y", "Z"]].to_numpy().reshape(-1, 3)

    # Save to HDF5
    with h5py.File(output_filename, "w") as hf:
        hf.create_dataset("elecDen", data=elecDen, dtype="float64")
        hf.create_dataset("ionoAlt", data=ionoAlt, dtype="float64")
        hf.create_dataset("ionoLoc", data=ionoLoc, dtype="float64")

    print(f"💾 Saved {output_filename} | Obs: {elecDen.shape[0]}")

# ====================================================
# 5. Summary
# ====================================================
print("\n📌 Final combined dataframe shape:", final_df.shape)
print("⚠️ Stations with no data or failed requests:", failed_stations)
