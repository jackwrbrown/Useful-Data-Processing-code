# This just lets you see what's in your datafile, plot some histograms etc.

#Import modules
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define the file path
file_path = "/rds/projects/t/themendr-j-brown/aeneas_2.0/Madrigal_downloaded_data_20170714_20170718/July_2017_Storm_Madrigal_GNSS_AENeAS_formatted_data/data_20170714_0130.h5" # Change this

# Open the HDF5 file
with h5py.File(file_path, "r") as f:
    # Print the structure of the file
    def print_structure(name, obj):
        print(name, "(Group)" if isinstance(obj, h5py.Group) else "(Dataset)")

    print("File structure:")
    f.visititems(print_structure)

with h5py.File(file_path, "r") as f:
    for dataset in f.keys():
        data = f[dataset]
        print(f"{dataset}: shape {data.shape}, dtype {data.dtype}")


with h5py.File(file_path, "r") as f:
  df = pd.DataFrame({
      "STEC": np.array(f["STEC"]),
      "STEC_std": np.array(f["STEC_std"]),
      "recID": [x.decode("utf-8") for x in f["recID"]],  # Convert byte strings to normal strings
      "satID": np.array(f["satID"]),
      "rx_x": f["rxLoc"][:, 0],  # Receiver ECEF X
      "rx_y": f["rxLoc"][:, 1],  # Receiver ECEF Y
      "rx_z": f["rxLoc"][:, 2],  # Receiver ECEF Z
      "sat_x": f["satLoc"][:, 0],  # Satellite ECEF X
      "sat_y": f["satLoc"][:, 1],  # Satellite ECEF Y
      "sat_z": f["satLoc"][:, 2],  # Satellite ECEF Z
  })

# Print the STEC error spread
# Assuming your DataFrame is named df
plt.figure(figsize=(8, 6))
plt.hist(df['STEC_std'], bins=10000, edgecolor='black')
plt.title('Histogram of STEC Standard Deviation Madrigal data')
plt.xlabel('STEC_std')
plt.ylabel('Frequency')
plt.xlim(0,20)
plt.grid(True)
plt.tight_layout()
plt.show()
