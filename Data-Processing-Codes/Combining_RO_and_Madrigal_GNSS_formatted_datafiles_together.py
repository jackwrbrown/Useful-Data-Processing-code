# This script can be used to combine the RO and Madrigal GNSS TEC formatted datafiles together

# Import modules
import os
import h5py
import numpy as np
from tqdm import tqdm

# Get your 3 directories set up
gnss_dir = "/rds/projects/t/themendr-j-brown/aeneas_2.0/Madrigal_downloaded_data_20170714_20170718/July_2017_Storm_Madrigal_GNSS_AENeAS_formatted_data"
ro_dir = "/rds/projects/t/themendr-j-brown/aeneas_2.0/Madrigal_downloaded_data_20170714_20170718/July2017_full_RO_formatted_data"
combined_dir = "/rds/projects/t/themendr-j-brown/aeneas_2.0/Madrigal_downloaded_data_20170714_20170718/July2017_Madrigal_GNSS_and_RO_formatted_data"

os.makedirs(combined_dir, exist_ok=True)

# Get union of all unique filenames from both folders
all_filenames = set(f for f in os.listdir(gnss_dir) if f.endswith('.h5'))
all_filenames.update(f for f in os.listdir(ro_dir) if f.endswith('.h5'))
all_filenames = sorted(all_filenames)

def load_data(h5_path):
    with h5py.File(h5_path, 'r') as hdf:
        return {
            'recID': hdf['recID'][()],
            'rxLoc': hdf['rxLoc'][()],
            'satID': hdf['satID'][()],
            'STEC': hdf['STEC'][()],
            'satLoc': hdf['satLoc'][()],
            'STEC_std': hdf['STEC_std'][()],
            'attrs': dict(hdf.attrs)
        }

for file in tqdm(all_filenames):
    gnss_path = os.path.join(gnss_dir, file)
    ro_path = os.path.join(ro_dir, file)
    combined_path = os.path.join(combined_dir, file)

    datasets = []
    attrs = None

    if os.path.exists(gnss_path):
        gnss_data = load_data(gnss_path)
        datasets.append(gnss_data)
        attrs = gnss_data['attrs']

    if os.path.exists(ro_path):
        ro_data = load_data(ro_path)
        datasets.append(ro_data)
        attrs = ro_data['attrs']  # Overwrite with RO attrs if both exist

    if not datasets:
        print(f"Skipping {file}, no data found in either source.")
        continue

    # Concatenate all available data
    combined = {
        key: np.concatenate([d[key] for d in datasets])
        for key in ['recID', 'rxLoc', 'satID', 'STEC', 'satLoc', 'STEC_std']
    }

    # Write combined output
    with h5py.File(combined_path, 'w') as hdf:
        for key in combined:
            hdf.create_dataset(key, data=combined[key])
        for key, val in attrs.items():
            hdf.attrs[key] = val

    print(f"Wrote combined file: {file}")
