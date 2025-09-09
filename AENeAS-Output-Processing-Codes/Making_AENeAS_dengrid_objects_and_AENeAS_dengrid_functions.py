# Here's some useful bits of code that can ease you in to using AENeAS functions and maybe how to put some together

======================================================================================================================================

# This just prints the structure of an AENeAS output file

# We're going to try and look at the output from AENeAS for the 20250208 run
# Import modules
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Path to the AENeAS HDF5 file we want to look at
file_path = "/rds/projects/t/themendr-j-brown/aeneas_2.0/aeneas/20250208_DT_GNSS_data_run/output/y2025/d039/AENeAS_20250208_1200.hdf5"

# Open the HDF5 file
with h5py.File(file_path, "r") as f:
    # Print all the keys (datasets/groups) in the file
    def print_structure(name, obj):
        print(name)

    print("HDF5 File Structure:")
    f.visititems(print_structure)

======================================================================================================================================

# This makes dengrid objects, this code just grabs the mean electron density but you can edit to get other species and individual ensemble members
# Can also grab background to compare if the analysis has actually assimilated some data!
# V USEFUL

import sys
sys.path.append('/rds/projects/t/themendr-j-brown/aeneas_2.0/aeneas') # Change this

from aeneas.denGrid import aeneas2dg
from aeneas.denGrid import dengrid
# Define the file path
aFile = "/rds/projects/t/themendr-j-brown/aeneas_2.0/aeneas/20250208_DT_GNSS_data_run/output/y2025/d039/AENeAS_20250208_0800.hdf5" # Change this

# Species for electron density, longitude, latitude, and altitude
species_ne = "NE"  # Electron density
species_lon = "longitude"
species_lat = "latitude"
species_alt = "ZG"  # Altitude ZG (This is the old way to do, with the new fixed grid things are much easier change this to "altitude" for fixed grid)

# Get the dengrid object for the specified member (electron density, longitude, latitude, altitude)
den = aeneas2dg(aFile, species_ne, mem=None, nummems=32) 
alon = aeneas2dg(aFile, species_lon)
alat = aeneas2dg(aFile, species_lat)
aalt = aeneas2dg(aFile, species_alt, mem = None, nummems = 32) # Again with the fixed grid you can get rid of the mem and nummems bits)

# Extract the actual numpy arrays from the dengrid objects (assuming they have a 'dengrid' attribute)
den_data = den.dengrid  # This should be the density data array, should have shape (72, 38, 43) for fixed grid
alon_data = alon.dengrid  # This should be the longitude data array, shape (72)
alat_data = alat.dengrid  # This should be the latitude data array, shape (38)
aalt_data = aalt.dengrid  # This should be the altitude data array, shape (43) for fixed grid

# Initialize the dengrid object for electron densities
electron_densities = dengrid(den_data, alon_data, alat_data, aalt_data, name='electron_densities', longname='Electron density grid', units = 'em-3')

# Now you can use electron_densities object for further manipulation

======================================================================================================================================

# Following on you can grab other parts of the AENeAS output eg a background member

# This opens the background mean electron density
with h5py.File(aFile, "r") as f:
    bck_mean_ne = f["background/member1/grids/NE"][:]  # Load the dataset
    bck_alt = f["altitudes/member1/ZG"][:] # Not needed with the fixed grid setup!
    print("Shape:", bck_mean_ne.shape)
    print("Data Type:", bck_mean_ne.dtype)
    print("Sample Data:", bck_mean_ne[:1])  # Print first 5 values

# Extract the actual numpy arrays from the dengrid objects (assuming they have a 'dengrid' attribute)
bck_den_data = bck_mean_ne  # This should be the density data array
aalt_data = bck_alt  # This should be the altitude data array, edit this for the fixed grid set up

# Initialize the dengrid object for background electron densities
electron_densities = dengrid(bck_den_data, alon_data, alat_data, aalt_data, name='electron_densities', longname='Electron density grid', units = 'em-3') # Uses the lat and lon from 
                                                                                                                                                         # before (same grid)

======================================================================================================================================

# Here's some funky functions you can use once you have a dengrid object set up, these are all in the AENeAS .py files like denGrid.py etc, see the documentation for each one

electron_densities.plot_map('tec', res=0.5, title='tec')

# Basic function to get an estimate of fof2 and hmf2, this is quite bad there's better ones but it's a basic starter
def summarize_ne_grid(ne_data, lon, lat, alt, time, label="analysis"):
    electron_densities = dengrid(ne_data, lon, lat, alt,
                                 name=f'{label}_electron_densities',
                                 longname=f'{label.capitalize()} Electron density grid',
                                 units='em-3')
    
    alt_km = np.arange(100, 600, 5)
    lon_iono = -1.5
    lat_iono = 51.7
    
    # Vertical profile
    ne_profile = np.array([electron_densities.getColumn(lon_iono, lat_iono, h)[0] for h in alt_km])
    
    # Store profile data
    profile_data = [{
        "Datetime": time,
        "Altitude (km)": h,
        "Electron Density (el/m³)": ne,
        "Source": label
    } for h, ne in zip(alt_km, ne_profile)]
    
    # Peak values
    peak_idx = np.argmax(ne_profile)
    hmF2 = alt_km[peak_idx]
    NmF2 = ne_profile[peak_idx]
    foF2 = (NmF2 / 1.24e10) ** 0.5
    
    summary = {
        "Datetime": time,
        "hmF2 (km)": hmF2,
        "NmF2 (el/m³)": NmF2,
        "foF2 (MHz)": foF2,
        "Source": label
    }
    
    return summary, profile_data







