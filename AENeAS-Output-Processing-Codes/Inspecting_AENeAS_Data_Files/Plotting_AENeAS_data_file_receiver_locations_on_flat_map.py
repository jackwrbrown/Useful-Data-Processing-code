# Simple code to make sure your receivers in a datafile are in pretty much the right place after coord transforms etc

import numpy as np
import h5py

# WGS84 Parameters
a = 6378.137  # Earth equatorial radius in km
e2 = 0.00669437999014  # Earth's eccentricity squared

# Function to convert ECEF to Lat/Lon
def ecef_to_latlon(ecef_coords):
    x, y, z = ecef_coords[:, 0], ecef_coords[:, 1], ecef_coords[:, 2]
    
    # Longitude calculation
    lon = np.arctan2(y, x) * (180 / np.pi)  # Convert to degrees
    
    # Iterative latitude calculation
    r = np.sqrt(x**2 + y**2)  # Distance from Z-axis
    lat = np.arctan2(z, r)  # Initial estimate
    
    for _ in range(5):  # Iterate to refine latitude
        N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        lat = np.arctan2(z + e2 * N * np.sin(lat), r)
    
    lat = np.degrees(lat)  # Convert to degrees
    
    return np.column_stack((lat, lon))

# Load HDF5 file and extract rxLoc
file_path = "/rds/homes/j/jxb1605/themendr-j-brown/DT_Feb_8_2025_aeneas_formatted_data/data_20250208_0015.h5"

with h5py.File(file_path, "r") as h5file:
    rxLoc = h5file["rxLoc"][:]  # Extract receiver locations in ECEF

# Get unique receiver locations
unique_rxLoc = np.unique(rxLoc, axis=0)

# Convert to Lat/Lon
rxLatLon = ecef_to_latlon(unique_rxLoc)

# Print the first few results
print("Converted Receiver Locations (Lat, Lon):")
print(rxLatLon[:5])

# Extract latitudes and longitudes
latitudes, longitudes = rxLatLon[:, 0], rxLatLon[:, 1]

# Create a figure and plot
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": ccrs.PlateCarree()})
ax.set_global()

# Add features
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgray")

# Plot receiver locations
ax.scatter(longitudes, latitudes, color='red', s=10, label="Receiver Locations")

# Labels and legend
ax.set_title("Receiver Locations on World Map")
ax.legend()

# Show the plot
plt.show()
