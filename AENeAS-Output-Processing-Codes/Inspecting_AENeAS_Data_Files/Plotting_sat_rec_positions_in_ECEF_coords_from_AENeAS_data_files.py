# Look at the positions of the receivers and satellites in an AENeAS datafile
# Import modules
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Define the file path
file_path = '/rds/homes/j/jxb1605/elvidgsm-serene/aeneas_example_data/data_20170604_0015.h5' # Change this

# Open the HDF5 file
with h5py.File(file_path, "r") as f:
    # Access the receiver and satellite locations (assuming they are in ECEF coordinates)
    rx_loc = f['rxLoc'][:]
    sat_loc = f['satLoc'][:]

# Earth's radius (in ECEF, approximate radius of Earth in kilometers)
earth_radius = 6378.1370

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot Earth's surface (simple sphere for Earth)
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = earth_radius * np.outer(np.cos(u), np.sin(v))
y = earth_radius * np.outer(np.sin(u), np.sin(v))
z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color='b', alpha=0.2)  # Earth surface in blue with some transparency

# Loop through the receiver and satellite data (assuming they are arrays of ECEF coordinates)
for i in range(len(rx_loc)):
    # Receiver's ECEF coordinates
    x_rec, y_rec, z_rec = rx_loc[i]
    
    # Satellite's ECEF coordinates
    x_sat, y_sat, z_sat = sat_loc[i]
    
    # Plot receiver's position on Earth's surface
    ax.scatter(x_rec, y_rec, z_rec, color='r', s=20)  # Receiver in red
    
    # Plot satellite's position
    ax.scatter(x_sat, y_sat, z_sat, color='g', s=10)  # Satellite in green
    
    # Draw a line (vector) between receiver and satellite
    ax.plot([x_rec, x_sat], [y_rec, y_sat], [z_rec, z_sat], color='k', linewidth=0.5)

# Set axis labels
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')

# Adjust azimuth and elevation angles
ax.view_init(azim=90, elev=45)

# Title
ax.set_title('Receiver and Satellite Positions with Projection Lines')

# Show the plot
plt.show()

# If all is good the lines between the receivers and satellites should join, you should get multiple lines to each satellite from multiple ground recievers
# This is a good first check to see if your data looks about right, eg if you've done a coord transform correctly


