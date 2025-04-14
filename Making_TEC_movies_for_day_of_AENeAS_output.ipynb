import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import imageio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize

# Path to the folder where the output files are stored
folder_path = "/rds/projects/t/themendr-j-brown/aeneas/2wk_spinup_20250208_DT_GNSS_data/20250208_1.5_std_dev_baseline_run_(wonky_input_params_off)/"  # Change to your path

# Generate the file names based on 15-minute intervals (assuming they're named like "AENeAS_20250208_0000.hdf5", etc.)
times = [f"{hour:02d}{minute:02d}" for hour in range(24) for minute in [0, 15, 30, 45]]
file_names = [f"AENeAS_20250208_{time}.hdf5" for time in times]

# Initialise a list to store TEC frames for the video
tec_frames = []

# Initialize variables to store the min and max TEC values
tec_min = float('inf')
tec_max = float('-inf')

# Latitude and longitude values
latitude_values = np.concatenate([np.arange(-90, -87.5, 2.5),
                                  np.arange(-87.5, 87.5, 5),
                                  np.arange(87.5, 90.1, 2.5)])
longitude_values = -180 + 5 * np.arange(73)

# Function to compute TEC
def compute_tec(file_name):
    electron_density_list = []
    height_list = []

    # Loop over all 32 ensemble members (assuming they are named member0, member1, ..., member31)
    for member_idx in range(32):
        member_name = f"analysis/member{member_idx}/grids/NE"
        height_name = f"altitudes/member{member_idx}/ZG"

        with h5py.File(file_name, "r") as f:
            if member_name in f and height_name in f:
                electron_density = f[member_name][:]
                heights = f[height_name][:]
                electron_density_list.append(electron_density)
                height_list.append(heights)

    electron_density_array = np.array(electron_density_list)
    height_array = np.array(height_list)

    mean_height_m = np.mean(height_array, axis=0) * 1e3  # Now in meters
    mean_electron_density = np.mean(electron_density_array, axis=0)

    TEC = np.trapz(mean_electron_density, mean_height_m, axis=-1)
    TEC /= 1e16  # Convert to TECU
    return TEC

# Loop over each file to process TEC data and update min/max values
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    
    # Compute TEC for the current file
    TEC = compute_tec(file_path)

    # Update min and max TEC values
    tec_min = min(tec_min, TEC.min())
    tec_max = max(tec_max, TEC.max())

# Now generate the TEC map movie
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)

    # Compute TEC for the current file
    TEC = compute_tec(file_path)

    # Create the plot for the TEC map
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')

    cmap = plt.cm.plasma
    cmap = cmap(np.arange(cmap.N))
    cmap[:, -1] = 0.85
    cmap = matplotlib.colors.ListedColormap(cmap)

    # Use Normalize to set a constant color scale range
    norm = Normalize(vmin=tec_min, vmax=tec_max)

    # Plot TEC data with fixed color scale
    im = ax.imshow(TEC.T, origin="lower", cmap=cmap, aspect="auto",
                   extent=[longitude_values.min(), longitude_values.max(), latitude_values.min(), latitude_values.max()],
                   norm=norm)

    plt.colorbar(im, ax=ax, label="Total Electron Content (TECU)")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.title(f"Total Electron Content (TEC) Map - {file_name}")

    ax.set_xticks(np.linspace(longitude_values.min(), longitude_values.max(), 6))
    ax.set_yticks(np.linspace(latitude_values.min(), latitude_values.max(), 6))

    ax.grid(True)
    plt.tight_layout()

    frame_path = f"tec_frame_{file_name}.png"
    plt.savefig(frame_path)
    plt.close()

    tec_frames.append(frame_path)

# Now compile the frames into a video
with imageio.get_writer('tec_movie_with_map_fixed_color_scale.mp4', fps=2) as writer:
    for frame in tec_frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Clean up the individual frame images
for frame in tec_frames:
    os.remove(frame)

print("Movie with global map and transparent TEC created successfully with a fixed color scale!")
